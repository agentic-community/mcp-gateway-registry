package main

import (
	"context"
	"fmt"
	"log"
	"net/url"
	"os"
	"strings"
	"sync/atomic"
	"time"

	"go.mongodb.org/mongo-driver/bson"
	"go.mongodb.org/mongo-driver/mongo"
	"go.mongodb.org/mongo-driver/mongo/options"
)

// userGeneratedClientID is the sentinel client_id on self-signed user tokens;
// it must never be treated as an M2M client for group enrichment (mirrors
// auth_server/mongodb_groups_enrichment.py).
const userGeneratedClientID = "user-generated"

// scopeDoc is one mcp_scopes document: _id is the scope name, group_mappings is
// the list of IdP groups that grant it.
type scopeDoc struct {
	ID            string   `bson:"_id"`
	GroupMappings []string `bson:"group_mappings"`
}

// m2mClient is one idp_m2m_clients document (the M2M group-enrichment source).
type m2mClient struct {
	ClientID string      `bson:"client_id"`
	Groups   []string    `bson:"groups"`
	Enabled  interface{} `bson:"enabled"`
}

// scopeSnapshot is an immutable, atomically-swapped view of the two collections
// the scope resolution needs. Per-request resolution reads it lock-free.
type scopeSnapshot struct {
	scopes    []scopeDoc          // natural order (matches Python's cursor iteration)
	m2mGroups map[string][]string // client_id -> enriched groups (enabled records only)
}

// scopeResolver loads mcp_scopes + idp_m2m_clients into a TTL-refreshed snapshot
// and resolves a token's groups/client into the same scope set the Python
// /validate handler would produce.
type scopeResolver struct {
	client   *mongo.Client
	db       *mongo.Database
	scopesC  string
	clientsC string
	snap     atomic.Pointer[scopeSnapshot]
	ready    atomic.Bool
}

// recordEnabled mirrors _is_record_enabled: active only when `enabled` is absent
// (backward compat) or the boolean true. Any other value is disabled (fail closed).
func recordEnabled(enabled interface{}) bool {
	if enabled == nil {
		return true // field absent decodes to nil
	}
	b, isBool := enabled.(bool)
	if !isBool {
		return false
	}
	return b
}

// buildMongoURI assembles the connection string the way the registry does
// (username/password -> SCRAM-SHA-256 against authSource=admin for non-DocumentDB).
func buildMongoURI() (string, bool) {
	host := os.Getenv("DOCUMENTDB_HOST")
	if host == "" {
		return "", false // scope resolution disabled when DB not configured
	}
	port := getenv("DOCUMENTDB_PORT", "27017")
	dbName := getenv("DOCUMENTDB_DATABASE", "mcp_registry")
	user := os.Getenv("DOCUMENTDB_USERNAME")
	pass := os.Getenv("DOCUMENTDB_PASSWORD")
	params := "authSource=admin&authMechanism=SCRAM-SHA-256"
	if getenv("DOCUMENTDB_DIRECT_CONNECTION", "true") == "true" {
		params += "&directConnection=true"
	}
	if getenv("DOCUMENTDB_USE_TLS", "false") == "true" {
		params += "&tls=true"
		// DocumentDB serves a cert signed by the Amazon RDS CA, which is NOT a
		// public root, so the Mongo driver must trust the bundle explicitly (same
		// path + env the Python side uses). Without this the handshake fails with
		// "x509: certificate signed by unknown authority" and scope parity never
		// loads. The bundle is baked into the image at the default path.
		caFile := getenv("DOCUMENTDB_TLS_CA_FILE", "/app/certs/global-bundle.pem")
		params += "&tlsCAFile=" + url.QueryEscape(caFile)
	}
	if user != "" && pass != "" {
		return fmt.Sprintf("mongodb://%s:%s@%s:%s/%s?%s",
			url.QueryEscape(user), url.QueryEscape(pass), host, port, dbName, params), true
	}
	return fmt.Sprintf("mongodb://%s:%s/%s", host, port, dbName), true
}

// newScopeResolver connects to Mongo and starts a background snapshot refresh.
// Returns nil when the DB is not configured (caller then runs fast-path without
// scope parity, which means it must fall back for scope-bearing requests).
func newScopeResolver(refreshSec int) *scopeResolver {
	uri, ok := buildMongoURI()
	if !ok {
		log.Printf("scope resolver: DOCUMENTDB_HOST unset; scope parity disabled")
		return nil
	}
	ctx, cancel := context.WithTimeout(context.Background(), 8*time.Second)
	defer cancel()
	client, err := mongo.Connect(ctx, options.Client().ApplyURI(uri))
	if err != nil {
		log.Printf("scope resolver: connect failed (%v); scope parity disabled until it loads", err)
		return nil
	}
	ns := getenv("DOCUMENTDB_NAMESPACE", "default")
	r := &scopeResolver{
		client:   client,
		db:       client.Database(getenv("DOCUMENTDB_DATABASE", "mcp_registry")),
		scopesC:  "mcp_scopes_" + ns,
		clientsC: "idp_m2m_clients_" + ns,
	}
	if err := r.refresh(); err != nil {
		log.Printf("scope resolver: initial load failed (%v); will retry", err)
	}
	go func() {
		ticker := time.NewTicker(time.Duration(refreshSec) * time.Second)
		defer ticker.Stop()
		for range ticker.C {
			if err := r.refresh(); err != nil {
				log.Printf("scope resolver: refresh failed, keeping last-good snapshot: %v", err)
			}
		}
	}()
	return r
}

// refresh reloads both collections into a new snapshot and swaps it in.
func (r *scopeResolver) refresh() error {
	ctx, cancel := context.WithTimeout(context.Background(), 8*time.Second)
	defer cancel()

	scur, err := r.db.Collection(r.scopesC).Find(ctx, bson.D{})
	if err != nil {
		return err
	}
	var scopes []scopeDoc
	if err := scur.All(ctx, &scopes); err != nil {
		return err
	}

	ccur, err := r.db.Collection(r.clientsC).Find(ctx, bson.D{})
	if err != nil {
		return err
	}
	var clients []m2mClient
	if err := ccur.All(ctx, &clients); err != nil {
		return err
	}
	m2m := make(map[string][]string, len(clients))
	for _, c := range clients {
		if c.ClientID == "" || !recordEnabled(c.Enabled) {
			continue // fail-closed: disabled records never grant groups
		}
		m2m[c.ClientID] = c.Groups
	}

	r.snap.Store(&scopeSnapshot{scopes: scopes, m2mGroups: m2m})
	r.ready.Store(true)
	return nil
}

// mapGroupsToScopes returns the scope names whose group_mappings intersect the
// given groups, in scope-document order, de-duplicated (mirrors
// get_group_mappings_bulk + map_groups_to_scopes dedupe).
func (snap *scopeSnapshot) mapGroupsToScopes(groups []string) []string {
	if len(groups) == 0 {
		return nil
	}
	want := make(map[string]bool, len(groups))
	for _, g := range groups {
		if g != "" {
			want[g] = true
		}
	}
	seen := make(map[string]bool)
	var out []string
	for _, doc := range snap.scopes {
		for _, gm := range doc.GroupMappings {
			if want[gm] {
				if !seen[doc.ID] {
					seen[doc.ID] = true
					out = append(out, doc.ID)
				}
				break
			}
		}
	}
	return out
}

// resolve returns the scope set for a token, matching Python's /validate, and a
// bool for whether the fast path could resolve it. It returns ok=false (caller
// must fall back to Python) for cases it cannot replicate exactly -- e.g. a user
// token with empty groups that would need idp_user_groups enrichment.
func (r *scopeResolver) resolve(tokenGroups []string, clientID string) (scopes []string, ok bool) {
	snap := r.snap.Load()
	if snap == nil {
		return nil, false
	}
	// Case A: token carries groups -> map directly (same for user and M2M).
	if len(tokenGroups) > 0 {
		return snap.mapGroupsToScopes(tokenGroups), true
	}
	// Case B: empty groups + a real M2M client -> enrich from idp_m2m_clients.
	if clientID != "" && clientID != userGeneratedClientID {
		if groups, found := snap.m2mGroups[clientID]; found {
			return snap.mapGroupsToScopes(groups), true
		}
	}
	// Case C: empty groups, not a known M2M client -> may need idp_user_groups
	// enrichment, which we do not replicate here. Fall back to Python.
	return nil, false
}

// scopesToHeader joins scopes with a space (the X-Scopes wire format).
func scopesToHeader(scopes []string) string {
	return strings.Join(scopes, " ")
}
