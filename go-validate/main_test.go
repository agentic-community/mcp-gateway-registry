package main

import (
	"net/http"
	"net/http/httptest"
	"net/http/httputil"
	"net/url"
	"testing"
	"time"
)

// blockingReader blocks forever on Read until unblocked. It stands in for the
// phantom body nginx declares (Content-Length) but never sends on an
// auth_request subrequest for a POST/PUT/PATCH origin request.
type blockingReader struct{ ch chan struct{} }

func (b blockingReader) Read(p []byte) (int, error) { <-b.ch; return 0, nil }
func (b blockingReader) Close() error               { return nil }

// TestHandleValidate_DoesNotBlockOnPhantomBody is the regression test for the
// gateway 500: a /validate subrequest that declares Content-Length but whose
// body never arrives must be answered from headers alone, not hang while the
// fallback proxy waits to copy a body that never comes.
func TestHandleValidate_DoesNotBlockOnPhantomBody(t *testing.T) {
	backend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("X-User", "alice")
		w.WriteHeader(http.StatusOK)
	}))
	defer backend.Close()
	target, _ := url.Parse(backend.URL)

	// FastPathReady=false -> handler goes straight to the fallback proxy, which is
	// exactly where the phantom body used to block.
	s := &server{
		cfg:      Config{FastPathReady: false},
		fallback: httputil.NewSingleHostReverseProxy(target),
	}

	body := blockingReader{ch: make(chan struct{})} // never unblocked
	req := httptest.NewRequest(http.MethodGet, "/validate", body)
	req.ContentLength = 40
	req.Header.Set("Content-Length", "40")
	req.Header.Set("X-Original-Method", "POST")
	rr := httptest.NewRecorder()

	done := make(chan struct{})
	go func() {
		s.handleValidate(rr, req)
		close(done)
	}()

	select {
	case <-done:
		if rr.Code != http.StatusOK {
			t.Fatalf("want 200 from fallback, got %d", rr.Code)
		}
	case <-time.After(3 * time.Second):
		t.Fatal("handleValidate blocked on a phantom request body (regression: #1652)")
	}
}

func TestExtractBearer_Precedence(t *testing.T) {
	tests := []struct {
		name    string
		xAuth   string
		auth    string
		origURL string
		wantTok string
		wantOK  bool
	}{
		{"x-authorization wins", "Bearer AAA", "Bearer BBB", "http://h/api/x", "AAA", true},
		{"authorization fallback", "", "Bearer BBB", "http://h/api/x", "BBB", true},
		{"agent path no fallback to Authorization", "", "Bearer BBB", "http://h/agent/foo", "", false},
		{"agent path uses x-authorization", "Bearer AAA", "Bearer BBB", "http://h/agent/foo", "AAA", true},
		{"no creds", "", "", "http://h/api/x", "", false},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			r := httptest.NewRequest("GET", "/validate", nil)
			if tc.xAuth != "" {
				r.Header.Set("X-Authorization", tc.xAuth)
			}
			if tc.auth != "" {
				r.Header.Set("Authorization", tc.auth)
			}
			r.Header.Set("X-Original-URL", tc.origURL)
			tok, ok := extractBearer(r)
			if tok != tc.wantTok || ok != tc.wantOK {
				t.Fatalf("got (%q,%v) want (%q,%v)", tok, ok, tc.wantTok, tc.wantOK)
			}
		})
	}
}

func TestCanonicalAuthMethod(t *testing.T) {
	for _, m := range []string{"keycloak", "cognito", "entra", "okta", "auth0", "pingfederate", "self_signed", "session_cookie", "jwt", "boto3"} {
		if got := canonicalAuthMethod(m); got != "oauth2" {
			t.Errorf("canonicalAuthMethod(%q)=%q, want oauth2", m, got)
		}
	}
	for _, m := range []string{"federation-static", "network-trusted", "weird"} {
		if got := canonicalAuthMethod(m); got != m {
			t.Errorf("canonicalAuthMethod(%q)=%q, want passthrough", m, got)
		}
	}
}

func TestServerNameFromOriginalURL(t *testing.T) {
	cases := map[string]string{
		"http://localhost/currenttime/mcp":     "currenttime",
		"http://localhost/api/tokens/generate": "",
		"http://localhost/api":                 "",
		"http://localhost/":                    "",
		"http://localhost/onlyserver":          "onlyserver",
		"":                                     "",
	}
	for in, want := range cases {
		if got := serverNameFromOriginalURL(in); got != want {
			t.Errorf("serverNameFromOriginalURL(%q)=%q, want %q", in, got, want)
		}
	}
}

func TestValidateSecretKey(t *testing.T) {
	if validateSecretKey("this-is-a-strong-32char-secret-value!!") != "" {
		t.Error("strong key should pass")
	}
	for _, weak := range []string{"", "   ", "secret", "changeme", "short"} {
		if validateSecretKey(weak) == "" {
			t.Errorf("weak/empty key %q should be rejected", weak)
		}
	}
}

func TestRecordEnabled_FailClosed(t *testing.T) {
	if !recordEnabled(nil) {
		t.Error("absent enabled -> active (backward compat)")
	}
	if !recordEnabled(true) {
		t.Error("enabled=true -> active")
	}
	for _, v := range []any{false, "true", "false", 0, 1, ""} {
		if recordEnabled(v) {
			t.Errorf("enabled=%v (non-true) must be treated as disabled", v)
		}
	}
}

func TestScopeSnapshot_MapGroupsToScopes(t *testing.T) {
	snap := &scopeSnapshot{
		scopes: []scopeDoc{
			{ID: "registry-admins", GroupMappings: []string{"admins"}},
			{ID: "mcp-servers-unrestricted/read", GroupMappings: []string{"admins", "users"}},
			{ID: "mcp-servers-unrestricted/execute", GroupMappings: []string{"admins"}},
			{ID: "unrelated", GroupMappings: []string{"nobody"}},
		},
	}
	got := snap.mapGroupsToScopes([]string{"admins"})
	want := []string{"registry-admins", "mcp-servers-unrestricted/read", "mcp-servers-unrestricted/execute"}
	if len(got) != len(want) {
		t.Fatalf("got %v want %v", got, want)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("order mismatch: got %v want %v", got, want)
		}
	}
	if len(snap.mapGroupsToScopes(nil)) != 0 {
		t.Error("empty groups -> no scopes")
	}
}

func TestScopeResolver_Resolve(t *testing.T) {
	snap := &scopeSnapshot{
		scopes: []scopeDoc{
			{ID: "s-read", GroupMappings: []string{"g1"}},
			{ID: "s-admin", GroupMappings: []string{"m2m-grp"}},
		},
		m2mGroups: map[string][]string{"svc-client": {"m2m-grp"}},
	}
	r := &scopeResolver{}
	r.snap.Store(snap)

	// Case A: token has groups -> map directly.
	if got, ok := r.resolve([]string{"g1"}, "anyclient"); !ok || len(got) != 1 || got[0] != "s-read" {
		t.Fatalf("case A got (%v,%v)", got, ok)
	}
	// Case B: empty groups + known M2M client -> enrich then map.
	if got, ok := r.resolve(nil, "svc-client"); !ok || len(got) != 1 || got[0] != "s-admin" {
		t.Fatalf("case B got (%v,%v)", got, ok)
	}
	// Case C: empty groups + unknown client -> fall back to Python.
	if _, ok := r.resolve(nil, "unknown"); ok {
		t.Fatal("case C: unknown client must not resolve (fallback)")
	}
	// user-generated sentinel is never treated as M2M.
	if _, ok := r.resolve(nil, userGeneratedClientID); ok {
		t.Fatal("user-generated sentinel must not resolve as M2M")
	}
}
