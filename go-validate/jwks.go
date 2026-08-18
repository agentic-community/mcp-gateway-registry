package main

import (
	"crypto/rsa"
	"encoding/base64"
	"encoding/json"
	"log"
	"math/big"
	"net/http"
	"sync/atomic"
	"time"
)

// jwk is one key in a JWKS document (RSA only).
type jwk struct {
	Kty string `json:"kty"`
	Kid string `json:"kid"`
	N   string `json:"n"`
	E   string `json:"e"`
}

type jwksDoc struct {
	Keys []jwk `json:"keys"`
}

// keysetCache holds kid -> *rsa.PublicKey behind an atomic.Pointer so request
// handlers read lock-free. On refresh failure it retains the last-good keyset (B4).
type keysetCache struct {
	url     string
	keys    atomic.Pointer[map[string]*rsa.PublicKey]
	client  *http.Client
	healthy atomic.Bool
}

// key returns the public key for a kid, or nil if absent.
func (k *keysetCache) key(kid string) *rsa.PublicKey {
	m := k.keys.Load()
	if m == nil {
		return nil
	}
	return (*m)[kid]
}

// parseKey converts a JWK to an *rsa.PublicKey.
func parseKey(j jwk) (*rsa.PublicKey, error) {
	nBytes, err := base64.RawURLEncoding.DecodeString(j.N)
	if err != nil {
		return nil, err
	}
	eBytes, err := base64.RawURLEncoding.DecodeString(j.E)
	if err != nil {
		return nil, err
	}
	e := 0
	for _, b := range eBytes {
		e = e<<8 | int(b)
	}
	return &rsa.PublicKey{N: new(big.Int).SetBytes(nBytes), E: e}, nil
}

// refresh fetches the JWKS once and swaps the keyset wholesale. On any error it
// keeps the previous keyset and marks the cache unhealthy (never clears keys).
func (k *keysetCache) refresh() error {
	resp, err := k.client.Get(k.url)
	if err != nil {
		k.healthy.Store(false)
		return err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		k.healthy.Store(false)
		return errNotJWT
	}
	var doc jwksDoc
	if err := json.NewDecoder(resp.Body).Decode(&doc); err != nil {
		k.healthy.Store(false)
		return err
	}
	m := make(map[string]*rsa.PublicKey, len(doc.Keys))
	for _, j := range doc.Keys {
		if j.Kty != "RSA" {
			continue
		}
		pub, err := parseKey(j)
		if err != nil {
			continue
		}
		m[j.Kid] = pub
	}
	if len(m) == 0 {
		k.healthy.Store(false)
		return errNotJWT
	}
	k.keys.Store(&m)
	k.healthy.Store(true)
	return nil
}

// newKeysetCache loads the keyset once at startup and refreshes it in the
// background every refreshSec seconds. A startup failure is logged but not fatal:
// the sidecar still serves via fallback until the keyset loads.
func newKeysetCache(url string, refreshSec int) *keysetCache {
	k := &keysetCache{
		url:    url,
		client: &http.Client{Timeout: 5 * time.Second},
	}
	if err := k.refresh(); err != nil {
		log.Printf("WARN initial JWKS load failed (serving via fallback until it loads): %v", err)
	}
	go func() {
		ticker := time.NewTicker(time.Duration(refreshSec) * time.Second)
		defer ticker.Stop()
		for range ticker.C {
			if err := k.refresh(); err != nil {
				log.Printf("WARN JWKS refresh failed, keeping last-good keyset: %v", err)
			}
		}
	}()
	return k
}
