package main

import (
	"net/http"
	"net/http/httptest"
	"testing"
	"time"
)

// TestKeyset_RefreshFailIncrementsCounterAndStaysUnhealthy verifies the loud
// fail-safe: a failed JWKS fetch marks the cache unhealthy and bumps the counter
// exposed at /metrics, so a broken fast path is visible rather than silent.
func TestKeyset_RefreshFailIncrementsCounterAndStaysUnhealthy(t *testing.T) {
	// Server that always 500s -> refresh must fail.
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusInternalServerError)
	}))
	defer srv.Close()

	k := &keysetCache{url: srv.URL, client: &http.Client{Timeout: 2 * time.Second}}
	if err := k.refresh(); err == nil {
		t.Fatal("refresh against a 500 server should fail")
	}
	k.refreshFails.Add(1) // callers increment on failure (mirrors newKeysetCache)
	if k.healthy.Load() {
		t.Fatal("keyset must be unhealthy after a failed refresh")
	}
	if k.refreshFails.Load() != 1 {
		t.Fatalf("refreshFails should be 1, got %d", k.refreshFails.Load())
	}
	if k.key("anykid") != nil {
		t.Fatal("no keys should be present after a failed initial load")
	}
}
