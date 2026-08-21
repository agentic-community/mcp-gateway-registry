package main

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func post(t *testing.T, body string) *httptest.ResponseRecorder {
	t.Helper()
	req := httptest.NewRequest(http.MethodPost, "/mcp", strings.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	rr := httptest.NewRecorder()
	handleMCP(rr, req)
	return rr
}

func decode(t *testing.T, rr *httptest.ResponseRecorder) map[string]any {
	t.Helper()
	var m map[string]any
	if err := json.Unmarshal(rr.Body.Bytes(), &m); err != nil {
		t.Fatalf("bad JSON: %v (%s)", err, rr.Body.String())
	}
	return m
}

func TestInitialize(t *testing.T) {
	rr := post(t, `{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-06-18"}}`)
	if rr.Code != 200 {
		t.Fatalf("want 200, got %d", rr.Code)
	}
	if rr.Header().Get(sessionHeader) == "" {
		t.Error("initialize must set an Mcp-Session-Id header")
	}
	res := decode(t, rr)["result"].(map[string]any)
	si := res["serverInfo"].(map[string]any)
	if si["name"] != serverName {
		t.Errorf("serverInfo.name = %v", si["name"])
	}
	if res["protocolVersion"] != "2025-06-18" {
		t.Errorf("protocolVersion echo failed: %v", res["protocolVersion"])
	}
}

func TestToolsList(t *testing.T) {
	rr := post(t, `{"jsonrpc":"2.0","id":2,"method":"tools/list"}`)
	tools := decode(t, rr)["result"].(map[string]any)["tools"].([]any)
	if len(tools) != 1 || tools[0].(map[string]any)["name"] != "echo" {
		t.Fatalf("expected one echo tool, got %v", tools)
	}
}

func TestToolsCallEcho(t *testing.T) {
	rr := post(t, `{"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":"echo","arguments":{"message":"hi-there"}}}`)
	content := decode(t, rr)["result"].(map[string]any)["content"].([]any)
	if content[0].(map[string]any)["text"] != "hi-there" {
		t.Fatalf("echo returned %v", content)
	}
}

func TestPing(t *testing.T) {
	rr := post(t, `{"jsonrpc":"2.0","id":4,"method":"ping"}`)
	if rr.Code != 200 || decode(t, rr)["result"] == nil {
		t.Fatalf("ping failed: %d %s", rr.Code, rr.Body.String())
	}
}

func TestNotificationsInitializedIs202(t *testing.T) {
	rr := post(t, `{"jsonrpc":"2.0","method":"notifications/initialized"}`)
	if rr.Code != http.StatusAccepted {
		t.Fatalf("want 202, got %d", rr.Code)
	}
}

func TestUnknownMethodIsJSONRPCError(t *testing.T) {
	rr := post(t, `{"jsonrpc":"2.0","id":9,"method":"does/not/exist"}`)
	if decode(t, rr)["error"] == nil {
		t.Fatal("unknown method must return a JSON-RPC error")
	}
}

func TestGetIs405(t *testing.T) {
	req := httptest.NewRequest(http.MethodGet, "/mcp", nil)
	rr := httptest.NewRecorder()
	handleMCP(rr, req)
	if rr.Code != http.StatusMethodNotAllowed {
		t.Fatalf("GET want 405, got %d", rr.Code)
	}
}
