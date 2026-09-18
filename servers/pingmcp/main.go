// Command pingmcp is a tiny, fast MCP server that speaks the streamable-http
// transport and exposes a single `echo` tool. It exists to be a FAST upstream
// behind the gateway so an end-to-end load test is bounded by the /validate
// auth check (Python vs the Go sidecar), not by the upstream.
//
// Design notes: stdlib only. It answers each JSON-RPC POST with a single
// application/json response (the streamable-http spec permits this for a POST
// carrying one request), which keeps it trivial and extremely fast.
package main

import (
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"io"
	"log"
	"net/http"
	"os"
)

const (
	serverName    = "pingmcp"
	serverVersion = "1.0.0"
	defaultProtoc = "2025-06-18"
	sessionHeader = "Mcp-Session-Id"
)

type rpcRequest struct {
	JSONRPC string          `json:"jsonrpc"`
	ID      json.RawMessage `json:"id"` // absent/null => notification
	Method  string          `json:"method"`
	Params  json.RawMessage `json:"params"`
}

type rpcError struct {
	Code    int    `json:"code"`
	Message string `json:"message"`
}

// newSessionID returns a random opaque session id for the initialize response.
func newSessionID() string {
	b := make([]byte, 16)
	_, _ = rand.Read(b)
	return hex.EncodeToString(b)
}

// writeResult sends a JSON-RPC success response.
func writeResult(w http.ResponseWriter, id json.RawMessage, result any) {
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(map[string]any{
		"jsonrpc": "2.0",
		"id":      rawOrNull(id),
		"result":  result,
	})
}

// writeError sends a JSON-RPC error response.
func writeError(w http.ResponseWriter, id json.RawMessage, code int, msg string) {
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(map[string]any{
		"jsonrpc": "2.0",
		"id":      rawOrNull(id),
		"error":   rpcError{Code: code, Message: msg},
	})
}

func rawOrNull(id json.RawMessage) any {
	if len(id) == 0 {
		return nil
	}
	return id
}

// toolsList is the static tool catalog (one tool: echo).
func toolsList() any {
	return map[string]any{
		"tools": []any{
			map[string]any{
				"name":        "echo",
				"description": "Echo back the provided message. Minimal tool for load testing.",
				"inputSchema": map[string]any{
					"type": "object",
					"properties": map[string]any{
						"message": map[string]any{
							"type":        "string",
							"description": "Text to echo back",
						},
					},
					"required": []string{"message"},
				},
			},
		},
	}
}

// callEcho handles tools/call for the echo tool.
func callEcho(params json.RawMessage) any {
	var p struct {
		Name      string `json:"name"`
		Arguments struct {
			Message string `json:"message"`
		} `json:"arguments"`
	}
	_ = json.Unmarshal(params, &p)
	msg := p.Arguments.Message
	if msg == "" {
		msg = "pong"
	}
	return map[string]any{
		"content": []any{
			map[string]any{"type": "text", "text": msg},
		},
		"isError": false,
	}
}

// handleMCP is the streamable-http endpoint: one JSON-RPC request per POST.
func handleMCP(w http.ResponseWriter, r *http.Request) {
	if r.Method == http.MethodGet {
		// This server does not offer a standalone SSE stream; per the
		// streamable-http spec a server MAY reject GET. Clients that only POST
		// (and read the direct JSON response) work fine.
		w.WriteHeader(http.StatusMethodNotAllowed)
		return
	}
	if r.Method != http.MethodPost {
		w.WriteHeader(http.StatusMethodNotAllowed)
		return
	}
	body, err := io.ReadAll(io.LimitReader(r.Body, 1<<20))
	if err != nil {
		writeError(w, nil, -32700, "parse error")
		return
	}
	var req rpcRequest
	if err := json.Unmarshal(body, &req); err != nil {
		writeError(w, nil, -32700, "parse error")
		return
	}

	switch req.Method {
	case "initialize":
		// Echo the client's requested protocol version when present.
		proto := defaultProtoc
		var p struct {
			ProtocolVersion string `json:"protocolVersion"`
		}
		if json.Unmarshal(req.Params, &p) == nil && p.ProtocolVersion != "" {
			proto = p.ProtocolVersion
		}
		w.Header().Set(sessionHeader, newSessionID())
		writeResult(w, req.ID, map[string]any{
			"protocolVersion": proto,
			"capabilities":    map[string]any{"tools": map[string]any{"listChanged": false}},
			"serverInfo":      map[string]any{"name": serverName, "version": serverVersion},
		})
	case "notifications/initialized":
		// A notification: acknowledge with 202 and no body.
		w.WriteHeader(http.StatusAccepted)
	case "ping":
		writeResult(w, req.ID, map[string]any{})
	case "tools/list":
		writeResult(w, req.ID, toolsList())
	case "tools/call":
		writeResult(w, req.ID, callEcho(req.Params))
	default:
		writeError(w, req.ID, -32601, "method not found: "+req.Method)
	}
}

func main() {
	port := os.Getenv("PORT")
	if port == "" {
		port = "8100"
	}
	mux := http.NewServeMux()
	// The gateway proxies /pingmcp/<x> -> http://pingmcp-server:PORT/<x>, so the
	// MCP endpoint arrives as /mcp. Handle /mcp, /mcp/, and / for robustness.
	mux.HandleFunc("/mcp", handleMCP)
	mux.HandleFunc("/mcp/", handleMCP)
	mux.HandleFunc("/", handleMCP)
	mux.HandleFunc("/health", func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte("ok\n"))
	})
	addr := ":" + port
	log.Printf("pingmcp listening on %s (streamable-http, tool=echo)", addr)
	log.Fatal(http.ListenAndServe(addr, mux))
}
