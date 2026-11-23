package status

import (
	"context"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func TestOllamaInventory(t *testing.T) {
	up := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != "GET" || r.Header.Get("Authorization") != "" {
			t.Errorf("unexpected %s %s", r.Method, r.URL.Path)
		}
		switch r.URL.Path {
		case "/api/tags":
			fmt.Fprint(w, `{"models":[{"name":"local:latest"},{"model":"idle:latest"}]}`)
		case "/api/ps":
			fmt.Fprint(w, `{"models":[{"name":"local:latest","size":100,"size_vram":60}]}`)
		default:
			t.Errorf("unexpected request: %s", r.URL.Path)
		}
	}))
	defer up.Close()
	data := probe(context.Background(), endpoint{provider: "ollama", base: up.URL})
	active := data.state("local", "")
	if active.Reachability != "reachable" || active.Installed == nil || !*active.Installed || active.Loaded == nil || !*active.Loaded || active.Memory != "cpu_gpu" {
		t.Errorf("active: %+v", active)
	}
	idle := data.state("idle:latest", "")
	if idle.Installed == nil || !*idle.Installed || idle.Loaded == nil || *idle.Loaded || idle.Memory != "unknown" {
		t.Errorf("idle: %+v", idle)
	}
	missing := data.state("absent", "")
	if missing.Installed == nil || *missing.Installed {
		t.Errorf("missing: %+v", missing)
	}
}

func TestOpenAIInventoryAndUnavailableMetadata(t *testing.T) {
	t.Setenv("STATUS_TEST_KEY", "key-for-one-service")
	for _, tc := range []struct {
		body      string
		status    int
		reachable string
		listed    bool
	}{
		{`{"data":[{"id":"remote"}]}`, 200, "reachable", true},
		{`{}`, 200, "metadata_error", false},
		{`{"data":null}`, 200, "metadata_error", false},
		{strings.Repeat(" ", 2<<20) + `{"data":[]}`, 200, "metadata_error", false},
		{"PRIVATE upstream error", 404, "metadata_error", false},
	} {
		up := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			if r.URL.Path != "/v1/models" || r.Method != "GET" || r.Header.Get("Authorization") != "Bearer key-for-one-service" {
				t.Error("unexpected upstream metadata request")
			}
			w.WriteHeader(tc.status)
			fmt.Fprint(w, tc.body)
		}))
		data := probe(context.Background(), endpoint{provider: "openai-compatible", base: up.URL + "/v1/", keyEnv: "STATUS_TEST_KEY"})
		up.Close()
		s := data.state("remote", "")
		if s.Reachability != tc.reachable || s.Installed != nil || s.Loaded != nil || s.Memory != "unknown" || (s.Listed != nil) != tc.listed || s.Listed != nil && !*s.Listed {
			t.Errorf("status %d: %+v", tc.status, s)
		}
	}
}

func TestRedirectAndMissingCredential(t *testing.T) {
	t.Setenv("STATUS_TEST_KEY", "test-key")
	dest := httptest.NewServer(http.HandlerFunc(func(http.ResponseWriter, *http.Request) { t.Error("followed redirect") }))
	defer dest.Close()
	calls := 0
	up := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		calls++
		http.Redirect(w, r, dest.URL, 302)
	}))
	defer up.Close()
	e := endpoint{provider: "openai-compatible", base: up.URL, keyEnv: "STATUS_TEST_KEY"}
	if s := probe(context.Background(), e).state("m", ""); s.Reachability != "metadata_error" {
		t.Errorf("%+v", s)
	}
	t.Setenv("STATUS_TEST_KEY", "")
	if s := probe(context.Background(), e).state("m", ""); s.Reachability != "credential_missing" {
		t.Errorf("%+v", s)
	}
	if calls != 1 {
		t.Errorf("expected 1 upstream call, got %d", calls)
	}
}
