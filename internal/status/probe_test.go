package status

import (
	"context"
	"fmt"
	"net"
	"net/http"
	"net/http/httptest"
	"path/filepath"
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

func TestNPUStatusMetadataOnly(t *testing.T) {
	digest := strings.Repeat("a", 64)
	for _, tc := range []struct {
		name, device, loadedField string
		valid, loaded             bool
	}{
		{"ready", "NPU", `,"loaded":true`, true, true},
		{"idle", "NPU", `,"loaded":false`, true, false},
		{"missing", "NPU", "", false, false},
		{"wrong-device", "CPU", `,"loaded":true`, false, false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			path := filepath.Join(t.TempDir(), "runtime.sock")
			l, err := net.Listen("unix", path)
			if err != nil {
				t.Fatal(err)
			}
			s := &http.Server{Handler: http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				if r.Method != http.MethodGet || r.URL.Path != "/status" {
					t.Errorf("unexpected %s %s", r.Method, r.URL.Path)
				}
				fmt.Fprintf(w, `{"profile":%q,"device":%q%s}`, digest, tc.device, tc.loadedField)
			})}
			go s.Serve(l)
			defer s.Close()
			i := probe(context.Background(), endpoint{provider: "openvino-npu", base: "unix://" + path})
			state := i.state("tiny", "sha256:"+digest)
			if !tc.valid {
				if state.Reachability != "metadata_error" || state.Loaded != nil || state.Listed != nil {
					t.Fatal(state)
				}
				return
			}
			if state.Reachability != "reachable" || state.Listed == nil || !*state.Listed || state.Loaded == nil || *state.Loaded != tc.loaded || state.Memory != "unknown" {
				t.Fatalf("%+v", state)
			}
			wrong := i.state("tiny", strings.Repeat("b", 64))
			if wrong.Listed == nil || *wrong.Listed || wrong.Loaded == nil || *wrong.Loaded {
				t.Fatal(wrong)
			}
		})
	}
}
