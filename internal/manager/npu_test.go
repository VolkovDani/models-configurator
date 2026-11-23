package manager

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net"
	"net/http"
	"net/http/httptest"
	"path/filepath"
	"regexp"
	"strings"
	"testing"
	"time"
)

func socketRuntime(t *testing.T, h http.HandlerFunc) string {
	t.Helper()
	path := filepath.Join(t.TempDir(), "runtime.sock")
	l, err := net.Listen("unix", path)
	if err != nil {
		t.Fatal(err)
	}
	s := &http.Server{Handler: h}
	go s.Serve(l)
	t.Cleanup(func() { s.Close() })
	return "unix://" + path
}

func npuModel(base string) Model {
	return Model{Name: "intel", Provider: "openvino-npu", BaseURL: base, Role: "chat", Model: "tiny", Digest: "sha256:" + strings.Repeat("a", 64)}
}

func TestNPUConfig(t *testing.T) {
	m := npuModel("unix:///run/models-npu/runtime.sock")
	mustNew(t, m)
	for _, base := range []string{"http://127.0.0.1", "unix://host/tmp/runtime.sock", "unix:relative", "unix:///", "unix:///tmp/../run/sock", "unix:///tmp/sock?secret", "unix:///tmp/sock#secret", "unix:///tmp/sock/"} {
		bad := m
		bad.BaseURL = base
		if bad.validate() == nil {
			t.Errorf("accepted socket %q", base)
		}
	}
	for _, change := range []func(*Model){
		func(m *Model) { m.Digest = "v1" }, func(m *Model) { m.Role = "embeddings" }, func(m *Model) { m.Revision = "v1" }, func(m *Model) { m.KeyEnv = "KEY" },
	} {
		bad := m
		change(&bad)
		if bad.validate() == nil {
			t.Error("accepted invalid NPU model", bad)
		}
	}
	raw := m
	raw.Digest = strings.TrimPrefix(m.Digest, "sha256:")
	if raw.ID() != m.ID() {
		t.Error("profile digest prefix changed identity")
	}
}

func TestNPUChatBinding(t *testing.T) {
	digest := strings.Repeat("a", 64)
	for _, scenario := range []string{"ok", "wrong-health", "wrong-device", "wrong-result", "cpu-result", "invalid-content", "runtime-error", "oversize", "missing-binding", "malformed-result"} {
		t.Run(scenario, func(t *testing.T) {
			var calls int
			base := socketRuntime(t, func(w http.ResponseWriter, r *http.Request) {
				calls++
				if r.Header.Get("Authorization") != "" {
					t.Error("leaked authorization")
				}
				switch r.Method + " " + r.URL.Path {
				case "GET /health":
					if scenario == "missing-binding" {
						io.WriteString(w, `{"device":"NPU"}`)
						return
					}
					profile, device := digest, "NPU"
					if scenario == "wrong-health" {
						profile = strings.Repeat("b", 64)
					}
					if scenario == "wrong-device" {
						device = "CPU"
					}
					fmt.Fprintf(w, `{"profile":%q,"device":%q}`, profile, device)
				case "POST /v1/generate":
					var body struct {
						Profile             string
						Messages            []message
						Schema              map[string]any
						MaxCompletionTokens int `json:"max_completion_tokens"`
					}
					d := json.NewDecoder(r.Body)
					d.DisallowUnknownFields()
					if err := d.Decode(&body); err != nil || body.Profile != digest || len(body.Messages) != 1 || body.Messages[0] != (message{"user", "hello"}) || body.Schema["type"] != "object" || body.MaxCompletionTokens != 1024 {
						t.Errorf("bad generation body %+v %v", body, err)
					}
					if scenario == "runtime-error" {
						http.Error(w, "private runtime dump", 500)
						return
					}
					if scenario == "malformed-result" {
						io.WriteString(w, "private invalid JSON")
						return
					}
					if scenario == "oversize" {
						io.WriteString(w, strings.Repeat(" ", 8<<20)+`{}`)
						return
					}
					profile, device, content := digest, "NPU", `{"answer":42}`
					if scenario == "wrong-result" {
						profile = strings.Repeat("c", 64)
					}
					if scenario == "cpu-result" {
						device = "AUTO"
					}
					if scenario == "invalid-content" {
						content = "private runtime dump"
					}
					fmt.Fprintf(w, `{"profile":%q,"device":%q,"content":%q}`, profile, device, content)
				default:
					t.Errorf("unexpected call %s %s", r.Method, r.URL.Path)
				}
			})
			m := npuModel(base)
			h := mustNew(t, m)
			body := strings.Replace(chatRequest(m.ID()), `"max_completion_tokens":42`, `"max_completion_tokens":1024`, 1)
			w := perform(h, "POST", "/v1/chat/completions", body)
			if scenario == "ok" {
				var out struct {
					Model   string
					Choices []choice
				}
				if w.Code != 200 || json.Unmarshal(w.Body.Bytes(), &out) != nil || out.Model != m.ID() || len(out.Choices) != 1 || out.Choices[0] != (choice{0, "stop", message{"assistant", `{"answer":42}`}}) {
					t.Fatal(w.Code, w.Body)
				}
			} else {
				want := 502
				if scenario == "wrong-health" || scenario == "wrong-result" || scenario == "cpu-result" || scenario == "wrong-device" {
					want = 409
				}
				if (scenario == "oversize" || scenario == "missing-binding" || scenario == "malformed-result" || scenario == "invalid-content") &&
					!strings.Contains(w.Body.String(), `"code":"INVALID_OUTPUT"`) {
					t.Error("missing invalid output diagnostic:", w.Body)
				}
				if w.Code != want || strings.Contains(w.Body.String(), "private") {
					t.Fatal(w.Code, w.Body)
				}
			}
			want := 2
			if scenario == "wrong-health" || scenario == "wrong-device" || scenario == "missing-binding" {
				want = 1
			}
			if calls != want {
				t.Errorf("got %d runtime calls, want %d", calls, want)
			}
		})
	}
}

func TestNPUUnsupportedRequests(t *testing.T) {
	m := npuModel(socketRuntime(t, func(http.ResponseWriter, *http.Request) { t.Error("unexpected runtime request") }))
	h := mustNew(t, m)
	req := chatRequest(m.ID())
	for _, body := range []string{strings.Replace(req, `"stream":false`, `"temperature":0`, 1), strings.Replace(req, `"stream":false`, `"top_p":1`, 1), fmt.Sprintf(`{"model":%q,"messages":[{"role":"user","content":"hello"}]}`, m.ID())} {
		if w := perform(h, "POST", "/v1/chat/completions", body); w.Code != 400 {
			t.Error(w.Code, w.Body)
		}
	}
	if w := perform(h, "GET", "/v1/models", ""); w.Code != 200 || !strings.Contains(w.Body.String(), m.ID()) || strings.Contains(w.Body.String(), "runtime.sock") {
		t.Error(w.Code, w.Body)
	}
	if w := perform(h, "POST", "/v1/embeddings", fmt.Sprintf(`{"model":%q,"input":"hello"}`, m.ID())); w.Code != 404 {
		t.Error(w.Code, w.Body)
	}
}

func TestNPUGenerationCancellation(t *testing.T) {
	started, stopped := make(chan struct{}), make(chan struct{})
	digest := strings.Repeat("a", 64)
	m := npuModel(socketRuntime(t, func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/health" {
			fmt.Fprintf(w, `{"profile":%q,"device":"NPU"}`, digest)
			return
		}
		io.Copy(io.Discard, r.Body)
		close(started)
		<-r.Context().Done()
		close(stopped)
	}))
	h := mustNew(t, m)
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	req := httptest.NewRequestWithContext(ctx, "POST", "/v1/chat/completions", strings.NewReader(chatRequest(m.ID())))
	done := make(chan struct{})
	go func() { h.ServeHTTP(httptest.NewRecorder(), req); close(done) }()
	select {
	case <-started:
	case <-time.After(2 * time.Second):
		t.Fatal("generation not started")
	}
	cancel()
	for _, ch := range []chan struct{}{stopped, done} {
		select {
		case <-ch:
		case <-time.After(2 * time.Second):
			t.Fatal("request did not cancel")
		}
	}
}

func TestNPUCallIDCorrelation(t *testing.T) {
	ids := make(chan string, 4)
	profile := strings.Repeat("a", 64)
	m := npuModel(socketRuntime(t, func(w http.ResponseWriter, r *http.Request) {
		ids <- r.Header.Get("X-NPU-Call-ID")
		if r.URL.Path == "/health" {
			fmt.Fprintf(w, `{"profile":%q,"device":"NPU"}`, profile)
		} else {
			fmt.Fprintf(w, `{"profile":%q,"device":"NPU","content":"{}"}`, profile)
		}
	}))
	h := mustNew(t, m)
	seen := map[string]bool{}
	for _, supplied := range []string{"private-operator-name", strings.Repeat("a", 32)} {
		req := httptest.NewRequest("POST", "/v1/chat/completions", strings.NewReader(chatRequest(m.ID())))
		req.Header.Set("X-NPU-Call-ID", supplied)
		w := httptest.NewRecorder()
		h.ServeHTTP(w, req)
		id := w.Header().Get("X-NPU-Call-ID")
		if w.Code != 200 || !regexp.MustCompile("^[a-f0-9]{32}$").MatchString(id) || seen[id] || id == supplied {
			t.Fatalf("invalid response status/ID: %d %q", w.Code, id)
		}
		seen[id] = true
		if health, generate := <-ids, <-ids; health != id || generate != id {
			t.Errorf("different IDs: response %q, health %q, generation %q", id, health, generate)
		}
	}
}

func TestRuntimeFailureDiagnostics(t *testing.T) {
	cases := []struct {
		code   string
		status int
		want   string
	}{
		{"PROFILE_CHANGED", 409, "PROFILE_CHANGED"},
		{"NPU_UNAVAILABLE", 503, "NPU_UNAVAILABLE"},
		{"NPU_WORKER_EXIT", 503, "NPU_WORKER_EXIT"},
		{"NPU_RECOVERY_FAILED", 503, "NPU_RECOVERY_FAILED"},
		{"TIMEOUT", 504, "TIMEOUT"},
		{"INVALID_OUTPUT", 502, "INVALID_OUTPUT"},
		{"PRIVATE", 503, ""},
		{"TIMEOUT", 409, ""},
		{strings.Repeat("PRIVATE", 1024), 503, ""},
	}
	for _, tc := range cases {
		m := npuModel(socketRuntime(t, func(w http.ResponseWriter, r *http.Request) {
			w.WriteHeader(tc.status)
			fmt.Fprintf(w, `{"error":"PRIVATE internal text","code":%q}`, tc.code)
		}))
		h := mustNew(t, m)
		w := perform(h, "POST", "/v1/chat/completions", chatRequest(m.ID()))
		var result struct {
			Error struct{ Message, Code string }
		}
		wantStatus := tc.status
		if tc.want == "" {
			wantStatus = 502
		}
		if json.Unmarshal(w.Body.Bytes(), &result) != nil || result.Error.Code != tc.want || w.Code != wantStatus ||
			strings.Contains(w.Body.String(), "PRIVATE") || w.Header().Get("X-NPU-Call-ID") == "" {
			t.Errorf("expected status %d and code %q; got %d %s", wantStatus, tc.want, w.Code, w.Body)
		}
	}
}

func TestMissingNPUSocketReportsUnavailable(t *testing.T) {
	m := npuModel("unix://" + filepath.Join(t.TempDir(), "private-unavailable"))
	w := perform(mustNew(t, m), "POST", "/v1/chat/completions", chatRequest(m.ID()))
	if w.Code != 503 || !strings.Contains(w.Body.String(), `"code":"NPU_UNAVAILABLE"`) || strings.Contains(w.Body.String(), "private") {
		t.Fatal(w.Code, w.Body)
	}
}

func TestOtherProviderCannotInjectNPUDiagnostics(t *testing.T) {
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if got := r.Header.Get("X-NPU-Call-ID"); got != "" {
			t.Errorf("forwarded call ID %q", got)
		}
		w.Header().Set("X-NPU-Call-ID", "private-runtime-id")
		w.WriteHeader(504)
		io.WriteString(w, `{"error":"private","code":"TIMEOUT"}`)
	}))
	defer upstream.Close()
	m := fixture(upstream.URL, "chat")
	w := perform(mustNew(t, m), "POST", "/v1/chat/completions", chatRequest(m.ID()))
	if w.Code != 502 || strings.Contains(w.Body.String(), "TIMEOUT") || strings.Contains(w.Body.String(), "private") || w.Header().Get("X-NPU-Call-ID") != "" {
		t.Fatal(w.Code, w.Header(), w.Body)
	}
}
