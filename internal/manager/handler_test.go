package manager

import (
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func fixture(base, role string) Model {
	return Model{Name: "local", Provider: "ollama", BaseURL: base, Model: "small:v1", Role: role, Digest: strings.Repeat("1", 64)}
}

func perform(h http.Handler, method, path, data string) *httptest.ResponseRecorder {
	w := httptest.NewRecorder()
	r := httptest.NewRequest(method, path, strings.NewReader(data))
	r.Header.Set("Authorization", "Bearer never-forward")
	h.ServeHTTP(w, r)
	return w
}

func mustNew(t *testing.T, m Model) http.Handler {
	t.Helper()
	h, err := New(Config{Models: []Model{m}})
	if err != nil {
		t.Fatal(err)
	}
	return h
}

func TestConfigAndInventory(t *testing.T) {
	m := fixture("http://ollama:11434", "chat")
	external := Model{Name: "hosted", Provider: "openai-compatible", BaseURL: "https://provider.example/v1", Model: "snapshot-1", Role: "chat", Revision: "snapshot-1", KeyEnv: "PROJECT_MODEL_TEST_TOKEN"}
	t.Setenv(external.KeyEnv, "secret")
	config := Config{Models: []Model{m, external}}
	payload, _ := json.Marshal(config)
	path := filepath.Join(t.TempDir(), "models.json")
	if err := os.WriteFile(path, payload, 0600); err != nil {
		t.Fatal(err)
	}
	loaded, err := LoadConfig(path)
	if err != nil {
		t.Fatal(err)
	}
	h, err := New(loaded)
	if err != nil {
		t.Fatal(err)
	}
	w := perform(h, "GET", "/v1/models", "")
	if w.Code != 200 || !strings.Contains(w.Body.String(), m.ID()) || !strings.Contains(w.Body.String(), external.ID()) {
		t.Fatal(w.Code, w.Body)
	}
	for _, s := range []string{"PROJECT_MODEL", "secret", "ollama:11434", "provider.example"} {
		if strings.Contains(w.Body.String(), s) {
			t.Fatal("private inventory")
		}
	}
	for _, f := range []func(*Model){
		func(m *Model) { m.Provider = "openai-compatible" }, func(m *Model) { m.BaseURL += "2" }, func(m *Model) { m.Model += "2" }, func(m *Model) { m.Role = "embeddings" }, func(m *Model) { m.Digest = strings.Repeat("2", 64) }, func(m *Model) { m.Revision = "another" },
	} {
		changed := m
		f(&changed)
		if m.ID() == changed.ID() {
			t.Fatal("identity collision")
		}
	}
	for _, f := range []func(*Model){func(m *Model) { m.BaseURL = "http://example.org" }, func(m *Model) { m.Model = "m:v2-cloud" }, func(m *Model) { m.Digest = "abc" }, func(m *Model) { m.KeyEnv = "KEY" }, func(m *Model) { m.Role = "completion" }} {
		bad := m
		f(&bad)
		if _, err := New(Config{[]Model{bad}}); err == nil {
			t.Errorf("accepted invalid %v", bad)
		}
	}
	for _, base := range []string{"ftp://example.com/v1", "https://token@provider.example/v1", "https://provider.example/v1?key=value", "https://provider.example/v1#fragment"} {
		bad := external
		bad.BaseURL = base
		if _, err := New(Config{[]Model{bad}}); err == nil {
			t.Errorf("accepted %v", base)
		}
	}
	external.BaseURL = "http://vllm:8000/v1"
	if _, err := New(Config{[]Model{external}}); err != nil {
		t.Fatal(err)
	}
	external.Revision = ""
	if _, err := New(Config{[]Model{external}}); err == nil {
		t.Error("accepted unversioned provider")
	}
}

func chatRequest(id string) string {
	return fmt.Sprintf(`{"model":%q,"messages":[{"role":"user","content":"hello"}],"stream":false,"response_format":{"type":"json_schema","json_schema":{"name":"result","strict":true,"schema":{"type":"object","additionalProperties":false}}},"max_completion_tokens":42}`, id)
}

func TestLocalChatBinding(t *testing.T) {
	for _, scenario := range []string{"ok", "cloud", "changed", "incomplete", "length", "tool", "remote-response"} {
		t.Run(scenario, func(t *testing.T) {
			var paths []string
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				paths = append(paths, r.URL.Path)
				if r.Header.Get("Authorization") != "" {
					t.Error("caller token reached local runtime")
				}
				switch r.URL.Path {
				case "/api/tags":
					digest := strings.Repeat("1", 64)
					if scenario == "changed" && len(paths) == 3 {
						digest = strings.Repeat("2", 64)
					}
					remote := ""
					if scenario == "cloud" {
						remote = "hosted-model"
					}
					fmt.Fprintf(w, `{"models":[{"name":"small:v1","digest":%q,"remote_model":%q}]}`, digest, remote)
				case "/api/chat":
					var req struct {
						Model    string
						Stream   bool
						Truncate *bool
						Shift    *bool
						Format   map[string]any
						Options  map[string]int
						Messages []struct{ Content string }
					}
					if err := json.NewDecoder(r.Body).Decode(&req); err != nil || req.Model != "small:v1" || req.Stream || req.Truncate == nil || *req.Truncate || req.Shift == nil || *req.Shift || req.Format["additionalProperties"] != false || req.Options["num_predict"] != 42 || len(req.Messages) != 1 || req.Messages[0].Content != "hello" {
						t.Errorf("bad Ollama request %+v %v", req, err)
					}
					done := scenario != "incomplete"
					reason := "stop"
					if scenario == "length" {
						reason = "length"
					}
					tool := ""
					if scenario == "tool" {
						tool = `,"tool_calls":[{}]`
					}
					remoteHost := ""
					if scenario == "remote-response" {
						remoteHost = "upstream"
					}
					fmt.Fprintf(w, `{"model":"small:v1","remote_host":%q,"done":%t,"done_reason":%q,"message":{"role":"assistant","content":"answer"%s}}`, remoteHost, done, reason, tool)
				default:
					t.Error("unexpected runtime path")
				}
			}))
			defer server.Close()
			m := fixture(server.URL, "chat")
			h := mustNew(t, m)
			for _, id := range []string{"local", "small:v1"} {
				if w := perform(h, "POST", "/v1/chat/completions", chatRequest(id)); w.Code < 400 || len(paths) > 0 {
					t.Fatal("unpinned routing", w.Code, paths)
				}
			}
			w := perform(h, "POST", "/v1/chat/completions", chatRequest(m.ID()))
			if scenario == "ok" || scenario == "length" {
				if w.Code != 200 || !strings.Contains(w.Body.String(), `"answer"`) || !strings.Contains(w.Body.String(), m.ID()) || len(paths) != 3 {
					t.Fatal(w.Code, w.Body, paths)
				}
				if scenario == "length" && !strings.Contains(w.Body.String(), `"finish_reason":"length"`) {
					t.Fatal("lost length finish reason")
				}
			} else if w.Code != 502 || strings.Contains(w.Body.String(), "answer") {
				t.Fatal(w.Code, w.Body)
			}
			if scenario == "cloud" && len(paths) != 1 {
				t.Fatal("cloud model called", paths)
			}
		})
	}
}

func TestLocalEmbeddings(t *testing.T) {
	for _, result := range []string{`[[1,0],[0,2]]`, `[[1,null],[0,2]]`, `[[0,0],[1,0]]`, `[[1],[1,2]]`, `[[1,2]]`} {
		t.Run(result, func(t *testing.T) {
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				if r.URL.Path == "/api/tags" {
					fmt.Fprintf(w, `{"models":[{"name":"small:v1","digest":%q}]}`, strings.Repeat("1", 64))
					return
				}
				b, _ := io.ReadAll(r.Body)
				if r.URL.Path != "/api/embed" || !strings.Contains(string(b), `"truncate":false`) || !strings.Contains(string(b), `"input":["a","b"]`) {
					t.Error(r.URL.Path, string(b))
				}
				fmt.Fprintf(w, `{"model":"small:v1","embeddings":%s}`, result)
			}))
			defer server.Close()
			m := fixture(server.URL, "embeddings")
			w := perform(mustNew(t, m), "POST", "/v1/embeddings", fmt.Sprintf(`{"model":%q,"input":["a","b"],"encoding_format":"float"}`, m.ID()))
			if result == `[[1,0],[0,2]]` {
				if w.Code != 200 || !strings.Contains(w.Body.String(), `"index":1`) {
					t.Fatal(w.Code, w.Body)
				}
			} else if w.Code != 502 {
				t.Fatal(w.Code, w.Body)
			}
		})
	}
}

func TestCompatibleRoutingAndRedirect(t *testing.T) {
	var calls int
	dst := httptest.NewServer(http.HandlerFunc(func(http.ResponseWriter, *http.Request) { t.Error("redirect followed") }))
	defer dst.Close()
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		calls++
		if r.Header.Get("Authorization") != "Bearer upstream-secret" {
			t.Error("wrong authorization")
		}
		if r.URL.Path != "/v1/chat/completions" {
			t.Error(r.URL.Path)
		}
		b, _ := io.ReadAll(r.Body)
		if !strings.Contains(string(b), `"model":"snapshot"`) || !strings.Contains(string(b), `"strict":true`) {
			t.Error(string(b))
		}
		if calls == 2 {
			http.Redirect(w, r, dst.URL, 307)
			return
		}
		fmt.Fprint(w, `{"model":"snapshot","choices":[{"index":0,"finish_reason":"length","message":{"role":"assistant","content":"text"}}]}`)
	}))
	defer server.Close()
	t.Setenv("TEST_MODEL_KEY", "upstream-secret")
	m := Model{Name: "remote", Provider: "openai-compatible", Role: "chat", Model: "snapshot", Revision: "1", KeyEnv: "TEST_MODEL_KEY", BaseURL: server.URL + "/v1"}
	h := mustNew(t, m)
	if w := perform(h, "POST", "/v1/chat/completions", chatRequest(m.ID())); w.Code != 200 || !strings.Contains(w.Body.String(), `"finish_reason":"length"`) || !strings.Contains(w.Body.String(), m.ID()) {
		t.Fatal(w.Code, w.Body)
	}
	if w := perform(h, "POST", "/v1/chat/completions", chatRequest(m.ID())); w.Code != 502 || strings.Contains(w.Body.String(), "upstream-secret") {
		t.Fatal(w.Code, w.Body)
	}
}

func TestEmptyInventory(t *testing.T) {
	h, err := New(Config{Models: []Model{}})
	if err != nil {
		t.Fatal(err)
	}
	w := perform(h, "GET", "/v1/models", "")
	if w.Code != 200 || !strings.Contains(w.Body.String(), `"data":[]`) {
		t.Fatal(w.Code, w.Body)
	}
}

func TestInboundValidation(t *testing.T) {
	runtime := httptest.NewServer(http.HandlerFunc(func(http.ResponseWriter, *http.Request) { t.Error("invalid request reached provider") }))
	defer runtime.Close()
	m := fixture(runtime.URL, "chat")
	h := mustNew(t, m)
	req := chatRequest(m.ID())
	for _, bad := range []string{strings.Replace(req, `"stream":false`, `"stream":true`, 1), strings.Replace(req, `"strict":true`, `"strict":false`, 1), strings.Replace(req, `"max_completion_tokens":42`, `"max_completion_tokens":-1`, 1), strings.Replace(req, `"stream":false`, `"url":"https://unconfigured.example"`, 1), req + " {}"} {
		if w := perform(h, "POST", "/v1/chat/completions", bad); w.Code != 400 {
			t.Error(w.Code, w.Body)
		}
	}
	if w := perform(h, "POST", "/v1/chat/completions", strings.Repeat(" ", 2<<20)+req); w.Code != 413 {
		t.Fatal(w.Code)
	}
	if w := perform(h, "POST", "/v1/embeddings", fmt.Sprintf(`{"model":%q,"input":"a"}`, m.ID())); w.Code != 404 {
		t.Fatal(w.Code)
	}
}

func TestCompatibleEmbeddingRows(t *testing.T) {
	for _, tc := range []struct {
		name, model, rows string
		status            int
	}{
		{"reordered", "snap", `[{"index":1,"embedding":[0,1]},{"index":0,"embedding":[1,0]}]`, 200},
		{"drift", "unknown", `[{"index":1,"embedding":[0,1]},{"index":0,"embedding":[1,0]}]`, 502},
		{"duplicate", "snap", `[{"index":1,"embedding":[0,1]},{"index":1,"embedding":[1,0]}]`, 502},
		{"missing-index", "snap", `[{"embedding":[0,1]},{"index":1,"embedding":[1,0]}]`, 502},
		{"dimension", "snap", `[{"index":0,"embedding":[2]},{"index":1,"embedding":[1]}]`, 502},
		{"overflow", "snap", `[{"index":0,"embedding":[1e100,1]},{"index":1,"embedding":[1,2]}]`, 502},
	} {
		t.Run(tc.name, func(t *testing.T) {
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				if r.URL.Path != "/v1/embeddings" || r.Header.Get("Authorization") != "" {
					t.Error("unexpected provider request", r.URL.Path)
				}
				var req embeddingRequest
				if err := json.NewDecoder(r.Body).Decode(&req); err != nil || len(req.Input) != 2 || req.EncodingFormat != "float" || req.Model != "alias" || req.Dimensions == nil || *req.Dimensions != 2 {
					t.Error("provider request", req, err)
				}
				fmt.Fprintf(w, `{"model":%q,"data":%s}`, tc.model, tc.rows)
			}))
			defer server.Close()
			m := Model{Name: "embed", Provider: "openai-compatible", Role: "embeddings", Model: "alias", Revision: "snap", BaseURL: server.URL + "/v1"}
			w := perform(mustNew(t, m), "POST", "/v1/embeddings", fmt.Sprintf(`{"model":%q,"input":["one","two"],"dimensions":2,"encoding_format":"float"}`, m.ID()))
			if w.Code != tc.status {
				t.Fatal(w.Code, w.Body)
			}
			if w.Code == 200 {
				var out struct{ Data []embeddingRow }
				if err := json.Unmarshal(w.Body.Bytes(), &out); err != nil || len(out.Data) != 2 || *out.Data[0].Index != 0 || out.Data[0].Embedding[0] != 1 {
					t.Errorf("embedding order: %+v", out)
				}
			}
		})
	}
}

func TestConfiguredOllamaRequiresExplicitTag(t *testing.T) {
	for _, name := range []string{"small", "registry:5000/org/small", "small:", "small:tag:other"} {
		m := fixture("http://ollama:11434", "chat")
		m.Model = name
		if _, err := New(Config{Models: []Model{m}}); err == nil {
			t.Errorf("accepted implicit or malformed tag %q", name)
		}
	}
}

func TestOllamaLatestResponseName(t *testing.T) {
	for _, role := range []string{"chat", "embeddings"} {
		t.Run(role, func(t *testing.T) {
			const base = "registry:5000/org/small"
			m := fixture("", role)
			m.Model = base + ":latest"
			s := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				switch r.URL.Path {
				case "/api/tags":
					fmt.Fprintf(w, `{"models":[{"name":%q,"digest":%q}]}`, base, m.Digest)
				case "/api/chat":
					fmt.Fprintf(w, `{"model":%q,"done":true,"done_reason":"stop","message":{"role":"assistant","content":"good"}}`, base)
				case "/api/embed":
					fmt.Fprintf(w, `{"model":%q,"embeddings":[[1]]}`, base)
				default:
					t.Errorf("unexpected path %s", r.URL.Path)
				}
			}))
			defer s.Close()
			m.BaseURL = s.URL
			path, req := "/v1/chat/completions", chatRequest(m.ID())
			if role == "embeddings" {
				path, req = "/v1/embeddings", fmt.Sprintf(`{"model":%q,"input":"a"}`, m.ID())
			}
			if w := perform(mustNew(t, m), "POST", path, req); w.Code != 200 {
				t.Fatal(w.Code, w.Body)
			}
		})
	}
}
