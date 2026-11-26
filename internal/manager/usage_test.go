package manager

import (
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"reflect"
	"strings"
	"testing"

	"github.com/VolkovDani/models-configurator/internal/history"
)

func TestProviderUsageInHistory(t *testing.T) {
	for _, tc := range []struct {
		name, provider, role, path, response, want string
	}{
		{"ollama-chat", "ollama", "chat", "/api/chat", `{"done":true,"done_reason":"stop","message":{"role":"assistant","content":"hello"},"prompt_eval_count":24,"eval_count":0}`, `{"input_tokens":24,"output_tokens":0}`},
		{"external-chat", "openai-compatible", "chat", "/v1/chat/completions", `{"choices":[{"finish_reason":"length","message":{"role":"assistant","content":"hello"}}],"usage":{"prompt_tokens":14,"completion_tokens":3,"total_tokens":17}}`, `{"input_tokens":14,"output_tokens":3}`},
		{"external-partial", "openai-compatible", "chat", "/v1/chat/completions", `{"choices":[{"finish_reason":"stop","message":{"role":"assistant","content":"hello"}}],"usage":{"prompt_tokens":"malformed","completion_tokens":3}}`, `{"output_tokens":3}`},
		{"external-missing", "openai-compatible", "chat", "/v1/chat/completions", `{"choices":[{"finish_reason":"stop","message":{"role":"assistant","content":"hello"}}],"usage":{"total_tokens":12}}`, `{}`},
		{"ollama-embed", "ollama", "embeddings", "/api/embed", `{"embeddings":[[1]],"prompt_eval_count":12}`, `{"input_tokens":12}`},
		{"external-embed", "openai-compatible", "embeddings", "/v1/embeddings", `{"data":[{"index":0,"embedding":[1]}],"usage":{"prompt_tokens":5,"total_tokens":5}}`, `{"input_tokens":5}`},
		{"external-embed-total", "openai-compatible", "embeddings", "/v1/embeddings", `{"data":[{"index":0,"embedding":[1]}],"usage":{"total_tokens":5}}`, `{"input_tokens":5}`},
	} {
		t.Run(tc.name, func(t *testing.T) {
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				if r.URL.Path == "/api/tags" {
					fmt.Fprintf(w, `{"models":[{"name":"small:v1","digest":%q}]}`, strings.Repeat("1", 64))
				} else if r.URL.Path == tc.path {
					io.WriteString(w, tc.response)
				} else {
					t.Errorf("unexpected path %s", r.URL.Path)
					w.WriteHeader(http.StatusNotFound)
				}
			}))
			defer server.Close()
			model := fixture(server.URL, tc.role)
			if tc.provider == "openai-compatible" {
				model.Provider, model.BaseURL, model.Revision, model.Digest = tc.provider, server.URL+"/v1", "test-revision", ""
			}
			var calls []history.Call
			h, err := NewWithHistory(Config{Models: []Model{model}}, 1, func(call history.Call) { calls = append(calls, call) })
			if err != nil {
				t.Fatal(err)
			}
			path, body := "/v1/chat/completions", chatRequest(model.ID())
			if tc.role == "embeddings" {
				path, body = "/v1/embeddings", fmt.Sprintf(`{"model":%q,"input":"hello"}`, model.ID())
			}
			w := perform(h, http.MethodPost, path, body)
			if w.Code != http.StatusOK || len(calls) != 1 || strings.Contains(w.Body.String(), `"usage"`) {
				t.Fatalf("status %d, response %s, calls %d", w.Code, w.Body, len(calls))
			}
			assertHistoryUsage(t, calls[0], tc.want)
		})
	}
}

func assertHistoryUsage(t *testing.T, call history.Call, want string) {
	t.Helper()
	raw, err := json.Marshal(call)
	if err != nil {
		t.Fatal(err)
	}
	var fields, expected map[string]json.RawMessage
	if err := json.Unmarshal(raw, &fields); err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal([]byte(want), &expected); err != nil {
		t.Fatal(err)
	}
	for _, name := range []string{"input_tokens", "output_tokens"} {
		if !reflect.DeepEqual(fields[name], expected[name]) {
			t.Errorf("%s = %s, want %s", name, fields[name], expected[name])
		}
	}
}
