package manager

import (
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"

	"github.com/VolkovDani/models-configurator/internal/history"
)

const tokenCountPath = "/v1/chat/token-count"

func readTokenCount(t *testing.T, w *httptest.ResponseRecorder) (input, maxInput, maxCompletion int, exact bool, model string) {
	t.Helper()
	var result struct {
		Input         int   `json:"input_tokens"`
		MaxInput      int   `json:"max_input_tokens"`
		MaxCompletion int   `json:"max_completion_tokens"`
		Exact         *bool `json:"exact"`
		Model         string
	}
	if w.Code != http.StatusOK || decodeStrict(w.Body.Bytes(), &result) != nil || result.Exact == nil {
		t.Fatalf("count response: %d %s", w.Code, w.Body)
	}
	return result.Input, result.MaxInput, result.MaxCompletion, *result.Exact, result.Model
}

func TestOllamaTokenCountEstimatedFromConfiguredBudget(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(http.ResponseWriter, *http.Request) {
		t.Error("preflight contacted Ollama")
	}))
	defer server.Close()
	contextLimit, completionLimit := 2048, 128
	model := fixture(server.URL, "chat")
	model.OllamaChat = &OllamaChatSettings{ContextTokens: &contextLimit, CompletionTokens: &completionLimit}
	h, err := NewWithHistory(Config{Models: []Model{model}}, 1, func(history.Call) {
		t.Error("preflight recorded a generation")
	})
	if err != nil {
		t.Fatal(err)
	}
	base := fmt.Sprintf(`{"model":%q,"messages":[{"role":"user","content":"é𐍈"}]}`, model.ID())
	baseline, maxInput, maxCompletion, exact, id := readTokenCount(t, perform(h, http.MethodPost, tokenCountPath, base))
	if baseline <= len("é𐍈") || maxInput != 1920 || maxCompletion != 128 || exact || id != model.ID() {
		t.Fatalf("unexpected local estimate: %d %d %d %t %q", baseline, maxInput, maxCompletion, exact, id)
	}
	withSchema := strings.Replace(chatRequest(model.ID()), "hello", "é𐍈", 1)
	input, maxInput, maxCompletion, exact, _ := readTokenCount(t, perform(h, http.MethodPost, tokenCountPath, withSchema))
	if input < baseline+len(`{"type":"object","additionalProperties":false}`) || maxInput != 2006 || maxCompletion != 42 || exact {
		t.Fatalf("schema or override missing from estimate: %d %d %d %t", input, maxInput, maxCompletion, exact)
	}
	long := strings.Replace(base, "é𐍈", strings.Repeat("é", 1400), 1)
	input, maxInput, _, _, _ = readTokenCount(t, perform(h, http.MethodPost, tokenCountPath, long))
	if input <= maxInput || input-baseline != len(strings.Repeat("é", 1400))-len("é𐍈") {
		t.Fatalf("long input clamped or UTF-8 length lost: %d, max %d, baseline %d", input, maxInput, baseline)
	}
	tooMuchOutput := strings.Replace(withSchema, `"max_completion_tokens":42`, `"max_completion_tokens":2048`, 1)
	if w := perform(h, http.MethodPost, tokenCountPath, tooMuchOutput); w.Code != 501 || !strings.Contains(w.Body.String(), `"code":"TOKEN_COUNT_UNSUPPORTED"`) {
		t.Fatalf("no valid reserve: %d %s", w.Code, w.Body)
	}
}

func TestTokenCountUnavailableAndRequestValidation(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(http.ResponseWriter, *http.Request) {
		t.Error("unsupported or invalid preflight contacted a provider")
	}))
	defer server.Close()
	model := fixture(server.URL, "chat")
	external := Model{Name: "remote", Provider: "openai-compatible", BaseURL: server.URL + "/v1", Model: "upstream", Revision: "pinned", Role: "chat"}
	npu := npuModel(socketRuntime(t, func(http.ResponseWriter, *http.Request) {
		t.Error("invalid preflight reached the NPU runtime")
	}))
	embed := fixture(server.URL, "embeddings")
	embed.Name = "embed"
	h, err := New(Config{Models: []Model{model, external, npu, embed}})
	if err != nil {
		t.Fatal(err)
	}
	for _, m := range []Model{model, external} {
		body := strings.Replace(chatRequest(m.ID()), "hello", "private-draft", 1)
		w := perform(h, http.MethodPost, tokenCountPath, body)
		if w.Code != 501 || !strings.Contains(w.Body.String(), `"code":"TOKEN_COUNT_UNSUPPORTED"`) || strings.Contains(w.Body.String(), "private-draft") {
			t.Fatalf("preflight for %s: %d %s", m.Name, w.Code, w.Body)
		}
	}
	for _, id := range []string{model.Name, model.Model, embed.ID()} {
		if w := perform(h, http.MethodPost, tokenCountPath, chatRequest(id)); w.Code != 404 {
			t.Errorf("preflight routed an unknown chat profile: %d %s", w.Code, w.Body)
		}
	}
	npuRequest := chatRequest(npu.ID())
	for _, body := range []string{
		strings.Replace(npuRequest, `"stream":false`, `"stream":true`, 1),
		strings.Replace(npuRequest, `"stream":false`, `"temperature":0`, 1),
		strings.Replace(npuRequest, `"stream":false`, `"top_p":1`, 1),
		strings.Replace(npuRequest, `"max_completion_tokens":42`, `"max_completion_tokens":0`, 1),
		strings.Replace(npuRequest, `"strict":true`, `"strict":false`, 1),
		strings.Replace(npuRequest, `"stream":false`, `"malformed_control":5`, 1),
		fmt.Sprintf(`{"model":%q,"messages":[{"role":"user","content":"hello"}]}`, npu.ID()),
	} {
		if w := perform(h, http.MethodPost, tokenCountPath, body); w.Code != 400 {
			t.Errorf("accepted invalid preflight: %d %s", w.Code, w.Body)
		}
	}
	if w := perform(h, http.MethodPost, tokenCountPath, strings.Repeat(" ", 2<<20)+npuRequest); w.Code != 413 {
		t.Errorf("accepted oversized body: %d %s", w.Code, w.Body)
	}
}

func TestNPUTokenCountRejectsOversizedSocketBody(t *testing.T) {
	m := npuModel(socketRuntime(t, func(http.ResponseWriter, *http.Request) {
		t.Error("oversized preflight reached the NPU runtime")
	}))
	// JSON escaping in the socket payload matters as much as source bytes.
	body := strings.Replace(chatRequest(m.ID()), "hello", strings.Repeat("<", 11000), 1)
	w := perform(mustNew(t, m), http.MethodPost, tokenCountPath, body)
	if w.Code != 413 || !strings.Contains(w.Body.String(), `"code":"TOKEN_COUNT_REQUEST_TOO_LARGE"`) || strings.Contains(w.Body.String(), strings.Repeat("<", 30)) {
		t.Fatalf("private NPU size limit: %d %s", w.Code, w.Body)
	}
}

func TestNPUTokenCountUsesBoundRuntime(t *testing.T) {
	digest := strings.Repeat("a", 64)
	for _, tc := range []struct {
		name      string
		device    string
		reply     string
		upstream  int
		status    int
		errorCode string
	}{
		{"above limit", "NPU", `"input_tokens":1800,"max_input_tokens":1600,"max_completion_tokens":36`, 200, 200, ""},
		{"invalid count", "NPU", `"input_tokens":-1,"max_input_tokens":1600,"max_completion_tokens":36`, 200, 502, "INVALID_OUTPUT"},
		{"missing limit", "NPU", `"input_tokens":42,"max_completion_tokens":36`, 200, 502, "INVALID_OUTPUT"},
		{"invalid reserve", "NPU", `"input_tokens":42,"max_input_tokens":1600,"max_completion_tokens":50`, 200, 502, "INVALID_OUTPUT"},
		{"binding changed", "NPU", `"device":"CPU","input_tokens":42,"max_input_tokens":1600,"max_completion_tokens":36`, 200, 409, "PROFILE_CHANGED"},
		{"wrong health", "CPU", "", 0, 409, "PROFILE_CHANGED"},
		{"unsupported", "NPU", "", 404, 501, "TOKEN_COUNT_UNSUPPORTED"},
		{"rejected", "NPU", "", 422, 502, ""},
	} {
		t.Run(tc.name, func(t *testing.T) {
			var countCalls atomic.Int32
			ids := make(chan string, 2)
			m := npuModel(socketRuntime(t, func(w http.ResponseWriter, r *http.Request) {
				if r.Header.Get("Authorization") != "" {
					t.Error("preflight forwarded the caller credential")
				}
				select {
				case ids <- r.Header.Get(npuCallHeader):
				default:
					t.Error("unexpected extra runtime call")
				}
				switch r.Method + " " + r.URL.Path {
				case "GET /health":
					fmt.Fprintf(w, `{"profile":%q,"device":%q}`, digest, tc.device)
				case "POST /v1/count-tokens":
					countCalls.Add(1)
					var in struct {
						Profile             string
						Messages            []message
						Schema              map[string]json.RawMessage
						MaxCompletionTokens int `json:"max_completion_tokens"`
					}
					raw, err := io.ReadAll(r.Body)
					if err != nil || decodeStrict(raw, &in) != nil || in.Profile != digest || len(in.Messages) != 1 || in.Messages[0] != (message{"user", "hello"}) || string(in.Schema["type"]) != `"object"` || in.MaxCompletionTokens != 42 {
						t.Errorf("malformed runtime preflight: %s", raw)
					}
					if tc.upstream != 200 {
						http.Error(w, "private-runtime-detail", tc.upstream)
						return
					}
					binding := `"device":"NPU",`
					if strings.Contains(tc.reply, `"device"`) {
						binding = ""
					}
					fmt.Fprintf(w, `{"profile":%q,%s%s,"private":"private-runtime-detail"}`, digest, binding, tc.reply)
				default:
					t.Errorf("unexpected runtime path %s %s", r.Method, r.URL.Path)
					http.NotFound(w, r)
				}
			}))
			w := perform(mustNew(t, m), http.MethodPost, tokenCountPath, chatRequest(m.ID()))
			if w.Code != tc.status || strings.Contains(w.Body.String(), "private-runtime-detail") {
				t.Fatalf("preflight status/body: %d %s", w.Code, w.Body)
			}
			if tc.status == 200 {
				input, maxInput, maxCompletion, exact, id := readTokenCount(t, w)
				if input != 1800 || maxInput != 1600 || maxCompletion != 36 || !exact || id != m.ID() {
					t.Fatalf("unexpected exact count: %d %d %d %t %q", input, maxInput, maxCompletion, exact, id)
				}
			} else if tc.errorCode != "" && !strings.Contains(w.Body.String(), fmt.Sprintf(`"code":%q`, tc.errorCode)) {
				t.Fatalf("unexpected NPU failure %s", w.Body)
			}
			wantCountCalls := int32(1)
			if tc.device != "NPU" {
				wantCountCalls = 0
			}
			if countCalls.Load() != wantCountCalls || len(ids) != int(1+wantCountCalls) || w.Header().Get(npuCallHeader) == "" {
				t.Fatalf("wrong count calls: %d, status %d", countCalls.Load(), w.Code)
			}
			for len(ids) > 0 {
				if id := <-ids; id != w.Header().Get(npuCallHeader) {
					t.Errorf("runtime call used ID %q, manager response used %q", id, w.Header().Get(npuCallHeader))
				}
			}
		})
	}
}
