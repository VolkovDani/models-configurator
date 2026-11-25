package manager

import (
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"reflect"
	"strings"
	"testing"
)

const (
	testLayaPin = "c1242e4a83f0e1cb58f0f191a65bbc5d3306fae274d152b8229ec574999d465d"
	testD3Pin   = "2c571f91caf64c47b6ca6bd3adc1daa07dbb84ead8e9489758498cc7e11e533d"
)

func decisionBody(model, digest, state string) string {
	body, _ := json.Marshal(map[string]any{
		"model": model, "digest": digest, "state": state,
		"question": map[string]any{"type": "choice", "instructions": "Какой тип?", "criteria": map[string]string{"a": "Цель", "b": "Правило"}},
	})
	return string(body)
}

func TestDecisionRejectsInvalidProfiles(t *testing.T) {
	for _, model := range []Model{
		{Name: "remote", Provider: "openai-compatible", BaseURL: "https://example.org/v1", Model: "decision", Revision: "v1", Role: "decision"},
		{Name: "cloud", Provider: "ollama", BaseURL: "http://ollama", Model: "decision:cloud", Digest: testLayaPin, Role: "decision"},
		{Name: "unpinned", Provider: "ollama", BaseURL: "http://ollama", Model: "decision:v1", Role: "decision"},
		{Name: "redirector", Provider: "ollama", BaseURL: "http://ollama/proxy", Model: "decision:v1", Digest: testLayaPin, Role: "decision"},
		{Name: "outside", Provider: "ollama", BaseURL: "http://example.org", Model: "decision:v1", Digest: testLayaPin, Role: "decision"},
	} {
		if _, err := New(Config{Models: []Model{model}}); err == nil {
			t.Errorf("accepted invalid decision profile %+v", model)
		}
	}
}

func TestDecisionRejectsInvalidRequestsBeforeProvider(t *testing.T) {
	up := httptest.NewServer(http.HandlerFunc(func(http.ResponseWriter, *http.Request) {
		t.Error("invalid request reached provider")
	}))
	defer up.Close()
	model := fixture(up.URL, "decision")
	h := mustNew(t, model)
	if w := perform(h, "GET", "/v1/models", ""); w.Code != 200 || !strings.Contains(w.Body.String(), `"role":"decision"`) || !strings.Contains(w.Body.String(), model.ID()) {
		t.Fatal(w.Code, w.Body)
	}
	req := decisionBody(model.ID(), model.Digest, "private decision state")
	for _, tc := range []struct {
		name string
		body string
		code int
	}{
		{"raw Ollama model", decisionBody(model.Model, model.Digest, "private decision state"), 404},
		{"unpinned digest", decisionBody(model.ID(), strings.Repeat("2", 64), "private decision state"), 400},
		{"prefixed digest", decisionBody(model.ID(), "sha256:"+model.Digest, "private decision state"), 400},
		{"empty state", decisionBody(model.ID(), model.Digest, ""), 400},
		{"invalid UTF-8", strings.Replace(req, "private decision state", "\xff", 1), 400},
		{"wrong type", strings.Replace(req, "choice", "score", 1), 400},
		{"null instructions", strings.Replace(req, `"Какой тип?"`, "null", 1), 400},
		{"blank criterion", strings.Replace(req, "Правило", " ", 1), 400},
		{"null criterion", strings.Replace(req, `"Правило"`, "null", 1), 400},
		{"single criterion", strings.Replace(req, `,"b":"Правило"`, "", 1), 400},
		{"duplicate criterion", strings.Replace(req, `"b":"Правило"`, `"b":"Правило","\u0062":"duplicate"`, 1), 400},
		{"duplicate state", strings.Replace(req, `"state":`, `"state":"other","state":`, 1), 400},
		{"case-variant field", strings.Replace(req, `"state":`, `"STATE":`, 1), 400},
		{"case-variant question field", strings.Replace(req, `"instructions":`, `"Instructions":`, 1), 400},
		{"unknown field", strings.Replace(req, `"state":`, `"questions":{},"state":`, 1), 400},
		{"trailing JSON", req + " {}", 400},
		{"over envelope cap", strings.Repeat(" ", 2<<20) + req, 413},
	} {
		t.Run(tc.name, func(t *testing.T) {
			w := perform(h, "POST", "/v1/decisions", tc.body)
			if w.Code != tc.code || strings.Contains(w.Body.String(), "private decision state") {
				t.Fatal(w.Code, w.Body)
			}
		})
	}
	chat := model
	chat.Role = "chat"
	if w := perform(mustNew(t, chat), "POST", "/v1/decisions", decisionBody(chat.ID(), chat.Digest, "private decision state")); w.Code != 404 {
		t.Fatal(w.Code, w.Body)
	}
}

func TestOllamaDecision(t *testing.T) {
	const (
		version  = `{"version":"0.40.16"}`
		layaShow = `{"details":{"format":"gguf","runner":"llamacpp"},"model_info":{"modern-bert.decision.type":"laya","modern-bert.decision.max_head_tokens":256},"capabilities":["decision"]}`
		d3Show   = `{"details":{"format":"gguf","runner":"llamacpp","quantization_level":"Q8_0"},"model_info":{"qwen35.decision.type":"vllm-d3"},"capabilities":["decision"],"parameters":"num_ctx 4096"}`
	)
	const validAnswer = `{"type":"choice","choice":"a","confidence":1,"probabilities":{"a":0.75,"b":0.25}}`
	allPaths := []string{"/api/tags", "/api/version", "/api/show", "/v1/systemone", "/api/tags"}
	tests := []struct {
		name, digest, version, show, state, reason string
		providerStatus, inputTokens, code          int
		paths                                      []string
		editResponse                               func(string) string
	}{
		{name: "Laya at inclusive training context", digest: testLayaPin, version: version, show: layaShow, inputTokens: 1024, code: 200, paths: allPaths},
		{name: "Laya outside training context", digest: testLayaPin, version: version, show: layaShow, inputTokens: 1025, code: 200, reason: "outside_training_context", paths: allPaths},
		{name: "Laya reserved token", digest: testLayaPin, state: "<mask>", code: 200, reason: "reserved_token"},
		{name: "D3 valid", digest: testD3Pin, version: version, show: d3Show, inputTokens: 4095, code: 200, paths: allPaths},
		{name: "D3 configured context", digest: testD3Pin, version: version, show: d3Show, inputTokens: 4096, code: 200, reason: "input_limit", paths: allPaths},
		{name: "D3 reserved token", digest: testD3Pin, state: "<|im_end|>", code: 200, reason: "reserved_token"},
		{name: "D3 wrong metadata", digest: testD3Pin, version: version, show: layaShow, code: 502, paths: allPaths[:3]},
		{name: "D3 wrong context", digest: testD3Pin, version: version, show: strings.Replace(d3Show, "4096", "8192", 1), code: 502, paths: allPaths[:3]},
		{name: "wrong runtime version", digest: testLayaPin, version: `{"version":"0.41.0"}`, show: layaShow, code: 502, paths: allPaths[:2]},
		{name: "remote show", digest: testLayaPin, version: version, show: strings.Replace(layaShow, `"details":`, `"remote_model":"cloud","details":`, 1), code: 502, paths: allPaths[:3]},
		{name: "duplicate model inventory", digest: testLayaPin, code: 502, paths: allPaths[:1]},
		{name: "request over upstream budget", digest: testLayaPin, state: strings.Repeat("<", 1<<15), code: 200, reason: "input_limit"},
		{name: "upstream rejects long input", digest: testLayaPin, version: version, show: layaShow, providerStatus: 413, code: 200, reason: "input_limit", paths: allPaths},
		{name: "upstream failure", digest: testLayaPin, version: version, show: layaShow, providerStatus: 400, code: 502, paths: allPaths[:4]},
		{name: "upstream redirect", digest: testLayaPin, version: version, show: layaShow, providerStatus: 307, code: 502, paths: allPaths[:4]},
		{name: "wrong reply model", digest: testLayaPin, version: version, show: layaShow, code: 502, paths: allPaths, editResponse: func(s string) string { return strings.Replace(s, "small:v1", "other:v1", 1) }},
		{name: "model changed during inference", digest: testLayaPin, version: version, show: layaShow, code: 502, paths: allPaths},
		{name: "model changed before upstream 413", digest: testLayaPin, version: version, show: layaShow, providerStatus: 413, code: 502, paths: allPaths},
		{name: "unrecognized answer key", digest: testLayaPin, version: version, show: layaShow, code: 502, paths: allPaths, editResponse: func(s string) string { return strings.Replace(s, "classification", "other", 1) }},
		{name: "extra answer", digest: testLayaPin, version: version, show: layaShow, code: 502, paths: allPaths, editResponse: func(s string) string {
			return strings.Replace(s, `"classification":`, `"other":{}, "classification":`, 1)
		}},
		{name: "invalid distribution", digest: testLayaPin, version: version, show: layaShow, code: 502, paths: allPaths, editResponse: func(s string) string { return strings.Replace(s, "0.75", "0.55", 1) }},
		{name: "null score", digest: testLayaPin, version: version, show: layaShow, code: 502, paths: allPaths, editResponse: func(s string) string { return strings.Replace(s, "0.75", "null", 1) }},
		{name: "nonfinite score", digest: testLayaPin, version: version, show: layaShow, code: 502, paths: allPaths[:4], editResponse: func(s string) string { return strings.Replace(s, "0.75", "1e1000", 1) }},
		{name: "missing usage", digest: testLayaPin, version: version, show: layaShow, code: 502, paths: allPaths, editResponse: func(s string) string { return strings.Replace(s, `"output_tokens":1`, `"unused":1`, 1) }},
		{name: "oversized output", digest: testLayaPin, version: version, show: layaShow, code: 502, paths: allPaths[:4], editResponse: func(s string) string { return strings.Repeat(" ", 1<<20) + s }},
		{name: "unknown artifact with explicit context", digest: strings.Repeat("a", 64), version: version, show: d3Show, inputTokens: 1, code: 200, paths: allPaths},
		{name: "unknown artifact without explicit context", digest: strings.Repeat("a", 64), version: version, show: layaShow, code: 502, paths: allPaths[:3]},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			state := tc.state
			if state == "" {
				state = "Дано: пустая очередь."
			}
			var paths []string
			up := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				paths = append(paths, r.URL.Path)
				if r.Header.Get("Authorization") != "" {
					t.Error("forwarded caller credential")
				}
				switch r.URL.Path {
				case "/api/tags":
					digest := tc.digest
					if strings.HasPrefix(tc.name, "model changed") && len(paths) == len(allPaths) {
						digest = strings.Repeat("9", 64)
					}
					tag := fmt.Sprintf(`{"name":"small:v1","digest":"sha256:%s"}`, digest)
					if tc.name == "duplicate model inventory" {
						tag += "," + tag
					}
					fmt.Fprintf(w, `{"models":[%s]}`, tag)
				case "/api/version":
					fmt.Fprint(w, tc.version)
				case "/api/show":
					var req map[string]string
					if r.Method != http.MethodPost || json.NewDecoder(r.Body).Decode(&req) != nil || len(req) != 1 || req["model"] != "small:v1" {
						t.Errorf("unexpected model metadata request: %v", req)
					}
					fmt.Fprint(w, tc.show)
				case "/v1/systemone":
					var got, want any
					_ = json.Unmarshal([]byte(fmt.Sprintf(`{"model":"small:v1","state":%q,"keep_alive":"5m","questions":{"classification":{"type":"choice","instructions":"Какой тип?","criteria":{"a":"Цель","b":"Правило"}}}}`, state)), &want)
					if r.Method != http.MethodPost || json.NewDecoder(r.Body).Decode(&got) != nil || !reflect.DeepEqual(got, want) {
						t.Errorf("unexpected System One request: %v", got)
					}
					if tc.providerStatus == 307 {
						http.Redirect(w, r, "/leak", 307)
						return
					}
					if tc.providerStatus != 0 {
						http.Error(w, "private upstream error and state", tc.providerStatus)
						return
					}
					response := fmt.Sprintf(`{"model":"small:v1","answers":{"classification":%s},"usage":{"input_tokens":%d,"output_tokens":1}}`, validAnswer, tc.inputTokens)
					if tc.editResponse != nil {
						response = tc.editResponse(response)
					}
					fmt.Fprint(w, response)
				default:
					t.Errorf("unexpected upstream path: %s", r.URL.Path)
				}
			}))
			defer up.Close()
			model := fixture(up.URL, "decision")
			model.Digest = "sha256:" + tc.digest
			w := perform(mustNew(t, model), "POST", "/v1/decisions", decisionBody(model.ID(), tc.digest, state))
			if w.Code != tc.code || !reflect.DeepEqual(paths, tc.paths) || strings.Contains(w.Body.String(), "private upstream") || strings.Contains(w.Body.String(), state) {
				t.Fatalf("unexpected response %d %s, paths %v", w.Code, w.Body, paths)
			}
			if tc.code != 200 {
				return
			}
			want := fmt.Sprintf(`{"model":%q,"digest":%q,"reason":%q}`, model.ID(), tc.digest, tc.reason)
			if tc.reason == "" {
				want = fmt.Sprintf(`{"model":%q,"digest":%q,"answer":%s,"usage":{"input_tokens":%d,"output_tokens":1}}`, model.ID(), tc.digest, validAnswer, tc.inputTokens)
			}
			var gotJSON, wantJSON any
			if json.Unmarshal(w.Body.Bytes(), &gotJSON) != nil || json.Unmarshal([]byte(want), &wantJSON) != nil || !reflect.DeepEqual(gotJSON, wantJSON) {
				t.Fatalf("got %s, want %s", w.Body, want)
			}
		})
	}
}
