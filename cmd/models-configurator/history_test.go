package main

import (
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
	"time"

	"github.com/VolkovDani/models-configurator/internal/manager"
)

func TestModelCallHistory(t *testing.T) {
	path := filepath.Join(t.TempDir(), "private", "calls.jsonl")
	t.Setenv("MODELS_HISTORY_PATH", path)
	t.Setenv("HISTORY_TEST_KEY", "private-key")
	const input = "private-prompt"
	const output = "private-answer"
	up := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.Method + " " + r.URL.Path {
		case "GET /models":
			fmt.Fprint(w, `{"data":[]}`)
		case "POST /chat/completions":
			fmt.Fprintf(w, `{"model":"deployment","choices":[{"finish_reason":"stop","message":{"role":"assistant","content":%q}}],"usage":{"prompt_tokens":14,"completion_tokens":7}}`, output)
		case "POST /embeddings":
			http.Error(w, "private-upstream-error", http.StatusServiceUnavailable)
		default:
			t.Errorf("unexpected provider request: %s %s", r.Method, r.URL.Path)
		}
	}))
	defer up.Close()

	chat := manager.Model{Name: "chat-profile", Provider: "openai-compatible", Model: "deployment", Role: "chat", BaseURL: up.URL, Revision: "r1", KeyEnv: "HISTORY_TEST_KEY"}
	embed := chat
	embed.Name, embed.Role = "embed-profile", "embeddings"
	cfg := manager.Config{Models: []manager.Model{chat, embed}}
	h, err := service(cfg)
	if err != nil {
		t.Fatal(err)
	}
	do := func(method, url, body string, code int) *httptest.ResponseRecorder {
		t.Helper()
		w := httptest.NewRecorder()
		h.ServeHTTP(w, httptest.NewRequest(method, url, strings.NewReader(body)))
		if w.Code != code {
			t.Fatalf("%s %s = %d, want %d: %s", method, url, w.Code, code, w.Body)
		}
		return w
	}
	const base = "http://localhost:8317"
	do("GET", base+"/status", "", 200)
	do("GET", base+"/v1/models", "", 200)
	do("POST", base+"/v1/chat/completions", `{}`, 400)
	chatBody := fmt.Sprintf(`{"model":%q,"messages":[{"role":"user","content":%q}]}`, chat.ID(), input)
	do("POST", "http://untrusted.example/v1/chat/completions", chatBody, 403)
	do("POST", base+"/v1/chat/completions", chatBody, 200)
	do("POST", base+"/v1/embeddings", fmt.Sprintf(`{"model":%q,"input":%q}`, embed.ID(), input), 502)

	h, err = service(manager.Config{Models: []manager.Model{}})
	if err != nil {
		t.Fatal(err)
	}
	w := do("GET", base+"/history.json", "", 200)
	var page struct {
		Calls []struct {
			ModelID      string    `json:"model_id"`
			Profile      string    `json:"profile"`
			Provider     string    `json:"provider"`
			Operation    string    `json:"operation"`
			StartedAt    time.Time `json:"started_at"`
			DurationMS   int64     `json:"duration_ms"`
			Status       int       `json:"status"`
			Outcome      string    `json:"outcome"`
			InputTokens  *int      `json:"input_tokens"`
			OutputTokens *int      `json:"output_tokens"`
		} `json:"calls"`
	}
	if err := json.Unmarshal(w.Body.Bytes(), &page); err != nil {
		t.Fatal(err)
	}
	if len(page.Calls) != 2 {
		t.Fatalf("got %d calls, want 2: %s", len(page.Calls), w.Body)
	}
	wantModels := []manager.Model{embed, chat}
	wantCodes := []int{502, 200}
	wantOutcomes := []string{"failed", "succeeded"}
	for i, call := range page.Calls {
		model := wantModels[i]
		if call.ModelID != model.ID() || call.Profile != model.Name || call.Provider != model.Provider ||
			call.Operation != model.Role || call.Status != wantCodes[i] || call.Outcome != wantOutcomes[i] ||
			call.StartedAt.IsZero() || call.DurationMS < 0 {
			t.Errorf("call %d: %+v", i, call)
		}
	}
	if page.Calls[0].InputTokens != nil || page.Calls[0].OutputTokens != nil ||
		!reflect.DeepEqual(page.Calls[1].InputTokens, new(14)) || !reflect.DeepEqual(page.Calls[1].OutputTokens, new(7)) {
		t.Fatalf("wrong persisted token counts: %s", w.Body)
	}
	contents, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	for _, secret := range []string{input, output, "private-upstream-error", "private-key", up.URL, "HISTORY_TEST_KEY"} {
		if strings.Contains(string(contents), secret) || strings.Contains(w.Body.String(), secret) {
			t.Errorf("history contains %q", secret)
		}
	}

	for _, fragment := range []string{`href="/history?lang=ru"`, "История вызовов"} {
		if body := do("GET", base+"/?lang=ru", "", 200).Body.String(); !strings.Contains(body, fragment) {
			t.Errorf("status page lacks %q", fragment)
		}
	}
	html := do("GET", base+"/history?lang=ru", "", 200)
	for _, fragment := range []string{"История вызовов", "embed-profile", "Ошибка", "Успешно"} {
		if !strings.Contains(html.Body.String(), fragment) {
			t.Errorf("history page lacks %q", fragment)
		}
	}
	do("GET", "http://untrusted.example/history.json", "", 403)
	do("POST", base+"/history", "", 405)
	do("GET", base+"/history.json?before=bogus", "", 400)
	do("GET", base+"/history.js", "", 200)
}
