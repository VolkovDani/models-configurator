package main

import (
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"io"
	"mime/multipart"
	"net/http"
	"net/http/httptest"
	"net/url"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/VolkovDani/models-configurator/internal/manager"
)

func TestWebSettingsExportAndApply(t *testing.T) {
	t.Setenv("MODELS_HISTORY_PATH", filepath.Join(t.TempDir(), "history.jsonl"))
	t.Setenv("MODELS_MAX_PARALLEL_REQUESTS", "1")
	t.Setenv("MODELS_SETTINGS_PATH", "")
	digest := strings.Repeat("a", 64)
	nextDigest := strings.Repeat("b", 64)
	chatOptions := make(chan map[string]float64, 2)
	ollama := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/api/tags":
			fmt.Fprintf(w, `{"models":[{"name":"qwen:7b","digest":%q},{"name":"qwen:14b","digest":%q},{"name":"remote:7b","digest":%q,"remote_host":"https://remote.example"}]}`, digest, nextDigest, digest)
		case "/api/chat":
			var req struct {
				Model   string
				Options map[string]float64
			}
			if err := json.NewDecoder(r.Body).Decode(&req); err != nil || req.Model != "qwen:14b" {
				t.Errorf("unexpected Ollama chat: %+v %v", req, err)
				w.WriteHeader(http.StatusBadRequest)
				return
			}
			chatOptions <- req.Options
			fmt.Fprint(w, `{"model":"qwen:14b","done":true,"done_reason":"stop","message":{"role":"assistant","content":"Hello"}}`)
		default:
			t.Errorf("settings reached %s", r.URL.Path)
			http.NotFound(w, r)
		}
	}))
	defer ollama.Close()
	model := manager.Model{Name: "chat", Model: "qwen:7b", Role: "chat", Provider: "ollama", BaseURL: ollama.URL, Digest: digest}
	cfg := manager.Config{Models: []manager.Model{model, {Name: "secondary", Model: "qwen:7b", Role: "chat", Provider: "ollama", BaseURL: ollama.URL, Digest: digest}}}
	h, err := service(cfg)
	if err != nil {
		t.Fatal(err)
	}

	status := httptest.NewRecorder()
	h.ServeHTTP(status, httptest.NewRequest("GET", "http://localhost/settings?lang=ru", nil))
	if status.Header().Get("Referrer-Policy") != "same-origin" {
		t.Fatal("browser form submissions must keep the same-origin header")
	}
	if status.Code != 200 || !strings.Contains(status.Body.String(), "Настройки моделей") ||
		!strings.Contains(status.Body.String(), "qwen:14b") || strings.Contains(status.Body.String(), "remote:7b") {
		t.Fatalf("settings page: %d %s", status.Code, status.Body)
	}

	form := url.Values{
		"parallel": {"3"},
		"0.name":   {"new-chat"}, "0.role": {"chat"}, "0.enabled": {"on"},
		"0.installed": {"qwen:14b@" + nextDigest},
		"0.context":   {"4096"}, "0.completion": {"1024"}, "0.temperature": {"0.25"}, "0.top_p": {"0.9"},
		"1.name": {"secondary"}, "1.role": {"chat"},
	}
	save := func(host string) *httptest.ResponseRecorder {
		r := httptest.NewRequest("POST", "http://"+host+"/settings", strings.NewReader(form.Encode()))
		r.Header.Set("Content-Type", "application/x-www-form-urlencoded")
		r.Header.Set("Origin", "http://"+host)
		w := httptest.NewRecorder()
		h.ServeHTTP(w, r)
		return w
	}
	export := save("localhost")
	if export.Code != 200 || !strings.Contains(export.Header().Get("Content-Disposition"), "model-settings.json") ||
		strings.Contains(export.Body.String(), ollama.URL) {
		t.Fatalf("settings export: %d %s", export.Code, export.Body)
	}
	form.Set("0.installed", "remote:7b@"+digest)
	if got := save("localhost"); got.Code != http.StatusBadRequest {
		t.Fatalf("remote profile accepted: %d", got.Code)
	}
	inventory := func(h http.Handler) []string {
		t.Helper()
		w := httptest.NewRecorder()
		h.ServeHTTP(w, httptest.NewRequest("GET", "http://localhost/v1/models", nil))
		var result struct{ Data []struct{ Model string } }
		if w.Code != 200 || json.Unmarshal(w.Body.Bytes(), &result) != nil {
			t.Fatalf("inventory: %d %s", w.Code, w.Body)
		}
		var names []string
		for _, entry := range result.Data {
			names = append(names, entry.Model)
		}
		return names
	}
	if got := inventory(h); strings.Join(got, ",") != "qwen:7b,qwen:7b" {
		t.Fatalf("saving changed running profiles: %v", got)
	}
	path := filepath.Join(t.TempDir(), "model-settings.json")
	if err := os.WriteFile(path, export.Body.Bytes(), 0600); err != nil {
		t.Fatal(err)
	}
	t.Setenv("MODELS_SETTINGS_PATH", path)
	applied, err := service(cfg)
	if err != nil {
		t.Fatal(err)
	}
	if got := inventory(applied); strings.Join(got, ",") != "qwen:14b" {
		t.Fatalf("settings were not applied at startup: %v", got)
	}

	model.Model, model.Digest, model.Name = "qwen:14b", nextDigest, "new-chat"
	var draft manager.Settings
	if err := json.Unmarshal(export.Body.Bytes(), &draft); err != nil {
		t.Fatal(err)
	}
	configured, err := draft.Apply(cfg)
	if err != nil {
		t.Fatal(err)
	}
	if configured.Models[0].ID() == model.ID() {
		t.Fatal("chat defaults must change the inference profile ID")
	}
	for _, tc := range []struct {
		override                string
		completion, temperature float64
	}{
		{"", 1024, 0.25},
		{`,"max_completion_tokens":256,"temperature":0.5`, 256, 0.5},
	} {
		w := httptest.NewRecorder()
		body := fmt.Sprintf(`{"model":%q,"messages":[{"role":"user","content":"Hi"}]%s}`, configured.Models[0].ID(), tc.override)
		applied.ServeHTTP(w, httptest.NewRequest("POST", "http://localhost/v1/chat/completions", strings.NewReader(body)))
		if w.Code != http.StatusOK {
			t.Fatalf("chat: %d %s", w.Code, w.Body)
		}
		opts := <-chatOptions
		if opts["num_ctx"] != 4096 || opts["num_predict"] != tc.completion || opts["temperature"] != tc.temperature || opts["top_p"] != 0.9 {
			t.Fatalf("Ollama options: %+v", opts)
		}
	}
}

func TestWebNPUManifestDerivesPinnedProfile(t *testing.T) {
	t.Setenv("MODELS_HISTORY_PATH", filepath.Join(t.TempDir(), "history.jsonl"))
	t.Setenv("MODELS_SETTINGS_PATH", "")
	const original = `{"device":"NPU","export_command":"опция <тест>","files":{"config.json":"hash"},"license":"Apache-2.0","max_new_tokens":512,"max_prompt_tokens":2048,"model_id":"Qwen","output_format":"json-schema-v1","revision":"commit","runtime":{"openvino":"1"},"version":2}`
	sum := fmt.Sprintf("%x", sha256.Sum256([]byte(original)))
	model := manager.Model{Name: "npu", Model: "Qwen", Role: "chat", Provider: "openvino-npu", BaseURL: "unix:///run/npu.sock", Digest: sum}
	h, err := service(manager.Config{Models: []manager.Model{model}})
	if err != nil {
		t.Fatal(err)
	}
	send := func(manifest, prompt string) *httptest.ResponseRecorder {
		t.Helper()
		var body strings.Builder
		form := multipart.NewWriter(&body)
		for key, value := range map[string]string{"profile": model.ID(), "prompt": prompt, "completion": "512"} {
			if err := form.WriteField(key, value); err != nil {
				t.Fatal(err)
			}
		}
		file, err := form.CreateFormFile("manifest", "original.json")
		if err != nil {
			t.Fatal(err)
		}
		if _, err = io.WriteString(file, manifest); err != nil {
			t.Fatal(err)
		}
		form.Close()
		r := httptest.NewRequest("POST", "http://localhost/settings/npu", strings.NewReader(body.String()))
		r.Header.Set("Content-Type", form.FormDataContentType())
		r.Header.Set("Origin", "http://localhost")
		w := httptest.NewRecorder()
		h.ServeHTTP(w, r)
		return w
	}
	want := strings.Replace(original, "2048", "4096", 1)
	got := send(original+"\n", "4096")
	newHash := fmt.Sprintf("%x", sha256.Sum256([]byte(want)))
	if got.Code != 200 || strings.TrimSpace(got.Body.String()) != want ||
		!strings.Contains(got.Header().Get("Content-Disposition"), newHash) {
		t.Fatalf("NPU manifest: %d %s (%s)", got.Code, got.Body, got.Header())
	}
	for _, invalid := range []struct{ manifest, prompt string }{
		{strings.Replace(original, "Qwen", "Other", 1), "4096"},
		{original, "32769"},
	} {
		if got := send(invalid.manifest, invalid.prompt); got.Code != http.StatusBadRequest {
			t.Fatalf("invalid manifest or budget accepted: %d", got.Code)
		}
	}
}
