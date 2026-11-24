package manager

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"os"
	"strings"
	"time"
	"unicode/utf8"
)

type inventoryItem struct {
	ID       string `json:"id"`
	Object   string `json:"object"`
	Provider string `json:"provider"`
	Model    string `json:"model"`
	Role     string `json:"role"`
}

func New(cfg Config) (http.Handler, error) {
	return NewWithConcurrency(cfg, 1)
}

func NewWithConcurrency(cfg Config, maxParallel int) (http.Handler, error) {
	if maxParallel < 1 {
		return nil, errors.New("inference concurrency must be positive")
	}
	if err := cfg.validate(); err != nil {
		return nil, err
	}
	transport := http.DefaultTransport.(*http.Transport).Clone()
	transport.Proxy = nil
	client := &http.Client{Transport: transport, CheckRedirect: func(*http.Request, []*http.Request) error { return http.ErrUseLastResponse }}
	entries := make([]inventoryItem, 0, len(cfg.Models))
	models := map[string]provider{}
	for _, m := range cfg.Models {
		key := ""
		if m.KeyEnv != "" {
			key = os.Getenv(m.KeyEnv)
			if key == "" || strings.TrimSpace(key) != key || strings.ContainsAny(key, "\r\n") {
				return nil, errors.New("configured provider credential is missing or invalid")
			}
		}
		id := m.ID()
		entries = append(entries, inventoryItem{id, "model", m.Provider, m.Model, m.Role})
		target := upstream{base: strings.TrimRight(m.BaseURL, "/"), key: key, client: client}
		if m.Provider == NPUProvider || m.Provider == BGEProvider {
			var err error
			target, err = socketUpstream(m.BaseURL)
			if err != nil {
				return nil, err
			}
		}
		models[id] = provider{model: m, upstream: target}
	}
	queue := newDispatcher(maxParallel)
	mux := http.NewServeMux()
	mux.HandleFunc("GET /v1/models", func(w http.ResponseWriter, r *http.Request) {
		writeJSON(w, 200, map[string]any{"object": "list", "data": entries})
	})
	mux.HandleFunc("POST /v1/chat/completions", func(w http.ResponseWriter, r *http.Request) {
		var in chatRequestBody
		if !readInput(w, r, &in) {
			return
		}
		if !in.valid() {
			fail(w, 400, "invalid chat request")
			return
		}
		p, ok := lookupProvider(w, models, in.Model, "chat")
		if !ok {
			return
		}
		if p.model.Provider == NPUProvider && (in.ResponseFormat == nil || in.TopP != nil || in.Temperature != nil) {
			fail(w, http.StatusBadRequest, "OpenVINO NPU requires strict JSON schema output and uses fixed sampling")
			return
		}
		queue.run(w, r, p, func(ctx context.Context) (any, error) { return p.chat(ctx, in) })
	})
	mux.HandleFunc("POST /v1/embeddings", func(w http.ResponseWriter, r *http.Request) {
		var in embeddingRequest
		if !readInput(w, r, &in) {
			return
		}
		if !in.valid() {
			fail(w, 400, "invalid embedding request")
			return
		}
		p, ok := lookupProvider(w, models, in.Model, "embeddings")
		if !ok {
			return
		}
		if p.model.Provider == BGEProvider && in.Dimensions != nil && *in.Dimensions != bgeDimensions {
			fail(w, 400, "BGE-M3 returns 1024 dimensions")
			return
		}
		queue.run(w, r, p, func(ctx context.Context) (any, error) { return p.embed(ctx, in) })
	})
	mux.HandleFunc("POST /v1/bge-m3/encode", func(w http.ResponseWriter, r *http.Request) {
		var in bgeRequest
		if !readInput(w, r, &in) {
			return
		}
		if !in.valid() {
			fail(w, 400, "invalid BGE-M3 request")
			return
		}
		p, ok := lookupProvider(w, models, in.Model, "embeddings")
		if !ok {
			return
		}
		if p.model.Provider != BGEProvider {
			fail(w, 404, "unknown model for this operation")
			return
		}
		queue.run(w, r, p, func(ctx context.Context) (any, error) { return p.bgeEncode(ctx, in.Input, in.Modes) })
	})
	return mux, nil
}

func lookupProvider(w http.ResponseWriter, models map[string]provider, id, role string) (provider, bool) {
	p, ok := models[id]
	if !ok || p.model.Role != role {
		fail(w, 404, "unknown model for this operation")
		return provider{}, false
	}
	if p.socket != "" {
		p.upstream = p.resolveSocket()
	}
	if p.model.Provider == NPUProvider {
		p.npuCallID = newNPUCallID()
		w.Header().Set(npuCallHeader, p.npuCallID)
	}
	return p, true
}

func (d *dispatcher) run(w http.ResponseWriter, r *http.Request, p provider, invoke func(context.Context) (any, error)) {
	ctx, cancel := context.WithTimeout(r.Context(), 2*time.Minute)
	defer cancel()
	if err := d.acquire(ctx, p.socket); err != nil {
		switch {
		case errors.Is(err, errQueueFull):
			fail(w, http.StatusTooManyRequests, errQueueFull.Error())
		case errors.Is(err, context.DeadlineExceeded):
			if p.model.Provider == NPUProvider {
				writeNPUFailure(w, npuFailure("TIMEOUT"))
			} else {
				fail(w, http.StatusGatewayTimeout, "inference request timed out")
			}
		default:
			fail(w, http.StatusRequestTimeout, "inference request canceled")
		}
		return
	}
	result, err := func() (any, error) {
		defer d.release(p.socket)
		return invoke(ctx)
	}()
	var npuErr npuFailure
	if p.model.Provider == NPUProvider && err != nil && errors.Is(ctx.Err(), context.DeadlineExceeded) {
		err = npuFailure("TIMEOUT")
	}
	if errors.As(err, &npuErr) {
		writeNPUFailure(w, npuErr)
		return
	}
	var workerStatus upstreamStatus
	if p.model.Provider == BGEProvider && errors.As(err, &workerStatus) {
		switch workerStatus {
		case 409, 413, 503:
			fail(w, int(workerStatus), http.StatusText(int(workerStatus)))
			return
		}
	}
	if err != nil {
		fail(w, 502, "model provider failed")
		return
	}
	writeJSON(w, 200, result)
}

func writeNPUFailure(w http.ResponseWriter, err npuFailure) {
	writeJSON(w, err.status(), map[string]any{"error": map[string]string{"message": "model provider failed", "code": err.Error()}})
}

func writeJSON(w http.ResponseWriter, code int, v any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(code)
	_ = json.NewEncoder(w).Encode(v)
}

func fail(w http.ResponseWriter, status int, message string) {
	writeJSON(w, status, map[string]any{"error": map[string]string{"message": message}})
}

func readInput(w http.ResponseWriter, r *http.Request, v any) bool {
	raw, err := readBounded(r.Body, 2<<20)
	if err != nil {
		fail(w, 413, "request body exceeds size limit")
		return false
	}
	if !utf8.Valid(raw) || decodeStrict(raw, v) != nil {
		fail(w, 400, "invalid request JSON")
		return false
	}
	return true
}

type message struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}
type responseFormat struct {
	Type       string `json:"type"`
	JSONSchema struct {
		Name   string          `json:"name"`
		Strict bool            `json:"strict"`
		Schema json.RawMessage `json:"schema"`
	} `json:"json_schema"`
}
type chatRequestBody struct {
	Model               string          `json:"model"`
	Messages            []message       `json:"messages"`
	Stream              bool            `json:"stream"`
	MaxCompletionTokens *int            `json:"max_completion_tokens,omitempty"`
	Temperature         *float64        `json:"temperature,omitempty"`
	TopP                *float64        `json:"top_p,omitempty"`
	ResponseFormat      *responseFormat `json:"response_format,omitempty"`
}

func (in chatRequestBody) valid() bool {
	if in.Stream || len(in.Messages) == 0 || len(in.Messages) > 128 {
		return false
	}
	if in.MaxCompletionTokens != nil && (*in.MaxCompletionTokens < 1 || *in.MaxCompletionTokens > 16384) {
		return false
	}
	if in.Temperature != nil && (*in.Temperature < 0 || *in.Temperature > 2) {
		return false
	}
	if in.TopP != nil && (*in.TopP <= 0 || *in.TopP > 1) {
		return false
	}
	for _, m := range in.Messages {
		if (m.Role != "system" && m.Role != "user" && m.Role != "assistant") || strings.TrimSpace(m.Content) == "" {
			return false
		}
	}
	if f := in.ResponseFormat; f != nil {
		if f.Type != "json_schema" || !f.JSONSchema.Strict || !namePattern.MatchString(f.JSONSchema.Name) {
			return false
		}
		var schema map[string]json.RawMessage
		if json.Unmarshal(f.JSONSchema.Schema, &schema) != nil || schema == nil {
			return false
		}
	}
	return true
}

type texts []string

func (v *texts) UnmarshalJSON(data []byte) error {
	if len(data) > 0 && data[0] == '"' {
		var s string
		if err := json.Unmarshal(data, &s); err != nil {
			return err
		}
		*v = texts{s}
		return nil
	}
	return json.Unmarshal(data, (*[]string)(v))
}

type embeddingRequest struct {
	Model          string `json:"model"`
	Input          texts  `json:"input"`
	EncodingFormat string `json:"encoding_format,omitempty"`
	Dimensions     *int   `json:"dimensions,omitempty"`
}

func (in embeddingRequest) valid() bool {
	if len(in.Input) == 0 || len(in.Input) > 32 || (in.EncodingFormat != "" && in.EncodingFormat != "float") || (in.Dimensions != nil && (*in.Dimensions <= 0 || *in.Dimensions > 65536)) {
		return false
	}
	for _, s := range in.Input {
		if strings.TrimSpace(s) == "" || utf8.RuneCountInString(s) > 10000 {
			return false
		}
	}
	return true
}
