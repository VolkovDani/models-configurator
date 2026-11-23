package manager

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"strings"
)

type provider struct {
	model  Model
	key    string
	client *http.Client
}

func (p provider) request(ctx context.Context, method, path string, input, out any) error {
	var body []byte
	if input != nil {
		var err error
		body, err = json.Marshal(input)
		if err != nil {
			return err
		}
	}
	req, err := http.NewRequestWithContext(ctx, method, strings.TrimRight(p.model.BaseURL, "/")+path, bytes.NewReader(body))
	if err != nil {
		return err
	}
	req.Header.Set("Accept", "application/json")
	if input != nil {
		req.Header.Set("Content-Type", "application/json")
	}
	if p.key != "" {
		req.Header.Set("Authorization", "Bearer "+p.key)
	}
	resp, err := p.client.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	if resp.StatusCode != 200 {
		return errors.New("provider rejected request")
	}
	raw, err := readBounded(resp.Body, 8<<20)
	if err != nil {
		return err
	}
	return json.Unmarshal(raw, out)
}

func (p provider) verify(ctx context.Context) error {
	var tags struct {
		Models []struct {
			Name        string `json:"name"`
			Model       string `json:"model"`
			Digest      string `json:"digest"`
			RemoteHost  string `json:"remote_host"`
			RemoteModel string `json:"remote_model"`
		} `json:"models"`
	}
	if err := p.request(ctx, "GET", "/api/tags", nil, &tags); err != nil {
		return err
	}
	matches := 0
	for _, tag := range tags.Models {
		if OllamaTag(tag.Name) != p.model.Model && OllamaTag(tag.Model) != p.model.Model {
			continue
		}
		if tag.RemoteHost != "" || tag.RemoteModel != "" || strings.TrimPrefix(tag.Digest, "sha256:") != strings.TrimPrefix(p.model.Digest, "sha256:") {
			return errors.New("model binding changed")
		}
		matches++
	}
	if matches != 1 {
		return errors.New("model binding unavailable or ambiguous")
	}
	return nil
}

type outputMessage struct {
	Role         string            `json:"role"`
	Content      *string           `json:"content"`
	ToolCalls    []json.RawMessage `json:"tool_calls"`
	FunctionCall json.RawMessage   `json:"function_call"`
	Refusal      *string           `json:"refusal"`
}
type choice struct {
	Index        int     `json:"index"`
	FinishReason string  `json:"finish_reason"`
	Message      message `json:"message"`
}

func chatResult(id, finish string, msg outputMessage) (any, error) {
	if (finish != "stop" && finish != "length") || msg.Role != "assistant" || msg.Content == nil || msg.Refusal != nil || len(msg.ToolCalls) != 0 || (msg.FunctionCall != nil && string(msg.FunctionCall) != "null") {
		return nil, errors.New("unsupported model completion")
	}
	return map[string]any{"object": "chat.completion", "model": id, "choices": []choice{{0, finish, message{msg.Role, *msg.Content}}}}, nil
}

func (p provider) chat(ctx context.Context, in chatRequestBody) (any, error) {
	id := in.Model
	in.Model = p.model.Model
	if p.model.Provider == "openai-compatible" {
		var resp struct {
			Model   string `json:"model"`
			Choices []struct {
				FinishReason string        `json:"finish_reason"`
				Message      outputMessage `json:"message"`
			} `json:"choices"`
		}
		if err := p.request(ctx, "POST", "/chat/completions", in, &resp); err != nil {
			return nil, err
		}
		if (resp.Model != "" && resp.Model != in.Model && resp.Model != p.model.Revision) || len(resp.Choices) != 1 {
			return nil, errors.New("unexpected chat completion")
		}
		return chatResult(id, resp.Choices[0].FinishReason, resp.Choices[0].Message)
	}
	if err := p.verify(ctx); err != nil {
		return nil, err
	}
	options := map[string]any{"num_ctx": 32768, "temperature": 0, "num_predict": 1024}
	if in.Temperature != nil {
		options["temperature"] = *in.Temperature
	}
	if in.TopP != nil {
		options["top_p"] = *in.TopP
	}
	if in.MaxCompletionTokens != nil {
		options["num_predict"] = *in.MaxCompletionTokens
	}
	body := map[string]any{"model": in.Model, "messages": in.Messages, "stream": false, "truncate": false, "shift": false, "options": options}
	if in.ResponseFormat != nil {
		body["format"] = in.ResponseFormat.JSONSchema.Schema
	}
	var resp struct {
		Model       string        `json:"model"`
		RemoteHost  string        `json:"remote_host"`
		RemoteModel string        `json:"remote_model"`
		Done        bool          `json:"done"`
		DoneReason  string        `json:"done_reason"`
		Message     outputMessage `json:"message"`
	}
	if err := p.request(ctx, "POST", "/api/chat", body, &resp); err != nil {
		return nil, err
	}
	if !resp.Done || resp.RemoteModel != "" || resp.RemoteHost != "" || (resp.Model != "" && OllamaTag(resp.Model) != in.Model) {
		return nil, errors.New("incomplete or unexpected Ollama completion")
	}
	result, err := chatResult(id, resp.DoneReason, resp.Message)
	if err != nil {
		return nil, err
	}
	return result, p.verify(ctx)
}

type vector []float32

func (v *vector) UnmarshalJSON(data []byte) error {
	// A JSON null component otherwise decodes silently to zero.
	if bytes.Contains(data, []byte("null")) {
		return errors.New("null embedding component")
	}
	return json.Unmarshal(data, (*[]float32)(v))
}

type embeddingRow struct {
	Index     *int   `json:"index"`
	Embedding vector `json:"embedding"`
	Object    string `json:"object"`
}

func embeddingResult(id string, rows []embeddingRow, n int, want *int) (any, error) {
	if len(rows) != n {
		return nil, errors.New("embedding count mismatch")
	}
	ordered := make([]embeddingRow, n)
	dim := len(rows[0].Embedding)
	if dim == 0 || dim > 65536 || (want != nil && dim != *want) {
		return nil, errors.New("unexpected embedding dimension")
	}
	for _, row := range rows {
		if row.Index == nil || *row.Index < 0 || *row.Index >= n || ordered[*row.Index].Index != nil || len(row.Embedding) != dim {
			return nil, errors.New("invalid embedding row")
		}
		nonzero := false
		for _, x := range row.Embedding {
			if x != 0 {
				nonzero = true
				break
			}
		}
		if !nonzero {
			return nil, errors.New("zero embedding vector")
		}
		row.Object = "embedding"
		ordered[*row.Index] = row
	}
	return map[string]any{"object": "list", "model": id, "data": ordered}, nil
}

func (p provider) embed(ctx context.Context, in embeddingRequest) (any, error) {
	id := in.Model
	in.Model = p.model.Model
	if p.model.Provider == "openai-compatible" {
		var resp struct {
			Model string         `json:"model"`
			Data  []embeddingRow `json:"data"`
		}
		if err := p.request(ctx, "POST", "/embeddings", in, &resp); err != nil {
			return nil, err
		}
		if resp.Model != "" && resp.Model != in.Model && resp.Model != p.model.Revision {
			return nil, errors.New("unexpected embedding model")
		}
		return embeddingResult(id, resp.Data, len(in.Input), in.Dimensions)
	}
	if err := p.verify(ctx); err != nil {
		return nil, err
	}
	body := map[string]any{"model": in.Model, "input": in.Input, "truncate": false}
	if in.Dimensions != nil {
		body["dimensions"] = *in.Dimensions
	}
	var resp struct {
		Model      string   `json:"model"`
		Embeddings []vector `json:"embeddings"`
	}
	if err := p.request(ctx, "POST", "/api/embed", body, &resp); err != nil {
		return nil, err
	}
	if resp.Model != "" && OllamaTag(resp.Model) != in.Model {
		return nil, errors.New("unexpected Ollama embedding model")
	}
	rows := make([]embeddingRow, len(resp.Embeddings))
	for i, v := range resp.Embeddings {
		rows[i] = embeddingRow{Index: &i, Embedding: v}
	}
	result, err := embeddingResult(id, rows, len(in.Input), in.Dimensions)
	if err != nil {
		return nil, err
	}
	return result, p.verify(ctx)
}
