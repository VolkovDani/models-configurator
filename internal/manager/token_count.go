package manager

import (
	"context"
	"errors"
)

var (
	errTokenCountUnsupported     = errors.New("TOKEN_COUNT_UNSUPPORTED")
	errTokenCountRequestTooLarge = errors.New("TOKEN_COUNT_REQUEST_TOO_LARGE")
)

type chatTokenBudget struct {
	InputTokens         int `json:"input_tokens"`
	MaxInputTokens      int `json:"max_input_tokens"`
	MaxCompletionTokens int `json:"max_completion_tokens"`
}

type chatTokenCount struct {
	Model string `json:"model"`
	chatTokenBudget
	Exact bool `json:"exact"`
}

func (p provider) chatTokenCount(ctx context.Context, in chatRequestBody) (any, error) {
	switch p.model.Provider {
	case NPUProvider:
		return p.npuTokenCount(ctx, in)
	case "ollama":
		settings := p.model.OllamaChat
		if settings == nil || settings.ContextTokens == nil {
			return nil, errTokenCountUnsupported
		}
		contextLimit, completionLimit := settings.tokenLimits(in.MaxCompletionTokens)
		if completionLimit >= contextLimit {
			return nil, errTokenCountUnsupported
		}
		return chatTokenCount{
			Model: in.Model,
			chatTokenBudget: chatTokenBudget{
				InputTokens:         estimateOllamaInputTokens(in),
				MaxInputTokens:      contextLimit - completionLimit,
				MaxCompletionTokens: completionLimit,
			},
			Exact: false,
		}, nil
	default:
		return nil, errTokenCountUnsupported
	}
}

func estimateOllamaInputTokens(in chatRequestBody) int {
	// Count UTF-8 bytes with allowances for common chat wrappers and special
	// tokens. The actual installed tokenizer and template are not inspected.
	const templateAllowance, messageAllowance = 256, 32
	estimate := templateAllowance
	for _, m := range in.Messages {
		estimate += len(m.Role) + len(m.Content) + messageAllowance
	}
	if in.ResponseFormat != nil {
		estimate += len(in.ResponseFormat.JSONSchema.Schema)
	}
	return estimate
}
