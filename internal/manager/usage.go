package manager

import "encoding/json"

type tokenUsage struct {
	input  *int
	output *int
}

// Inference replies and history metadata travel together inside the manager.
type measuredResult struct {
	body  any
	usage tokenUsage
}

func tokenCount(raw json.RawMessage) *int {
	var count *int
	if len(raw) == 0 || json.Unmarshal(raw, &count) != nil || count == nil || *count < 0 {
		return nil
	}
	return count
}

func compatibleUsage(raw json.RawMessage, embeddings bool) tokenUsage {
	var counts struct {
		Prompt     json.RawMessage `json:"prompt_tokens"`
		Completion json.RawMessage `json:"completion_tokens"`
		Total      json.RawMessage `json:"total_tokens"`
	}
	if len(raw) == 0 || json.Unmarshal(raw, &counts) != nil {
		return tokenUsage{}
	}
	input := tokenCount(counts.Prompt)
	if embeddings {
		if len(counts.Prompt) == 0 {
			input = tokenCount(counts.Total)
		}
		return tokenUsage{input: input}
	}
	return tokenUsage{input: input, output: tokenCount(counts.Completion)}
}
