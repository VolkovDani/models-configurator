package manager

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"math"
	"net/http"
	"slices"
	"strconv"
	"strings"
	"unicode/utf8"
)

const (
	decisionRequestLimit  = 64 << 10
	decisionResponseLimit = 1 << 20
	decisionOllamaVersion = "0.40.16"
	layaDecisionDigest    = "c1242e4a83f0e1cb58f0f191a65bbc5d3306fae274d152b8229ec574999d465d"
	d3DecisionDigest      = "2c571f91caf64c47b6ca6bd3adc1daa07dbb84ead8e9489758498cc7e11e533d"
)

type decisionQuestion struct {
	Type         string            `json:"type"`
	Instructions *string           `json:"instructions"`
	Criteria     map[string]string `json:"criteria"`
}

type decisionRequest struct {
	Model    string           `json:"model"`
	Digest   string           `json:"digest"`
	State    string           `json:"state"`
	Question decisionQuestion `json:"question"`
}

func (in *decisionRequest) UnmarshalJSON(data []byte) error {
	type request decisionRequest
	if err := json.Unmarshal(data, (*request)(in)); err != nil {
		return err
	}
	return uniqueDecisionJSON(data, decisionRequestFields)
}

func (in decisionRequest) valid() bool {
	q := in.Question
	if in.Model == "" || len(in.Digest) != 64 || !digestPattern.MatchString(in.Digest) || strings.TrimSpace(in.State) == "" ||
		q.Type != "choice" || q.Instructions == nil || len(q.Criteria) < 2 {
		return false
	}
	for key, criterion := range q.Criteria {
		if strings.TrimSpace(key) == "" || strings.TrimSpace(criterion) == "" {
			return false
		}
	}
	return true
}

type decisionAnswer struct {
	Type          string              `json:"type"`
	Choice        string              `json:"choice"`
	Confidence    *float64            `json:"confidence"`
	Probabilities map[string]*float64 `json:"probabilities"`
}

type decisionUsage struct {
	InputTokens  *int `json:"input_tokens"`
	OutputTokens *int `json:"output_tokens"`
}

type decisionReply struct {
	Model  string          `json:"model"`
	Digest string          `json:"digest"`
	Answer *decisionAnswer `json:"answer,omitempty"`
	Usage  *decisionUsage  `json:"usage,omitempty"`
	Reason string          `json:"reason,omitempty"`
}

type ollamaDecisionReply struct {
	Model       string                    `json:"model"`
	RemoteModel string                    `json:"remote_model"`
	RemoteHost  string                    `json:"remote_host"`
	Answers     map[string]decisionAnswer `json:"answers"`
	Usage       decisionUsage             `json:"usage"`
}

func (out *ollamaDecisionReply) UnmarshalJSON(data []byte) error {
	type reply ollamaDecisionReply
	if err := json.Unmarshal(data, (*reply)(out)); err != nil {
		return err
	}
	return uniqueDecisionJSON(data, nil)
}

type decisionFields map[string]decisionFields

var decisionRequestFields = decisionFields{
	"model": nil, "digest": nil, "state": nil,
	"question": {"type": nil, "instructions": nil, "criteria": nil},
}

// encoding/json accepts duplicate keys by default. Decision data must have an
// unambiguous interpretation on both sides of the adapter.
func uniqueDecisionJSON(data []byte, fields decisionFields) error {
	if !utf8.Valid(data) {
		return errors.New("invalid UTF-8 JSON")
	}
	d := json.NewDecoder(bytes.NewReader(data))
	d.UseNumber()
	return checkDecisionValue(d, fields)
}

func checkDecisionValue(d *json.Decoder, fields decisionFields) error {
	token, err := d.Token()
	if err != nil {
		return err
	}
	switch token {
	case json.Delim('{'):
		keys := map[string]bool{}
		for d.More() {
			key, err := d.Token()
			if err != nil {
				return err
			}
			name := key.(string)
			if keys[name] {
				return errors.New("duplicate decision JSON field")
			}
			childFields, known := fields[name]
			if fields != nil && !known {
				return errors.New("unknown decision JSON field")
			}
			keys[name] = true
			if err := checkDecisionValue(d, childFields); err != nil {
				return err
			}
		}
	case json.Delim('['):
		for d.More() {
			if err := checkDecisionValue(d, nil); err != nil {
				return err
			}
		}
	default:
		return nil
	}
	_, err = d.Token()
	return err
}

type measuredDecisionProfile struct {
	metadata          map[string]any
	quantization      string
	configuredContext int
	trainingContext   int
	reserved          []string
}

// Measured limits and control tokens attach to a weight digest, including
// installations using another local tag.
var measuredDecisionProfiles = map[string]measuredDecisionProfile{
	layaDecisionDigest: {
		metadata:        map[string]any{"modern-bert.decision.type": "laya", "modern-bert.decision.max_head_tokens": float64(256)},
		trainingContext: 1024,
		reserved:        []string{"<bos>", "<eos>", "<mask>"},
	},
	d3DecisionDigest: {
		metadata:          map[string]any{"qwen35.decision.type": "vllm-d3"},
		quantization:      "Q8_0",
		configuredContext: 4096,
		reserved: []string{
			"<|endoftext|>", "<|im_start|>", "<|im_end|>", "<|object_ref_start|>",
			"<|object_ref_end|>", "<|box_start|>", "<|box_end|>", "<|quad_start|>",
			"<|quad_end|>", "<|vision_start|>", "<|vision_end|>", "<|vision_pad|>",
			"<|image_pad|>", "<|video_pad|>", "<tool_call>", "</tool_call>",
			"<|fim_prefix|>", "<|fim_middle|>", "<|fim_suffix|>", "<|fim_pad|>",
			"<|repo_name|>", "<|file_sep|>", "<tool_response>", "</tool_response>",
			"<think>", "</think>", "<|audio_start|>", "<|audio_end|>",
			"<tts_pad>", "<tts_text_bos>", "<tts_text_eod>", "<tts_text_bos_single>",
			"<|audio_pad|>",
		},
	},
}

func (m measuredDecisionProfile) hasReservedToken(in decisionRequest) bool {
	for _, token := range m.reserved {
		if strings.Contains(in.State, token) || strings.Contains(*in.Question.Instructions, token) {
			return true
		}
		for key, criterion := range in.Question.Criteria {
			if strings.Contains(key, token) || strings.Contains(criterion, token) {
				return true
			}
		}
	}
	return false
}

func configuredDecisionContext(parameters string) int {
	limit := 0
	for line := range strings.SplitSeq(parameters, "\n") {
		fields := strings.Fields(line)
		if len(fields) == 0 || fields[0] != "num_ctx" {
			continue
		}
		if len(fields) != 2 || limit != 0 {
			return 0
		}
		n, err := strconv.Atoi(fields[1])
		if err != nil || n <= 0 {
			return 0
		}
		limit = n
	}
	return limit
}

func (p provider) verifyDecisionRuntime(ctx context.Context, measured measuredDecisionProfile) (int, error) {
	if err := p.verify(ctx); err != nil {
		return 0, err
	}
	var version struct {
		Version string `json:"version"`
	}
	if err := p.requestBounded(ctx, http.MethodGet, "/api/version", nil, &version, decisionResponseLimit); err != nil {
		return 0, err
	}
	if version.Version != decisionOllamaVersion {
		return 0, errors.New("unsupported local decision runtime version")
	}
	var show struct {
		Parameters string `json:"parameters"`
		Details    struct {
			Format       string `json:"format"`
			Runner       string `json:"runner"`
			Quantization string `json:"quantization_level"`
		} `json:"details"`
		Info         map[string]any `json:"model_info"`
		Capabilities []string       `json:"capabilities"`
		RemoteModel  string         `json:"remote_model"`
		RemoteHost   string         `json:"remote_host"`
	}
	payload := map[string]string{"model": p.model.Model}
	if err := p.requestBounded(ctx, http.MethodPost, "/api/show", payload, &show, decisionResponseLimit); err != nil {
		return 0, err
	}
	if !slices.Contains(show.Capabilities, "decision") || show.RemoteHost != "" || show.RemoteModel != "" {
		return 0, errors.New("unsupported local decision runtime")
	}
	if len(measured.metadata) > 0 && (show.Details.Format != "gguf" || show.Details.Runner != "llamacpp") {
		return 0, errors.New("unsupported measured decision renderer")
	}
	for key, expected := range measured.metadata {
		if show.Info[key] != expected {
			return 0, errors.New("decision model metadata mismatch")
		}
	}
	if measured.quantization != "" && show.Details.Quantization != measured.quantization {
		return 0, errors.New("decision model quantization mismatch")
	}
	if measured.trainingContext != 0 {
		// Laya has an inclusive training limit independent of configured num_ctx.
		return 0, nil
	}
	limit := configuredDecisionContext(show.Parameters)
	if limit == 0 || measured.configuredContext != 0 && limit != measured.configuredContext {
		return 0, errors.New("invalid configured decision context")
	}
	return limit, nil
}

func (p provider) decide(ctx context.Context, in decisionRequest) (decisionReply, error) {
	out := decisionReply{Model: in.Model, Digest: in.Digest}
	measured := measuredDecisionProfiles[in.Digest]
	if measured.hasReservedToken(in) {
		out.Reason = "reserved_token"
		return out, nil
	}
	body, err := json.Marshal(map[string]any{
		"model": p.model.Model, "state": in.State, "keep_alive": "5m",
		"questions": map[string]decisionQuestion{"classification": in.Question},
	})
	if err != nil {
		return decisionReply{}, err
	}
	if len(body) > decisionRequestLimit {
		out.Reason = "input_limit"
		return out, nil
	}
	contextLimit, err := p.verifyDecisionRuntime(ctx, measured)
	if err != nil {
		return decisionReply{}, err
	}
	var res ollamaDecisionReply
	err = p.requestBounded(ctx, http.MethodPost, "/v1/systemone", json.RawMessage(body), &res, decisionResponseLimit)
	tooLarge := err == upstreamStatus(http.StatusRequestEntityTooLarge)
	if err != nil && !tooLarge {
		return decisionReply{}, err
	}
	if err := p.verify(ctx); err != nil {
		return decisionReply{}, err
	}
	if tooLarge {
		out.Reason = "input_limit"
		return out, nil
	}
	answer := res.Answers["classification"]
	if res.Model != p.model.Model || res.RemoteHost != "" || res.RemoteModel != "" || len(res.Answers) != 1 ||
		res.Usage.InputTokens == nil || *res.Usage.InputTokens < 0 || res.Usage.OutputTokens == nil || *res.Usage.OutputTokens < 0 ||
		!answer.valid(in.Question.Criteria) {
		return decisionReply{}, errors.New("invalid local decision output")
	}
	switch {
	case contextLimit > 0 && *res.Usage.InputTokens >= contextLimit:
		out.Reason = "input_limit"
	case measured.trainingContext > 0 && *res.Usage.InputTokens > measured.trainingContext:
		out.Reason = "outside_training_context"
	default:
		out.Answer, out.Usage = &answer, &res.Usage
	}
	return out, nil
}

func (a decisionAnswer) valid(criteria map[string]string) bool {
	if a.Type != "choice" || criteria[a.Choice] == "" || !validDecisionScore(a.Confidence) || len(a.Probabilities) != len(criteria) {
		return false
	}
	total := 0.0
	for key := range criteria {
		score := a.Probabilities[key]
		if !validDecisionScore(score) {
			return false
		}
		total += *score
	}
	return math.Abs(total-1) <= 1e-6
}

func validDecisionScore(score *float64) bool {
	return score != nil && *score >= 0 && *score <= 1
}
