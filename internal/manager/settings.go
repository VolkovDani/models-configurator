package manager

import (
	"errors"
	"fmt"
	"os"
	"slices"
)

// Settings is an operator draft tied to the profiles in the deployment inventory.
type Settings struct {
	MaxParallel int                        `json:"max_parallel_requests"`
	Profiles    map[string]ProfileSettings `json:"profiles"`
}

type ProfileSettings struct {
	Name       string              `json:"name"`
	Role       string              `json:"role"`
	Disabled   bool                `json:"disabled,omitempty"`
	Model      string              `json:"model,omitempty"`
	Digest     string              `json:"digest,omitempty"`
	OllamaChat *OllamaChatSettings `json:"ollama_chat,omitempty"`
}

type OllamaChatSettings struct {
	ContextTokens    *int     `json:"context_tokens,omitempty"`
	CompletionTokens *int     `json:"completion_tokens,omitempty"`
	Temperature      *float64 `json:"temperature,omitempty"`
	TopP             *float64 `json:"top_p,omitempty"`
}

func ReadSettings(path string) (Settings, error) {
	file, err := os.Open(path)
	if err != nil {
		return Settings{}, err
	}
	defer file.Close()
	data, err := readBounded(file, 1<<20)
	if err != nil {
		return Settings{}, err
	}
	var draft Settings
	if err := decodeStrict(data, &draft); err != nil {
		return Settings{}, errors.New("invalid model settings JSON")
	}
	return draft, nil
}

func (cfg Config) Settings(maxParallel int) Settings {
	settings := Settings{MaxParallel: maxParallel, Profiles: make(map[string]ProfileSettings, len(cfg.Models))}
	for _, model := range cfg.Models {
		settings.Profiles[model.SettingsKey()] = ProfileSettings{Name: model.Name, Role: model.Role, OllamaChat: model.OllamaChat}
	}
	return settings
}

// Apply validates the complete inventory before dropping disabled profiles.
// Deployment-owned fields are always read from the original configuration.
func (s Settings) Apply(cfg Config) (Config, error) {
	if s.MaxParallel < 1 || len(s.Profiles) != len(cfg.Models) {
		return Config{}, errors.New("invalid concurrency or outdated profile inventory")
	}
	models := slices.Clone(cfg.Models)
	active := make([]Model, 0, len(models))
	for i, model := range models {
		edit, ok := s.Profiles[model.SettingsKey()]
		if !ok {
			return Config{}, errors.New("settings refer to a different profile inventory")
		}
		if edit.Model != "" || edit.Digest != "" {
			if model.Provider != "ollama" || model.Role == "decision" || edit.Model == "" || edit.Digest == "" {
				return Config{}, fmt.Errorf("invalid installed model selection for %q", model.Name)
			}
			model.Model, model.Digest = edit.Model, edit.Digest
		}
		if !slices.Contains(model.EditableRoles(), edit.Role) {
			return Config{}, fmt.Errorf("unsupported role for %q", model.Name)
		}
		model.Name, model.Role, model.OllamaChat = edit.Name, edit.Role, edit.OllamaChat
		models[i] = model
		if !edit.Disabled {
			active = append(active, model)
		}
	}
	if err := (Config{Models: models}).validate(); err != nil {
		return Config{}, err
	}
	return Config{Models: active}, nil
}

// SettingsKey binds a draft to its deployment. An NPU manifest rotation can
// happen alongside a settings edit without invalidating that model's draft.
func (m Model) SettingsKey() string {
	if m.Provider == NPUProvider {
		m.Digest = ""
	}
	return m.ID()
}

// ValidOllamaSelection applies the same constraints as config loading.
func (m Model) ValidOllamaSelection() bool {
	return m.Provider == "ollama" && m.validate() == nil
}

// Ollama can serve chat or embeddings with a compatible installed model.
// A decision profile depends on its own model and protocol capabilities.
func (m Model) EditableRoles() []string {
	if m.Provider == "ollama" && m.Role != "decision" {
		return []string{"chat", "embeddings"}
	}
	return []string{m.Role}
}

const (
	ollamaDefaultContext    = 32768
	ollamaDefaultCompletion = 1024
)

func (s *OllamaChatSettings) tokenLimits() (context, completion int) {
	context, completion = ollamaDefaultContext, ollamaDefaultCompletion
	if s != nil && s.ContextTokens != nil {
		context = *s.ContextTokens
	}
	if s != nil && s.CompletionTokens != nil {
		completion = *s.CompletionTokens
	}
	return
}

func (s *OllamaChatSettings) valid() bool {
	context, completion := s.tokenLimits()
	return context >= 1 && context <= 32768 &&
		completion >= 1 && completion <= 16384 && completion < context &&
		(s.Temperature == nil || *s.Temperature >= 0 && *s.Temperature <= 2) &&
		(s.TopP == nil || *s.TopP > 0 && *s.TopP <= 1)
}

func (s *OllamaChatSettings) options() map[string]any {
	context, completion := s.tokenLimits()
	options := map[string]any{"num_ctx": context, "temperature": 0, "num_predict": completion}
	if s == nil {
		return options
	}
	if s.Temperature != nil {
		options["temperature"] = *s.Temperature
	}
	if s.TopP != nil {
		options["top_p"] = *s.TopP
	}
	return options
}
