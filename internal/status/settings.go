package status

import (
	"context"
	_ "embed"
	"encoding/json"
	"errors"
	"net/http"
	"slices"
	"strconv"
	"strings"

	"github.com/VolkovDani/models-configurator/internal/manager"
)

//go:embed settings.html
var settingsHTML string

var settingsPage = newPanelTemplate("settings", settingsHTML, nil)

type settingsEditor struct {
	cfg     manager.Config
	current manager.Settings
}

type settingsRow struct {
	Index    int
	Model    manager.Model
	Edit     manager.ProfileSettings
	Choices  []installedChoice
	Selected string
}

type installedChoice struct {
	Name, Value string
}

type settingsView struct {
	locale
	Parallel int
	Rows     []settingsRow
	Error    string
}

func RegisterSettings(mux *http.ServeMux, cfg manager.Config, current manager.Settings) {
	editor := settingsEditor{cfg: cfg, current: current}
	mux.HandleFunc("GET /settings", func(w http.ResponseWriter, r *http.Request) { editor.render(w, r, http.StatusOK, "") })
	mux.HandleFunc("POST /settings", editor.save)
	mux.HandleFunc("POST /settings/npu", editor.deriveNPU)
}

func (s settingsEditor) render(w http.ResponseWriter, r *http.Request, code int, message string) {
	words := localize(r)
	view := settingsView{locale: words, Parallel: s.current.MaxParallel, Error: message}
	for i, model := range s.cfg.Models {
		edit := s.current.Profiles[model.SettingsKey()]
		row := settingsRow{Index: i, Model: model, Edit: edit}
		if model.Provider == "ollama" && model.Role != "decision" {
			row.Choices = installedChoices(r.Context(), model.BaseURL)
			if edit.Model != "" {
				row.Selected = edit.Model + "@" + strings.TrimPrefix(edit.Digest, "sha256:")
				if !slices.ContainsFunc(row.Choices, func(choice installedChoice) bool { return choice.Value == row.Selected }) {
					row.Choices = append(row.Choices, installedChoice{edit.Model, row.Selected})
				}
			}
		}
		view.Rows = append(view.Rows, row)
	}
	w.Header().Set("Content-Type", "text/html; charset=utf-8")
	w.Header().Set("Content-Language", words.Lang)
	w.Header().Set("Content-Security-Policy", "default-src 'none'; style-src 'self' 'unsafe-inline'; script-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'self'")
	w.Header().Set("Referrer-Policy", "same-origin")
	w.WriteHeader(code)
	_ = settingsPage.Execute(w, view)
}

func installedChoices(ctx context.Context, base string) []installedChoice {
	models, _ := readList(ctx, strings.TrimRight(base, "/"), "/api/tags", "")
	var choices []installedChoice
	for _, model := range models {
		tag := model.Name
		if tag == "" {
			tag = model.Model
		}
		digest := strings.TrimPrefix(model.Digest, "sha256:")
		candidate := manager.Model{Name: "candidate", Role: "chat", Provider: "ollama", BaseURL: base, Model: tag, Digest: digest}
		if model.RemoteHost != "" || model.RemoteModel != "" || !candidate.ValidOllamaSelection() {
			continue
		}
		choices = append(choices, installedChoice{Name: tag, Value: tag + "@" + digest})
	}
	slices.SortFunc(choices, func(a, b installedChoice) int { return strings.Compare(a.Name, b.Name) })
	return choices
}

func (s settingsEditor) save(w http.ResponseWriter, r *http.Request) {
	r.Body = http.MaxBytesReader(w, r.Body, 128<<10)
	draft, err := s.parseSettings(r)
	if err != nil {
		s.render(w, r, http.StatusBadRequest, "settings_invalid")
		return
	}
	downloadJSON(w, "model-settings.json")
	enc := json.NewEncoder(w)
	enc.SetIndent("", "  ")
	_ = enc.Encode(draft)
}

func (s settingsEditor) parseSettings(r *http.Request) (manager.Settings, error) {
	if err := r.ParseForm(); err != nil {
		return manager.Settings{}, err
	}
	parallel, err := strconv.Atoi(r.PostForm.Get("parallel"))
	if err != nil {
		return manager.Settings{}, err
	}
	draft := manager.Settings{MaxParallel: parallel, Profiles: make(map[string]manager.ProfileSettings)}
	for i, model := range s.cfg.Models {
		field := func(name string) string { return r.PostForm.Get(strconv.Itoa(i) + "." + name) }
		edit := manager.ProfileSettings{Name: field("name"), Role: field("role"), Disabled: field("enabled") != "on"}
		if model.Provider == "ollama" && model.Role != "decision" {
			if selected := field("installed"); selected != "" {
				current := s.current.Profiles[model.SettingsKey()]
				previous := current.Model + "@" + strings.TrimPrefix(current.Digest, "sha256:")
				if selected != previous && !slices.ContainsFunc(installedChoices(r.Context(), model.BaseURL), func(choice installedChoice) bool { return choice.Value == selected }) {
					return manager.Settings{}, errors.New("Ollama selection is no longer installed")
				}
				edit.Model, edit.Digest, _ = strings.Cut(selected, "@")
			}
			if edit.Role == "chat" {
				edit.OllamaChat, err = chatDefaults(field)
				if err != nil {
					return manager.Settings{}, err
				}
			}
		}
		draft.Profiles[model.SettingsKey()] = edit
	}
	_, err = draft.Apply(s.cfg)
	return draft, err
}

func optionalNumber[T int | float64](value string, parse func(string) (T, error)) (*T, error) {
	if value == "" {
		return nil, nil
	}
	parsed, err := parse(value)
	if err != nil {
		return nil, err
	}
	return &parsed, nil
}

func chatDefaults(field func(string) string) (*manager.OllamaChatSettings, error) {
	var opt manager.OllamaChatSettings
	var err error
	opt.ContextTokens, err = optionalNumber(field("context"), strconv.Atoi)
	if err != nil {
		return nil, err
	}
	opt.CompletionTokens, err = optionalNumber(field("completion"), strconv.Atoi)
	if err != nil {
		return nil, err
	}
	float := func(s string) (float64, error) { return strconv.ParseFloat(s, 64) }
	opt.Temperature, err = optionalNumber(field("temperature"), float)
	if err != nil {
		return nil, err
	}
	opt.TopP, err = optionalNumber(field("top_p"), float)
	if err != nil || opt == (manager.OllamaChatSettings{}) {
		return nil, err
	}
	return &opt, nil
}

func downloadJSON(w http.ResponseWriter, name string) {
	w.Header().Set("Content-Type", "application/json")
	w.Header().Set("Content-Disposition", `attachment; filename="`+name+`"`)
}
