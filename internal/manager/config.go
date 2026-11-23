package manager

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net"
	"net/url"
	"os"
	"regexp"
	"strings"
)

type Config struct {
	Models []Model `json:"models"`
}

type Model struct {
	Name     string `json:"name"`
	Provider string `json:"provider"`
	BaseURL  string `json:"base_url"`
	Model    string `json:"model"`
	Role     string `json:"role"`
	Digest   string `json:"digest,omitempty"`
	Revision string `json:"revision,omitempty"`
	KeyEnv   string `json:"key_env,omitempty"`
}

// ID includes the adapter contract version, so a semantic change to inference
// can invalidate stored profiles even when the weights stay the same.
func (m Model) ID() string {
	identity, _ := json.Marshal([]string{"models-configurator-v1", m.Provider, strings.TrimRight(m.BaseURL, "/"), m.Model, m.Role, strings.TrimPrefix(m.Digest, "sha256:"), m.Revision})
	digest := sha256.Sum256(identity)
	return m.Provider + "." + m.Name + "." + hex.EncodeToString(digest[:])
}

func readBounded(r io.Reader, limit int64) ([]byte, error) {
	data, err := io.ReadAll(io.LimitReader(r, limit+1))
	if err != nil || int64(len(data)) > limit {
		return nil, errors.New("invalid body size")
	}
	return data, nil
}

func decodeStrict(data []byte, out any) error {
	d := json.NewDecoder(bytes.NewReader(data))
	d.DisallowUnknownFields()
	if err := d.Decode(out); err != nil {
		return err
	}
	if err := d.Decode(new(json.RawMessage)); err != io.EOF {
		return errors.New("unexpected trailing JSON")
	}
	return nil
}

func LoadConfig(path string) (Config, error) {
	f, err := os.Open(path)
	if err != nil {
		return Config{}, err
	}
	defer f.Close()
	data, err := readBounded(f, 1<<20)
	if err != nil {
		return Config{}, err
	}
	var cfg Config
	if err := decodeStrict(data, &cfg); err != nil {
		return Config{}, errors.New("invalid models configuration JSON")
	}
	return cfg, cfg.validate()
}

var (
	namePattern      = regexp.MustCompile(`^[a-zA-Z0-9][a-zA-Z0-9_.-]{0,79}$`)
	ollamaTagPattern = regexp.MustCompile(`^(?:[a-zA-Z0-9][a-zA-Z0-9._-]*:[0-9]+/)?(?:[a-zA-Z0-9][a-zA-Z0-9._-]*/)*[a-zA-Z0-9][a-zA-Z0-9._-]*:[a-zA-Z0-9_][a-zA-Z0-9._-]*$`)
	modelPattern     = regexp.MustCompile(`^[a-zA-Z0-9][a-zA-Z0-9_./:-]{0,199}$`)
	digestPattern    = regexp.MustCompile(`^(sha256:)?[a-f0-9]{64}$`)
	envPattern       = regexp.MustCompile(`^[A-Za-z_][A-Za-z0-9_]*$`)
)

func (cfg Config) validate() error {
	if cfg.Models == nil {
		return errors.New("configure a models array")
	}
	seen := map[string]bool{}
	for i, m := range cfg.Models {
		if err := m.validate(); err != nil {
			return fmt.Errorf("model %d: %w", i+1, err)
		}
		if seen[m.Name] {
			return fmt.Errorf("duplicate model name %q", m.Name)
		}
		seen[m.Name] = true
	}
	return nil
}

func (m Model) validate() error {
	if !namePattern.MatchString(m.Name) || !modelPattern.MatchString(m.Model) || (m.Role != "chat" && m.Role != "embeddings") {
		return errors.New("invalid model name, upstream model or role")
	}
	u, err := url.Parse(m.BaseURL)
	if err != nil || u.Hostname() == "" || u.User != nil || u.RawQuery != "" || u.ForceQuery || u.Fragment != "" || strings.Contains(m.BaseURL, "#") {
		return errors.New("invalid provider URL")
	}
	loopback := u.Hostname() == "localhost" || net.ParseIP(u.Hostname()).IsLoopback()
	switch m.Provider {
	case "ollama":
		host := u.Hostname()
		if (!loopback && host != "ollama" && host != "ollama-embeddings" && host != "ollama-generator") || u.Scheme != "http" || (u.EscapedPath() != "" && u.EscapedPath() != "/") {
			return errors.New("Ollama requires a local HTTP runtime")
		}
		if !ollamaTagPattern.MatchString(m.Model) {
			return errors.New("Ollama requires an explicit model tag")
		}
		tag := strings.ToLower(m.Model)
		if !digestPattern.MatchString(m.Digest) || m.Revision != "" || m.KeyEnv != "" || strings.HasSuffix(tag, ":cloud") || strings.HasSuffix(tag, "-cloud") {
			return errors.New("Ollama requires a local model with a full SHA-256 digest")
		}
	case "openai-compatible":
		if u.Scheme != "https" && u.Scheme != "http" {
			return errors.New("compatible provider requires HTTP or HTTPS")
		}
		if m.Revision == "" || strings.TrimSpace(m.Revision) != m.Revision || len(m.Revision) > 200 || strings.ContainsAny(m.Revision, "\r\n") || m.Digest != "" {
			return errors.New("compatible provider requires an explicit revision")
		}
		if m.KeyEnv != "" && !envPattern.MatchString(m.KeyEnv) {
			return errors.New("invalid credential environment variable name")
		}
	default:
		return errors.New("unsupported provider")
	}
	return nil
}

// OllamaTag expands the implicit latest tag sometimes returned by Ollama.
// Configured models use explicit tags so inventory and inference agree.
func OllamaTag(name string) string {
	if name != "" && strings.LastIndex(name, ":") <= strings.LastIndex(name, "/") {
		return name + ":latest"
	}
	return name
}
