package status

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"os"
	"strings"
	"time"
)

const metadataLimit = 1 << 20

var metadataClient = &http.Client{
	Timeout:       2 * time.Second,
	Transport:     &http.Transport{MaxIdleConns: 16, IdleConnTimeout: 30 * time.Second, MaxResponseHeaderBytes: 64 << 10},
	CheckRedirect: func(*http.Request, []*http.Request) error { return http.ErrUseLastResponse },
}

type endpoint struct {
	provider, base, keyEnv string
}

type modelInfo struct {
	RemoteModel string `json:"remote_model"`
	RemoteHost  string `json:"remote_host"`
	Digest      string `json:"digest"`
	Name        string `json:"name"`
	Model       string `json:"model"`
	ID          string `json:"id"`
	Size        int64  `json:"size"`
	VRAM        *int64 `json:"size_vram"`
}

type inventory struct {
	provider, reachability string
	available, running     []modelInfo
}

type State struct {
	Reachability string `json:"reachability"`
	Listed       *bool  `json:"listed"`
	Installed    *bool  `json:"installed"`
	Loaded       *bool  `json:"loaded"`
	Memory       string `json:"memory"`
}

func probe(ctx context.Context, e endpoint) inventory {
	i := inventory{provider: e.provider, reachability: "unreachable"}
	var credential string
	if e.keyEnv != "" {
		credential = os.Getenv(e.keyEnv)
		if credential == "" {
			i.reachability = "credential_missing"
			return i
		}
	}
	if e.provider == "openai-compatible" {
		i.available, i.reachability = readList(ctx, e.base, "/models", credential)
		return i
	}
	i.available, i.reachability = readList(ctx, e.base, "/api/tags", credential)
	var psStatus string
	i.running, psStatus = readList(ctx, e.base, "/api/ps", credential)
	if psStatus == "reachable" || i.reachability == "unreachable" {
		i.reachability = psStatus
	}
	return i
}

func readList(ctx context.Context, base, path, credential string) ([]modelInfo, string) {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, strings.TrimRight(base, "/")+path, nil)
	if err != nil {
		return nil, "unreachable"
	}
	if credential != "" {
		req.Header.Set("Authorization", "Bearer "+credential)
	}
	res, err := metadataClient.Do(req)
	if err != nil {
		return nil, "unreachable"
	}
	defer res.Body.Close()
	if res.StatusCode != http.StatusOK {
		return nil, "metadata_error"
	}
	b, err := io.ReadAll(io.LimitReader(res.Body, metadataLimit+1))
	if err != nil || len(b) > metadataLimit {
		return nil, "metadata_error"
	}
	var reply struct {
		Models []modelInfo `json:"models"`
		Data   []modelInfo `json:"data"`
	}
	if json.Unmarshal(b, &reply) != nil {
		return nil, "metadata_error"
	}
	list := reply.Models
	if path == "/models" {
		list = reply.Data
	}
	if list == nil {
		return nil, "metadata_error"
	}
	for _, m := range list {
		if path == "/models" && m.ID == "" || path != "/models" && m.Name == "" && m.Model == "" {
			return nil, "metadata_error"
		}
	}
	return list, "reachable"
}

func (i inventory) state(name, digest string) State {
	s := State{Reachability: i.reachability, Memory: "unknown"}
	if i.provider == "openai-compatible" {
		if i.available != nil {
			listed := false
			for _, entry := range i.available {
				if entry.ID == name {
					listed = true
					break
				}
			}
			s.Listed = &listed
		}
		return s
	}
	s.Installed, _ = findOllama(i.available, name, digest)
	var active modelInfo
	s.Loaded, active = findOllama(i.running, name, digest)
	if active.VRAM == nil || active.Size <= 0 || *active.VRAM < 0 || *active.VRAM > active.Size {
		return s
	}
	switch *active.VRAM {
	case 0:
		s.Memory = "cpu"
	case active.Size:
		s.Memory = "gpu"
	default:
		s.Memory = "cpu_gpu"
	}
	return s
}

func findOllama(list []modelInfo, name, digest string) (*bool, modelInfo) {
	found := false
	if list == nil {
		return nil, modelInfo{}
	}
	unknown := false
	for _, entry := range list {
		if entry.RemoteModel != "" || entry.RemoteHost != "" {
			continue
		}
		if ollamaTag(entry.Name) != ollamaTag(name) && ollamaTag(entry.Model) != ollamaTag(name) {
			continue
		}
		if digest != "" {
			if entry.Digest == "" {
				unknown = true
				continue
			}
			if strings.TrimPrefix(entry.Digest, "sha256:") != strings.TrimPrefix(digest, "sha256:") {
				continue
			}
		}
		found = true
		return &found, entry
	}
	if unknown {
		return nil, modelInfo{}
	}
	return &found, modelInfo{}
}

func ollamaTag(name string) string {
	if name != "" && strings.LastIndex(name, ":") <= strings.LastIndex(name, "/") {
		return name + ":latest"
	}
	return name
}
