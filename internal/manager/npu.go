package manager

import (
	"context"
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"errors"
	"io"
	"net"
	"net/http"
	"net/url"
	"path/filepath"
	"strings"
)

const NPUProvider = "openvino-npu"

const npuCallHeader = "X-NPU-Call-ID"

func newNPUCallID() string {
	var id [16]byte
	rand.Read(id[:])
	return hex.EncodeToString(id[:])
}

type npuFailure string

func (e npuFailure) Error() string { return string(e) }

func (e npuFailure) status() int {
	switch e {
	case "PROFILE_CHANGED":
		return http.StatusConflict
	case "NPU_UNAVAILABLE", "NPU_WORKER_EXIT", "NPU_RECOVERY_FAILED":
		return http.StatusServiceUnavailable
	case "TIMEOUT":
		return http.StatusGatewayTimeout
	case "INVALID_OUTPUT":
		return http.StatusBadGateway
	}
	return 0
}

// Only a bounded, known code paired with its runtime HTTP status crosses the socket boundary.
func parseNPUFailure(status int, body io.Reader) error {
	var response struct {
		Code    npuFailure `json:"code"`
		Message string     `json:"error"`
	}
	raw, err := readBounded(body, 4096)
	if err == nil && decodeStrict(raw, &response) == nil && response.Code.status() == status {
		return response.Code
	}
	return errors.New("NPU runtime request failed")
}

func socketPath(base string) (string, error) {
	u, err := url.Parse(base)
	if err != nil || !strings.HasPrefix(base, "unix:///") || u.Host != "" || u.User != nil ||
		u.RawQuery != "" || u.ForceQuery || strings.Contains(base, "#") || !filepath.IsAbs(u.Path) ||
		u.Path == "/" || u.Path != filepath.Clean(u.Path) || strings.ContainsAny(u.Path, "\x00\r\n") {
		return "", errors.New("OpenVINO NPU requires an absolute Unix socket URL")
	}
	return u.Path, nil
}

func npuUpstream(base string) (upstream, error) {
	path, err := socketPath(base)
	if err != nil {
		return upstream{}, err
	}
	return unixSocketUpstream(path), nil
}

// Resolve filesystem aliases when handling a call: the runtime socket may
// appear after manager startup. Both admission and dialing use this path.
func (u upstream) resolveSocket() upstream {
	if path, err := filepath.EvalSymlinks(u.socket); err == nil && path != u.socket {
		return unixSocketUpstream(path)
	}
	return u
}

func unixSocketUpstream(path string) upstream {
	transport := &http.Transport{
		DialContext: func(ctx context.Context, _, _ string) (net.Conn, error) {
			return (&net.Dialer{}).DialContext(ctx, "unix", path)
		},
		DisableKeepAlives:      true,
		MaxResponseHeaderBytes: 64 << 10,
	}
	client := &http.Client{Transport: transport, CheckRedirect: func(*http.Request, []*http.Request) error { return http.ErrUseLastResponse }}
	// HTTP framing is local to the socket selected by DialContext.
	return upstream{base: "http://localhost", socket: path, client: client}
}

type npuBinding struct {
	Profile string `json:"profile"`
	Device  string `json:"device"`
}

func (b npuBinding) check(expectedProfile string) error {
	if b.Device == "" || len(b.Profile) != 64 || !digestPattern.MatchString(b.Profile) {
		return npuFailure("INVALID_OUTPUT")
	}
	if b.Device != "NPU" || (expectedProfile != "" && b.Profile != expectedProfile) {
		return npuFailure("PROFILE_CHANGED")
	}
	return nil
}

// NPUStatus reads passive metadata. The runtime never starts a worker for /status.
func NPUStatus(ctx context.Context, base string) (profile string, loaded bool, err error) {
	target, err := npuUpstream(base)
	if err != nil {
		return "", false, err
	}
	var status struct {
		npuBinding
		Loaded *bool `json:"loaded"`
	}
	if err := target.requestBounded(ctx, http.MethodGet, "/status", nil, &status, 4096); err != nil {
		return "", false, err
	}
	if status.check("") != nil || status.Loaded == nil {
		return "", false, errors.New("invalid NPU runtime metadata")
	}
	return status.Profile, *status.Loaded, nil
}

func (p provider) npuChat(ctx context.Context, in chatRequestBody) (any, error) {
	digest := strings.TrimPrefix(p.model.Digest, "sha256:")
	var health npuBinding
	if err := p.requestBounded(ctx, http.MethodGet, "/health", nil, &health, 4096); err != nil {
		return nil, err
	}
	if err := health.check(digest); err != nil {
		return nil, err
	}
	body := struct {
		Profile             string          `json:"profile"`
		Messages            []message       `json:"messages"`
		Schema              json.RawMessage `json:"schema"`
		MaxCompletionTokens *int            `json:"max_completion_tokens,omitempty"`
	}{digest, in.Messages, in.ResponseFormat.JSONSchema.Schema, in.MaxCompletionTokens}
	var resp struct {
		npuBinding
		Content string `json:"content"`
	}
	if err := p.request(ctx, http.MethodPost, "/v1/generate", body, &resp); err != nil {
		return nil, err
	}
	if err := resp.check(digest); err != nil {
		return nil, err
	}
	if !json.Valid([]byte(resp.Content)) {
		return nil, npuFailure("INVALID_OUTPUT")
	}
	return chatResult(in.Model, "stop", outputMessage{Role: "assistant", Content: &resp.Content})
}
