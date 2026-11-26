package status

import "net/http"

var translations = map[string]map[string]string{
	"en": {
		"empty":         "No models configured. Add a profile to the configuration file and restart Models Configurator.",
		"history_title": "Model call history", "history_chart_title": "Call durations", "history_latest": "Latest calls", "history_time": "Started (UTC)", "history_profile": "Model profile",
		"history_operation": "Operation", "history_outcome": "Result", "history_duration": "Duration", "history_ms": "ms",
		"history_empty": "No model calls recorded yet.", "history_end": "No older calls.", "history_more": "Load more",
		"history_load_error": "Could not load older calls. Try again.", "history_loaded": "Older calls loaded.", "history_cursor_error": "Invalid history cursor.",
		"bge_encode": "BGE-M3 encoding", "succeeded": "Succeeded", "failed": "Failed", "refused": "Refused", "timed_out": "Timed out", "canceled": "Canceled",
		"title": "Model inventory", "languages": "Language", "checked": "Last checked", "refresh": "Refresh",
		"api_docs": "API documentation",
		"upstream": "Upstream model", "id": "API model ID", "role": "Role", "provider": "Provider", "listed": "Listed by API",
		"installed": "Installed", "loaded": "Loaded", "memory": "Memory placement", "reachable": "Metadata available",
		"unreachable": "Unreachable", "metadata_error": "Metadata unavailable", "credential_missing": "Credential missing", "unchecked": "Unchecked",
		"unknown": "Unknown", "yes": "Yes", "no": "No", "cpu": "CPU", "gpu": "GPU", "cpu_gpu": "CPU + GPU",
		"openvino-bge": "OpenVINO BGE-M3 (GPU)", "openvino-npu": "OpenVINO NPU", "ollama": "Ollama", "openai-compatible": "OpenAI-compatible API", "chat": "Chat", "embeddings": "Embeddings", "decision": "Decision",
	},
	"ru": {
		"empty":         "Модели не настроены. Добавьте профиль в файл конфигурации и перезапустите Models Configurator.",
		"history_title": "История вызовов", "history_chart_title": "Длительность вызовов", "history_latest": "Последние вызовы", "history_time": "Начало (UTC)", "history_profile": "Профиль модели",
		"history_operation": "Операция", "history_outcome": "Результат", "history_duration": "Длительность", "history_ms": "мс",
		"history_empty": "Вызовов моделей пока нет.", "history_end": "Более ранних вызовов нет.", "history_more": "Загрузить ещё",
		"history_load_error": "Не удалось загрузить вызовы. Попробуйте ещё раз.", "history_loaded": "Более ранние вызовы загружены.", "history_cursor_error": "Некорректный курсор истории.",
		"bge_encode": "Кодирование BGE-M3", "succeeded": "Успешно", "failed": "Ошибка", "refused": "Отказ", "timed_out": "Тайм-аут", "canceled": "Отменён",
		"title": "Модели", "languages": "Язык", "checked": "Последняя проверка", "refresh": "Обновить",
		"api_docs": "Документация API",
		"upstream": "Модель у провайдера", "id": "Идентификатор для API", "role": "Назначение", "provider": "Провайдер", "listed": "В списке API",
		"installed": "Установлена", "loaded": "Загружена", "memory": "Размещение в памяти", "reachable": "Метаданные доступны",
		"unreachable": "Нет связи", "metadata_error": "Метаданные недоступны", "credential_missing": "Нет ключа доступа", "unchecked": "Проверка не выполнена",
		"unknown": "Неизвестно", "yes": "Да", "no": "Нет", "cpu": "CPU", "gpu": "GPU", "cpu_gpu": "CPU + GPU",
		"openvino-bge": "OpenVINO BGE-M3 (GPU)", "openvino-npu": "OpenVINO NPU", "ollama": "Ollama", "openai-compatible": "OpenAI-совместимый API", "chat": "Диалог", "embeddings": "Эмбеддинги", "decision": "Выбор варианта",
	},
}

type locale struct {
	Lang  string
	Words map[string]string
}

func localize(r *http.Request) locale {
	lang := r.URL.Query().Get("lang")
	if lang != "ru" {
		lang = "en"
	}
	return locale{Lang: lang, Words: translations[lang]}
}

func (v locale) Label(code string) string { return v.Words[code] }
