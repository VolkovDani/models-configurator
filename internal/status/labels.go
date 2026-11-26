package status

import "net/http"

var translations = map[string]map[string]string{
	"en": {
		"settings_title": "Model settings", "settings_restart": "Save a settings file, then restart Models Configurator with that file to apply it.",
		"settings_manager": "Manager", "settings_parallel": "Concurrent requests", "settings_name": "Profile name",
		"settings_installed": "Installed Ollama model", "settings_configured": "deployment default", "settings_context": "Chat context tokens",
		"settings_completion": "Maximum response tokens", "settings_temperature": "Temperature", "settings_top_p": "Top P",
		"settings_enabled": "Available to clients", "settings_save": "Save settings file",
		"settings_invalid":  "Check the profile names, roles, selected models and numeric limits.",
		"settings_manifest": "NPU token limits", "settings_manifest_current": "Current pinned manifest", "settings_prompt": "Maximum prompt tokens",
		"settings_manifest_help": "The download contains a new manifest. Its filename shows the SHA-256 to pin. Restart the NPU runtime and manager to apply it.",
		"settings_manifest_save": "Save new manifest", "settings_invalid_manifest": "Upload the manifest pinned by this NPU profile and enter valid token limits.",
		"empty":         "No models configured. Add a profile to the configuration file and restart Models Configurator.",
		"history_title": "Model call history", "history_chart_title": "Duration and tokens", "history_usage_type": "Usage type", "history_latest": "Latest calls", "history_time": "Started (UTC)", "history_profile": "Model profile",
		"history_input_tokens": "Input tokens", "history_output_tokens": "Output tokens", "history_total_tokens": "Total tokens",
		"history_token_metric": "Token count", "history_no_token_data": "No token counts are available for this operation.",
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
		"settings_title": "Настройки моделей", "settings_restart": "Сохраните файл настроек и перезапустите Models Configurator с этим файлом, чтобы применить изменения.",
		"settings_manager": "Менеджер", "settings_parallel": "Одновременные запросы", "settings_name": "Имя профиля",
		"settings_installed": "Установленная модель Ollama", "settings_configured": "из конфигурации", "settings_context": "Размер контекста в токенах",
		"settings_completion": "Максимум токенов ответа", "settings_temperature": "Температура", "settings_top_p": "Top P",
		"settings_enabled": "Доступен клиентам", "settings_save": "Сохранить файл настроек",
		"settings_invalid":  "Проверьте имена и назначения профилей, выбранные модели и числовые лимиты.",
		"settings_manifest": "Лимиты токенов NPU", "settings_manifest_current": "Текущий закреплённый манифест", "settings_prompt": "Максимум токенов входа",
		"settings_manifest_help": "Скачанный файл содержит новый манифест. SHA-256 для привязки указан в имени файла. Для применения перезапустите NPU runtime и менеджер.",
		"settings_manifest_save": "Сохранить новый манифест", "settings_invalid_manifest": "Загрузите манифест, закреплённый за этим профилем NPU, и укажите допустимые лимиты токенов.",
		"empty":         "Модели не настроены. Добавьте профиль в файл конфигурации и перезапустите Models Configurator.",
		"history_title": "История вызовов", "history_chart_title": "Длительность и токены", "history_usage_type": "Тип использования", "history_latest": "Последние вызовы", "history_time": "Начало (UTC)", "history_profile": "Профиль модели",
		"history_input_tokens": "Входные токены", "history_output_tokens": "Выходные токены", "history_total_tokens": "Всего токенов",
		"history_token_metric": "Счётчик токенов", "history_no_token_data": "Для этой операции пока нет данных о токенах.",
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
