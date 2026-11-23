package status

var translations = map[string]map[string]string{
	"en": {
		"empty": "No models configured. Add a profile to the configuration file and restart Models Configurator.",
		"title": "Model inventory", "languages": "Language", "checked": "Last checked", "refresh": "Refresh",
		"upstream": "Upstream model", "id": "API model ID", "role": "Role", "provider": "Provider", "listed": "Listed by API",
		"installed": "Installed", "loaded": "Loaded", "memory": "Memory placement", "reachable": "Metadata available",
		"unreachable": "Unreachable", "metadata_error": "Metadata unavailable", "credential_missing": "Credential missing", "unchecked": "Unchecked",
		"unknown": "Unknown", "yes": "Yes", "no": "No", "cpu": "CPU", "gpu": "GPU", "cpu_gpu": "CPU + GPU",
		"openvino-npu": "OpenVINO NPU", "ollama": "Ollama", "openai-compatible": "OpenAI-compatible API", "chat": "Chat", "embeddings": "Embeddings",
	},
	"ru": {
		"empty": "Модели не настроены. Добавьте профиль в файл конфигурации и перезапустите Models Configurator.",
		"title": "Модели", "languages": "Язык", "checked": "Последняя проверка", "refresh": "Обновить",
		"upstream": "Модель у провайдера", "id": "Идентификатор для API", "role": "Назначение", "provider": "Провайдер", "listed": "В списке API",
		"installed": "Установлена", "loaded": "Загружена", "memory": "Размещение в памяти", "reachable": "Метаданные доступны",
		"unreachable": "Нет связи", "metadata_error": "Метаданные недоступны", "credential_missing": "Нет ключа доступа", "unchecked": "Проверка не выполнена",
		"unknown": "Неизвестно", "yes": "Да", "no": "Нет", "cpu": "CPU", "gpu": "GPU", "cpu_gpu": "CPU + GPU",
		"openvino-npu": "OpenVINO NPU", "ollama": "Ollama", "openai-compatible": "OpenAI-совместимый API", "chat": "Диалог", "embeddings": "Эмбеддинги",
	},
}
