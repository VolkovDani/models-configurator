# Локальные модели выбора

[English version](decision-models.md)

Профиль с ролью `decision` доступен через `POST /v1/decisions`. Менеджер поддерживает локальные модели выбора Ollama 0.40.16, закреплённые по digest из `/api/tags`. Project Creator формирует вопрос, интерпретирует оценки модели и применяет правила ревью.

## Запуск локального Ollama

Базовый файл Compose содержит профиль `cpu`, который включается явно. Для Intel Arc используйте Docker Compose 2.24.4 или новее. Добавьте `compose.ollama-arc.yml` к действующему списку файлов и включите профиль `arc`. Ниже пример для установки только с базовым файлом.

```bash
export COMPOSE_FILE=compose.yml:compose.ollama-arc.yml
export COMPOSE_PROFILES=arc
export MODELS_OLLAMA_DRI_DEVICE=/dev/dri/renderD128
export MODELS_OLLAMA_RENDER_GID=$(stat -c %g "$MODELS_OLLAMA_DRI_DEVICE")
docker compose up -d --build ollama
docker compose exec ollama ollama pull d3:4b
```

Выберите render node своего Intel GPU. Для запуска через единый CLI Project Creator сохраните настройки в приватном `.env` менеджера. Для CPU используйте базовый файл и профиль `cpu`. У сервиса Ollama приватная сеть и отдельный постоянный том для моделей. При переносе уже закреплённых весов проверьте digest после запуска среды с импортированными файлами. Имя тега само по себе не подтверждает идентичность.

Добавьте профиль в существующий массив `models` в `models.json` менеджера.

```json
{"name":"d3-review","provider":"ollama","base_url":"http://ollama:11434","model":"d3:4b","role":"decision","digest":"<полный проверенный digest Ollama>"}
```

Запустите или пересоздайте менеджер командой `docker compose up -d --build --force-recreate models-configurator` и возьмите полный ID профиля из `GET /v1/models`. Для Laya добавьте отдельную запись с моделью `laya:322m-multilingual-bf16` и её точным digest.

## Запрос выбора

Передайте один вопрос с вариантами и ожидаемый digest артефакта из 64 строчных шестнадцатеричных символов. В поле `model` укажите неизменяемый ID из списка менеджера.

```json
{"model":"<ID профиля менеджера>","digest":"<проверенный digest артефакта>","state":"Администратор может изменить лимит платежа.","question":{"type":"choice","instructions":"Что описано в тексте?","criteria":{"a":"Разрешение","b":"Время ответа"}}}
```

Успешный ответ содержит ID профиля, digest, вариант ответа и число токенов. Если ввод отклонён по ограничению модели, ответ содержит ту же идентичность и поле `reason`, например `input_limit`, `reserved_token` или `outside_training_context`. Ошибки платформы и провайдера возвращаются как HTTP-ошибки. Запросы выбора используют общую очередь менеджера. Распределение оценок выдаёт модель, поэтому перед использованием результата требуется ревью.
