# BGE-M3 на Intel Arc

[English version](bge-arc.md)

Этот профиль подключается явно. OpenVINO запускает закреплённый FP16-энкодер BAAI BGE-M3 и две обученные проекции на Intel GPU. Рабочий процесс запускается без сети, читает веса из каталога, смонтированного только для чтения, и принимает запросы через приватный Unix-сокет. Токенизация и агрегация лексических весов выполняются на CPU.

## Подготовка профиля

Используйте FP16-экспорт из [аппаратной проверки Project Creator](https://github.com/VolkovDani/project-creator/blob/94003dd680fbf72812f4145d1345fbb34bfa44b1/docs/intel-device-verification.md). Контрольные суммы OpenVINO IR и токенизатора должны совпасть с закреплёнными в скрипте подготовки. Для скрипта нужно локальное Python-окружение с NumPy и PyTorch. Обе библиотеки есть в использованном при проверке окружении экспорта Project Creator.

```sh
pc_dir=/absolute/path/to/project-creator
test -e models.json || cp models.example.json models.json
export MODELS_BGE_MODEL_DIR="$PWD/.bge/models/bge-m3-arc"
export MODELS_BGE_PROFILE=$(
  "$pc_dir/.npu/export-env/bin/python" -m runtime.bge.prepare \
    --encoder-dir "$pc_dir/.npu/models/bge-m3-fp16" \
    --output-dir "$MODELS_BGE_MODEL_DIR" --download-heads
)
```

Команда скачивает только две проекции из закреплённой ревизии BAAI. Для подготовки без сети используйте `--heads-dir /absolute/path/to/heads` с официальными файлами `sparse_linear.pt` и `colbert_linear.pt`. Выходной каталог должен быть новым. Напечатанный SHA-256 закрепляет веса, токенизатор, предобработку и лимиты токенов.

## Запуск

Добавьте запись в приватный `models.json` и подставьте напечатанный digest.

```json
{"name":"bge-arc","provider":"openvino-bge","base_url":"unix:///run/models-bge/runtime.sock","model":"BAAI/bge-m3","role":"embeddings","digest":"<profile SHA-256>"}
```

Предоставленный Compose-файл ожидает Intel Arc по пути `/dev/dri/renderD128`. Сверьте render node на хосте. Сохраните настройки профиля в локальном `.env`, чтобы они сохранились при перезапуске оболочки.

```sh
export MODELS_BGE_SOCKET_DIR="$PWD/.bge/run"
install -d -m 0750 "$MODELS_BGE_SOCKET_DIR"
export MODELS_BGE_UID=$(id -u)
export MODELS_BGE_SOCKET_GID=$(stat -c %g "$MODELS_BGE_SOCKET_DIR")
export MODELS_BGE_DEVICE_GID=$(stat -c %g /dev/dri/renderD128)
docker compose -f compose.yml -f compose.bge-arc.yml up --build -d
curl --fail --silent http://127.0.0.1:8317/v1/models
```

Если NPU уже подключён, добавьте `compose.bge-arc.yml` к действующему списку Compose-файлов. Если после аварийной остановки остался файл сокета, проверьте, что его рабочий процесс остановлен, прежде чем удалять этот файл.

## Использование профиля

Передавайте полный ID модели из списка. Действующий метод `POST /v1/embeddings` возвращает плотные векторы размерности 1024 и принимает до 32 текстов по 4096 токенов. Для эксперимента отправьте следующий запрос на `POST /v1/bge-m3/encode`. Этот метод принимает до четырёх текстов и требует явного списка режимов. В запросах со Sparse или ColBERT действует лимит 128 токенов на текст. Лимиты включают специальные токены.

```json
{"model":"<manager model ID>","input":["Отправить заявку."],"modes":["dense","sparse","colbert"]}
```

Ответ содержит ID модели в менеджере и одну запись на входной текст. Sparse возвращает положительные веса по ID словаря. ColBERT возвращает векторы отдельных токенов. Смена профиля эмбеддингов в Project Creator ставит требования в очередь переиндексации. Текущий поиск использует плотные векторы.
