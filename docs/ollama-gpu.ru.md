# GPU-профили Ollama

[English version](ollama-gpu.md)

Используйте Linux и Docker Compose 2.24.4 или новее. Выберите один дополнительный файл Compose для Ollama на установку. Все четыре профиля используют существующий сервис `ollama` и том `ollama_data`. Менеджер обращается к Ollama по адресу `http://ollama:11434` в приватной сети рантаймов. Клиентские приложения используют API менеджера.

| GPU | Дополнительный файл Compose | Профиль | Выбор устройства |
| --- | --- | --- | --- |
| Intel Arc | `compose.ollama-arc.yml` | `arc` | Intel render-устройство и числовой ID его группы |
| NVIDIA | `compose.ollama-nvidia.yml` | `nvidia` | Один UUID NVIDIA из `nvidia-smi -L`. Требуется NVIDIA Container Toolkit. |
| AMD ROCm | `compose.ollama-rocm.yml` | `rocm` | Один UUID GPU из `rocminfo`, соответствующее AMD render-устройство и числовые ID групп этого устройства и `/dev/kfd`. Нужен GPU, поддерживаемый Ollama через ROCm. Закреплённый образ предназначен для Linux amd64. |
| AMD Vulkan | `compose.ollama-amd-vulkan.yml` | `amd-vulkan` | AMD render-устройство и числовой ID его группы. Используется драйвер RADV Vulkan. |

## Запуск Ollama

Для профиля с render-устройством определите PCI-адрес GPU с помощью `lspci -Dnn` и `ls -l /dev/dri/by-path/*-render`. Найдите render-устройство по PCI-адресу выбранной видеокарты. Для ROCm проверьте, что UUID и PCI-адрес, которые сообщает ROCm, относятся к той же видеокарте.

```bash
pci=0000:00:02.0 # Подставьте PCI-адрес выбранной видеокарты.
export MODELS_OLLAMA_DRI_DEVICE=$(readlink -e "/dev/dri/by-path/pci-${pci}-render")
export MODELS_OLLAMA_RENDER_GID=$(stat -c %g "$MODELS_OLLAMA_DRI_DEVICE")
```

Для NVIDIA или ROCm укажите полный UUID `GPU-…` выбранного устройства в `MODELS_OLLAMA_GPU_UUID`. Для ROCm также задайте `MODELS_OLLAMA_KFD_GID` через `$(stat -c %g /dev/kfd)`. Профили Vulkan передают одно render-устройство с драйвером Vulkan соответствующего производителя и выбирают индекс Vulkan 0 внутри контейнера.

Укажите профиль и добавьте его файл после `compose.yml`. В примере используются базовый файл и дополнительный файл Ollama. Сохраните уже используемые дополнительные файлы настроек, NPU или BGE. Если менеджер запускается через CLI Project Creator, сохраните переменные Compose и устройства в приватном `.env` менеджера.

```bash
export COMPOSE_PROFILES=arc # Или nvidia, rocm, amd-vulkan.
export COMPOSE_FILE="compose.yml:compose.ollama-${COMPOSE_PROFILES}.yml"
docker compose config --quiet
docker compose up -d --build ollama
```

По инструкции [Добавление моделей](../README.ru.md#добавление-моделей) закрепите для каждой активной роли точный тег Ollama и полный SHA-256 манифеста. Можно использовать уже установленные в томе менеджера веса, скачать нужную модель или восстановить локальную копию весов из прежней установки в собственный том менеджера. Сохраните прежний том для отката. После загрузки или восстановления и перед настройкой клиентов сверьте дайджест каждого манифеста. Менеджер проверяет дайджест модели Ollama до и после инференса.

## Проверка оборудования и размещения в памяти

Перезапустите Ollama с выбранным профилем. Отправьте через API менеджера по одному синтетическому запросу на каждую модель и роль. Для Project Creator проверьте структурированный ответ и эмбеддинги из 1024 компонентов, если эти роли используются. Для активной роли выбора варианта выполните её проверочный запрос. На старом и новом endpoint используйте одинаковые дайджесты моделей, входные данные и параметры запроса. Сразу после каждого запроса проверьте размещение, пока другая модель не вытеснила текущую. [Примеры API](../README.ru.md#подключение-приложений) · [Проверка выбора варианта](decision-models.ru.md#запрос-выбора)

Задайте точный тег Ollama в `model` и полный дайджест из настроек в `digest`. Команда читает приватный API Ollama из временного контейнера в том же сетевом пространстве. Выполняйте её в Bash с установленным `jq`.

```bash
set -o pipefail
ollama_container=$(docker compose ps -q ollama)
docker run --rm --network "container:$ollama_container" \
  alpine:3.22@sha256:49340b346798875b5592e5d420241f2f9cfba5f47e62ebb1446a1b34734e898f \
  wget -qO- http://127.0.0.1:11434/api/ps |
  jq -e --arg model "$model" --arg digest "${digest#sha256:}" '
    .models[] | select(.name == $model and (.digest | ltrimstr("sha256:")) == $digest
      and (.remote_host // "") == "" and (.remote_model // "") == "")
    | select((.size | type) == "number" and (.size_vram | type) == "number"
      and .size_vram > 0 and .size_vram <= .size)
    | {name, digest, size, size_vram}'
docker compose logs --no-color ollama
```

Запишите результат запроса, время, образ и версию Ollama, дайджест модели, имя и PCI-адрес или UUID GPU и оба размера в байтах. Сверьте бэкенд и устройство из строки `inference compute` в журнале Ollama с выбранной видеокартой и проверьте записи о загрузке модели и переносе слоёв. При `size_vram == size` Ollama сообщает о полном размещении на GPU, при меньшем положительном значении — о размещении одновременно на CPU и GPU. На встроенной графике память GPU выделяется из общей оперативной памяти. Выполнение только на CPU, отсутствие или несовпадение дайджеста, другое устройство или программный рендерер Vulkan означают, что проверка GPU не пройдена.

## Результаты проверки

27 ноября 2026 года через Ollama из Project Creator и API менеджера выполнили синтетические запросы структурированного чата и эмбеддингов на встроенной Intel Arc в Core Ultra 7 268V (`8086:64a0`, PCI `0000:00:02.0`, `/dev/dri/renderD128`). В обоих случаях использовали Ollama 0.40.16 и Vulkan. В прежнем образе использовалась Mesa `25.2.8-0ubuntu0.24.04.4`, в образе менеджера — `25.2.8-0ubuntu0.24.04.5`.

| Модель | Дайджест Ollama | `size_vram` и `size` на обоих endpoint |
| --- | --- | ---: |
| `qwen2.5:0.5b` | `a8b0c51577010a279d933d14c2a8ab4b268079d44c5c8830c0a93900f1827c67` | По 456 361 245 байт |
| `bge-m3:567m` | `7907646426070047a77226ac3e684fbbe8410524f7b4a74d02837e43f2146bab` | По 689 187 061 байту |

JSON-ответы и все 1024 компонента эмбеддингов совпали. Оба рантайма сообщили о переносе всех 25 слоёв каждой модели на GPU. В этом прогоне проверяли endpoint моделей на синтетических данных. Повторите проверку для моделей и ролей вашей установки. Для NVIDIA, AMD ROCm и AMD Vulkan ещё нужны реальные запросы и проверка размещения на соответствующем оборудовании. [Задача переноса](https://github.com/VolkovDani/models-configurator/issues/3)

## Переключение Project Creator после проверки

Когда профиль пройдёт проверку на целевой машине, подключите каждую проверенную роль к менеджеру по полному ID из `/v1/models`. Сохраните роли, которые ещё используют встроенный рантайм, и оставьте прежние настройки устройства и веса для отката. Переключайте установку командой `--device remote --switch-device`, только когда все активные роли работают через API менеджера. После смены ID профиля модели может потребоваться заново построить эмбеддинги для этого профиля. Следуйте инструкциям Project Creator по подключению и откату. [Руководство по установке Project Creator](https://github.com/VolkovDani/project-creator/blob/develop/docs/setup-cli.md) · [Профили менеджера](https://github.com/VolkovDani/project-creator/blob/develop/docs/adr/012-standalone-model-manager.ru.md#подключение-установки-compose)

Для возврата менеджера на CPU уберите дополнительный файл GPU, смените профиль Ollama на `cpu` и выполните `docker compose up -d --force-recreate ollama` в том же проекте Compose. Сохраните существующий том.

[Поддержка GPU в Ollama](https://docs.ollama.com/gpu) · [Резервирование GPU в Docker](https://docs.docker.com/compose/how-tos/gpu-support/) · [Доступ к устройствам AMD ROCm](https://rocm.docs.amd.com/en/latest/install/docker-containers.html)
