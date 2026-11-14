# RAG для МТС EnergyTool

Исходный корпус находится в `../knowledge-base`. В нём 19 Markdown-документов с описаниями продуктов, справочными данными и ограничениями. `../demo_questions.csv` содержит 18 примеров, включая историю диалога внутри одного поля CSV.

```text
rag/
├── build_index.py       # Markdown → чанки → BGE-M3 → FAISS
├── rag_pipeline.py      # вопрос → поиск → локальная Qwen → ответ
├── rag_common.py        # модель, ревизия, формат и атомарная запись
├── requirements.txt
├── Dockerfile
└── artifacts/
    └── index.zip        # создаётся командой индексации
```

## Как работает поиск

1. `build_index.py` обходит Markdown, выделяет секции по заголовкам и дробит длинные секции преимущественно по границам абзацев и предложений. По умолчанию размер фрагмента до 1400 символов, перекрытие 180. Каждый фрагмент хранит путь, цепочку заголовков и номера строк. Заголовки внутри ограждённых блоков кода остаются частью текста.
2. `BAAI/bge-m3` через SentenceTransformers кодирует заголовок и текст каждого фрагмента в нормализованный 1024-мерный вектор. `FAISS IndexFlatIP` даёт точный поиск по косинусному сходству. В одном `index.zip` хранятся бинарный индекс и метаданные, включая ревизию модели и текст фрагментов. Артефакт создаётся атомарно.
3. При инференсе контейнер читает готовый индекс и кодирует только запрос. При наличии истории последняя реплика пользователя идёт первой, за ней следует контекст предыдущих пользовательских реплик. Qwen получает полный вопрос с историей и до шести фрагментов. Промпт требует ссылок на фрагменты, сохраняет условия и единицы измерения, просит уточнить недостающие параметры и предусматривает отказ. Код проверяет ID ссылок на уровне ответа. Содержательную опору каждого утверждения определяет модель.

Используются версии `faiss-cpu`, `numpy` и `httpx` из исходного `../requirements.txt`. Для локального BGE-M3 в образ добавлены SentenceTransformers, Transformers и CPU-версия PyTorch. Docker загружает веса во время сборки и использует их из локального кеша. Готовый образ содержит текст корпуса в артефакте индекса.

## Запуск в Docker

Команды ниже рассчитаны на PowerShell и выполняются из папки `rag`. Для первой сборки нужен доступ к PyPI, PyTorch и Hugging Face. Контейнер генерации скачивает Qwen при первом `ollama pull`.

### 1. Собрать индекс

```powershell
docker build --target builder -t energytool-indexer .
$project = (Resolve-Path ..).Path
New-Item -ItemType Directory -Force -Path artifacts | Out-Null
docker run --rm --mount "type=bind,src=$project/knowledge-base,dst=/docs,readonly" --mount "type=bind,src=${PWD}/artifacts,dst=/app/artifacts" energytool-indexer python build_index.py --docs /docs
```

### 2. Собрать образ инференса и запустить Qwen

```powershell
docker build --target runtime -t energytool-rag .
docker network create energytool
docker run -d --name energytool-qwen --network energytool -e OLLAMA_CONTEXT_LENGTH=8192 -v energytool-ollama:/root/.ollama ollama/ollama
docker exec energytool-qwen ollama pull qwen2.5:7b
```

Если Qwen уже доступна по OpenAI-совместимому API, укажите её адрес с префиксом API в `LLM_BASE_URL`, имя модели в `LLM_MODEL` и при необходимости передайте `LLM_API_KEY`. Например, для Ollama на хосте Docker Desktop подходит `http://host.docker.internal:11434/v1`. Для других тегов Qwen проверьте окно контекста и запас токенов для генерации.

### 3. Задать вопрос или прогнать демонабор

```powershell
docker run --rm --network energytool -e LLM_BASE_URL=http://energytool-qwen:11434/v1 -e LLM_MODEL=qwen2.5:7b energytool-rag --question "Какие ценовые категории доступны при 670 кВт?" --json

docker run --rm --network energytool -e LLM_BASE_URL=http://energytool-qwen:11434/v1 -e LLM_MODEL=qwen2.5:7b --mount "type=bind,src=$project,dst=/input,readonly" --mount "type=bind,src=${PWD}/artifacts,dst=/output" energytool-rag --questions /input/demo_questions.csv --output /output/submission.csv
```

В режиме `--json` результат содержит ответ, баллы косинусного сходства и найденные источники. Пакетный режим записывает `id,answer` в UTF-8, сохраняет порядок ID и убирает служебные маркеры ссылок. `--top-k`, `--min-score`, `--max-tokens` и `--timeout` задаются отдельно. По умолчанию поиск передаёт ближайшие фрагменты без калиброванного порога релевантности. После изменений базы повторите индексацию и сборку образа инференса.

Этот комплект предназначен для собственного Docker-запуска. Автоматическая проверка хакатона принимает ZIP с `solution.py`, отдельным перечнем разрешённых зависимостей и доступом к прокси платформы. Формат CSV поддерживается в пакетном режиме.

[BAAI/bge-m3](https://huggingface.co/BAAI/bge-m3) · [Ollama OpenAI API](https://docs.ollama.com/api/openai-compatibility) · [Qwen2.5 в Ollama](https://ollama.com/library/qwen2.5:7b)
