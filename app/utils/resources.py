import pathlib
import httpx

async def fetch_resource(resource: str) -> str:
    """
    Fetches content from resource.
    - http(s):// → загружает по сети
    - file:// → читает локальный файл
    Возвращает текст (str), готовый для нарезки на чанки.
    """
    if resource.startswith("http://") or resource.startswith("https://"):
        async with httpx.AsyncClient() as client:
            resp = await client.get(resource)
            resp.raise_for_status()
            return resp.text  # TODO: для PDF/HTML добавить парсинг
    elif resource.startswith("file://"):
        path = pathlib.Path(resource[7:])
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        return path.read_text(encoding="utf-8", errors="ignore")
    else:
        raise ValueError(f"Unsupported resource type: {resource}")
