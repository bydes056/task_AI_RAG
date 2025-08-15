from concurrent.futures import ThreadPoolExecutor, as_completed
from langchain_community.document_loaders import WebBaseLoader
import pickle
from typing import List, Optional
from module_project.log_project import logger
import string
import os
from module_project.settings import CACHE_DIR


def load_links(file_path: str) -> List[str]:
    with open(file_path) as f:
        return f.read().split()


def load_documents_parallel(urls: list, max_workers: int = 5) -> list:
    docs = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(WebBaseLoader(url).load): url
            for url in urls if url.startswith(("http://", "https://"))
        }
        for future in as_completed(futures):
            url = futures[future]
            try:
                doc = future.result()
                docs.extend(doc)
                logger.info(f"Успешно загружен: {url}")
            except Exception as e:
                logger.error(f"Ошибка при загрузке {url}: {e}")
    return docs


def save_cache(data, filename: str):
    with open(os.path.join(CACHE_DIR, filename), "wb") as f:
        pickle.dump(data, f)


def load_cache(filename: str) -> Optional[any]:
    cache_path = os.path.join(CACHE_DIR, filename)
    if os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            return pickle.load(f)
    return None


def tokenize(s: str) -> List[str]:
    return s.lower().translate(
        str.maketrans("", "", string.punctuation)
        ).split()
