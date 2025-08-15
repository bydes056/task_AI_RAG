from langchain_core.prompts import ChatPromptTemplate
import os
from dotenv import load_dotenv


load_dotenv()
user_agent = os.getenv('USER_AGENT')
os.environ["USER_AGENT"] = user_agent


# Файлы
LOG_FILE = 'AI_task.log'
LOG_FILE_MODE = "a"
FILE_LINKS = 'links.txt'

# Кеш
CACHE_DIR = "cache"
MODEL_CACHE_FOLDER = "./cache/model_cache"
CACHE_DOCS = "cached_docs.pkl"
os.makedirs(CACHE_DIR, exist_ok=True)

# Разбивка доков
CHUNK_SIZE = 100000
CHUNK_OVERLAP = 2000

# Эмбеддинг
NAME_MODEL = "intfloat/multilingual-e5-large"
MODEL_KWARGS = {'device': 'cpu'}
ENCODE_KWARGS = {'normalize_embeddings': False}

# Gigachat модель
MODEL_GIGACHAT = 'GigaChat-2'
PROMPT_TEMPLATE = ChatPromptTemplate.from_template(
    '''Ответь на вопрос пользователя.
    Используй при этом только информацию из контекста.
    Если в контексте нет информации для ответа, сообщи об этом пользователю.
    Контекст: {context}
    Вопрос: {input}
    Ответ:''')

# BM25
NUM_DOC = 3
