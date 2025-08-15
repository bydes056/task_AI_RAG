import logging
from module_project.settings import LOG_FILE, LOG_FILE_MODE


logging.basicConfig(level=logging.INFO,
                    filename=LOG_FILE,
                    filemode=LOG_FILE_MODE,
                    format="%(asctime)s %(levelname)s %(message)s")
logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger()

