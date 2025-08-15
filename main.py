from typing import Any, List
import module_project.settings as settings
import module_project.utils as utils
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_gigachat import GigaChat
from langchain.chains import create_retrieval_chain
from langchain.retrievers import EnsembleRetriever
from langchain_community.vectorstores import FAISS
from langchain.retrievers.document_compressors import DocumentCompressorPipeline, EmbeddingsFilter
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from module_project.log_project import logger



def main() -> Any:
    try:
        links = utils.load_links(settings.FILE_LINKS)
        logger.info(f"Загружено {len(links)} ссылок.")

        cache_filename_docs = settings.CACHE_DOCS
        docs = utils.load_cache(cache_filename_docs)

        if docs is None:
            docs = utils.load_documents_parallel(links)
            utils.save_cache(docs, cache_filename_docs)
        else:
            logger.info("Документы загружены из кеша.")

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP
        )
        split_docs = text_splitter.split_documents(docs)
        embedding = HuggingFaceEmbeddings(
            model_name=settings.NAME_MODEL,
            model_kwargs=settings.MODEL_KWARGS,
            encode_kwargs=settings.ENCODE_KWARGS,
            cache_folder=settings.MODEL_CACHE_FOLDER
        )
        llm = GigaChat(
            credentials=settings.os.getenv('API_KEY'),
            model=settings.MODEL_GIGACHAT,
            verify_ssl_certs=False,
            profanity_check=False
        )
        bm25_retriever = BM25Retriever.from_documents(
            documents=split_docs,
            preprocess_func=utils.tokenize,
            k=settings.NUM_DOC
        )
        vectorstore = FAISS.from_documents(split_docs, embedding)
        vector_retriever = vectorstore.as_retriever(search_kwargs={"k": settings.NUM_DOC})

        ensemble_retriever = EnsembleRetriever(
            retrievers=[bm25_retriever, vector_retriever],
            weights=[0.5, 0.5]
        )
        relevant_filter = EmbeddingsFilter(embeddings=embedding, similarity_threshold=0.7)
        pipeline_compressor = DocumentCompressorPipeline(
            transformers=[relevant_filter]
        )
        compression_retriever = ContextualCompressionRetriever(
            base_compressor=pipeline_compressor,
            base_retriever=ensemble_retriever
        )

        document_chain = create_stuff_documents_chain(
            llm=llm,
            prompt=settings.PROMPT_TEMPLATE
        )
        retrieval_chain = create_retrieval_chain(compression_retriever, document_chain)
        while True:
            query = input("\nЗадайте ваш вопрос (или 'exit' для выхода): ").strip()
            if query.lower() == "exit":
                break
            if not query:
                print("Пустой запрос. Попробуйте ещё раз.")
                continue

            resp = retrieval_chain.invoke({'input': query})
            answer = resp['answer']
            sources = resp.get('context', [])

            if sources:
                answer += "\n\nИсточники:"
                for i, doc in enumerate(sources, start=1):
                    answer += f"\n[{i}] {doc.metadata.get('source', 'Неизвестный источник')}"

            print(answer)
            logger.info(
                f"Запрос: {query}\n"
                f"Ответ: {answer}\n"
                f"Источники: {[doc.metadata.get('source', 'Неизвестный источник') for doc in sources]}"
                )

    except Exception as e:
        logger.error(f"Критическая ошибка: {e}")
        raise

if __name__ == "__main__":
    main()
