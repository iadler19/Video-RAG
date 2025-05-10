from concurrent.futures import ThreadPoolExecutor, as_completed

from loguru import logger
from typing_extensions import Annotated
from zenml import get_step_context, step

from llm_engineering.application import utils
from llm_engineering.domain.base.nosql import NoSQLBaseDocument
from llm_engineering.domain.documents import VideoClipDocument


@step
def query_data_warehouse() -> Annotated[list, "raw_documents"]:
    documents = []
    authors = []

    logger.info(f"Querying data warehouse video clip captions")

    results = fetch_all_data()
    # documents = [doc for query_result in results.values() for doc in query_result]
    documents.extend(results)

    step_context = get_step_context()
    step_context.add_output_metadata(
        output_name="raw_documents", metadata=_get_metadata(documents)
    )

    return documents


def fetch_all_data() -> dict[str, list[NoSQLBaseDocument]]:
    try:
        return __fetch_captions()
    except Exception:
        logger.exception(f"Fetching captions failed.")


def __fetch_captions() -> list[NoSQLBaseDocument]:
    return VideoClipDocument.bulk_find()


def _get_metadata(documents: list[VideoClipDocument]) -> dict:
    metadata = {
        "num_documents": len(documents),
    }
    for document in documents:
        collection = document.get_collection_name()
        if collection not in metadata:
            metadata[collection] = {}
        if "title" not in metadata[collection]:
            metadata[collection]["titles"] = list()

        if "title_counts" not in metadata[collection]:
            metadata[collection]["title_counts"] = {}

        title = document.title
        if title not in metadata[collection]["title_counts"]:
            metadata[collection]["title_counts"][title] = 0
        metadata[collection]["title_counts"][title] += 1

        metadata[collection]["num_documents"] = (
            metadata[collection].get("num_documents", 0) + 1
        )

    return metadata
