from typing_extensions import Annotated
from zenml import get_step_context, step

from llm_engineering.application import utils
from llm_engineering.application.preprocessing import EmbeddingDispatcher

from llm_engineering.domain.embedded_chunks import EmbeddedChunk
from llm_engineering.domain.cleaned_documents import CleanedVideoClipDocument


@step
def chunk_and_embed(
    cleaned_documents: Annotated[list, "cleaned_documents"],
) -> Annotated[list, "embedded_documents"]:
    metadata = {
        "chunking": {},
        "embedding": {},
        "num_documents": len(cleaned_documents),
    }

    embedded_chunks = []

    for batched_chunks in utils.misc.batch(cleaned_documents, 10):
        batched_embedded_chunks = EmbeddingDispatcher.dispatch(batched_chunks)
        embedded_chunks.extend(batched_embedded_chunks)

    metadata["embedding"] = _add_embeddings_metadata(
        embedded_chunks, metadata["embedding"]
    )
    metadata["num_chunks"] = len(embedded_chunks)
    metadata["num_embedded_chunks"] = len(embedded_chunks)

    step_context = get_step_context()
    step_context.add_output_metadata(
        output_name="embedded_documents", metadata=metadata
    )

    return embedded_chunks


def _add_metadata(documents: list[CleanedVideoClipDocument]) -> dict:
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


def _add_embeddings_metadata(
    embedded_chunks: list[EmbeddedChunk], metadata: dict
) -> dict:
    for embedded_chunk in embedded_chunks:
        category = embedded_chunk.get_category()
        if category not in metadata:
            metadata[category] = embedded_chunk.metadata
        if "title" not in metadata[category]:
            metadata[category]["title"] = list()

        metadata[category]["title"].append(embedded_chunk.title)

    for value in metadata.values():
        if isinstance(value, dict) and "title" in value:
            value["title"] = list(set(value["title"]))

    return metadata
