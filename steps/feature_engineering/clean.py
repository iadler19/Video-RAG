from typing_extensions import Annotated
from zenml import get_step_context, step

from llm_engineering.application.preprocessing.cleaning_data import (
    VideoClipCleaningHandler,
)
from llm_engineering.domain.cleaned_documents import CleanedVideoClipDocument


@step
def clean_documents(
    documents: Annotated[list, "raw_documents"],
) -> Annotated[list, "cleaned_documents"]:
    cleaned_documents = []
    for document in documents:
        # this is what I'm changing
        # cleaned_document = CleaningDispatcher.dispatch(document)
        handler = VideoClipCleaningHandler()
        cleaned_document = handler.clean(data_model=document)

        cleaned_documents.append(cleaned_document)

    step_context = get_step_context()
    step_context.add_output_metadata(
        output_name="cleaned_documents", metadata=_get_metadata(cleaned_documents)
    )

    return cleaned_documents


def _get_metadata(documents: list[CleanedVideoClipDocument]) -> dict:
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
