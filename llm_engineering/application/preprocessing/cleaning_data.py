from abc import ABC, abstractmethod
from typing import Generic, TypeVar

from llm_engineering.domain.cleaned_documents import CleanedVideoClipDocument
from llm_engineering.domain.documents import VideoClipDocument

from .operations import clean_text


class VideoClipCleaningHandler:
    def clean(self, data_model: VideoClipDocument) -> CleanedVideoClipDocument:
        return CleanedVideoClipDocument(
            id=data_model.id,
            title=data_model.title,
            url=data_model.url,
            start_time=data_model.start_time,
            end_time=data_model.end_time,
            content=clean_text(data_model.content),
        )
