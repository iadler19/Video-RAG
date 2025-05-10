from abc import ABC
from typing import Optional

from pydantic import UUID4

from .base import VectorBaseDocument
from .types import DataCategory


class CleanedVideoClipDocument(VectorBaseDocument):
    title: str
    url: str
    start_time: float
    end_time: float
    content: str

    class Config:
        name = "cleaned_clips"
        category = DataCategory.CLIPS
        use_vector_index = False
