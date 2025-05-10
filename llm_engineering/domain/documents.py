from abc import ABC
from typing import Optional

from pydantic import UUID4, Field

from .base import NoSQLBaseDocument
from .types import DataCategory


class VideoClipDocument(NoSQLBaseDocument):
    title: str
    url: str
    start_time: float
    end_time: float
    content: str

    class Settings:
        name = DataCategory.CLIPS


# --- I can delete these later ---
