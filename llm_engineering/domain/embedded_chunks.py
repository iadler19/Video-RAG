from abc import ABC

from pydantic import UUID4, Field

from llm_engineering.domain.types import DataCategory

from .base import VectorBaseDocument


class EmbeddedChunk(VectorBaseDocument, ABC):
    content: str
    embedding: list[float] | None
    platform: str
    document_id: UUID4
    author_id: UUID4
    author_full_name: str
    metadata: dict = Field(default_factory=dict)

    @classmethod
    def to_context(cls, chunks: list["EmbeddedChunk"]) -> str:
        context = ""
        for i, chunk in enumerate(chunks):
            context += f"""
            Chunk {i + 1}:
            Type: {chunk.__class__.__name__}
            Platform: {chunk.platform}
            Author: {chunk.author_full_name}
            Content: {chunk.content}\n
            """

        return context


class EmbeddedVideoClipChunk(VectorBaseDocument):
    title: str
    url: str
    start_time: float
    end_time: float
    content: str
    embedding: list[float] | None
    metadata: dict = Field(default_factory=dict)

    class Config:
        name = "embedded_clips"
        category = DataCategory.CLIPS
        use_vector_index = True

    @classmethod
    def to_context(cls, chunks: list["EmbeddedVideoClipChunk"]) -> str:
        context = ""
        for i, chunk in enumerate(chunks):
            context += f"""
            Clip {i + 1}:
            Title: {chunk.title}
            URL: {chunk.url}
            Time: {chunk.start_time:.2f}s - {chunk.end_time:.2f}s
            Content: {chunk.content.strip()}\n
            """
        return context.strip()
