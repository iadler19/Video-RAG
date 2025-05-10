from abc import ABC, abstractmethod
from typing import Generic, TypeVar, cast

from llm_engineering.application.networks import EmbeddingModelSingleton
from llm_engineering.domain.embedded_chunks import EmbeddedChunk, EmbeddedVideoClipChunk
from llm_engineering.domain.cleaned_documents import CleanedVideoClipDocument
from llm_engineering.domain.queries import EmbeddedQuery, Query

ChunkT = TypeVar("ChunkT")
EmbeddedChunkT = TypeVar("EmbeddedChunkT")

embedding_model = EmbeddingModelSingleton()


class EmbeddingDataHandler(ABC, Generic[ChunkT, EmbeddedChunkT]):
    """
    Abstract class for all embedding data handlers.
    All data transformations logic for the embedding step is done here
    """

    def embed(self, data_model: ChunkT) -> EmbeddedChunkT:
        return self.embed_batch([data_model])[0]

    def embed_batch(self, data_model: list[ChunkT]) -> list[EmbeddedChunkT]:
        embedding_model_input = [data_model.content for data_model in data_model]
        embeddings = embedding_model(embedding_model_input, to_list=True)

        embedded_chunk = [
            self.map_model(data_model, cast(list[float], embedding))
            for data_model, embedding in zip(data_model, embeddings, strict=False)
        ]

        return embedded_chunk

    @abstractmethod
    def map_model(self, data_model: ChunkT, embedding: list[float]) -> EmbeddedChunkT:
        pass


class QueryEmbeddingHandler(EmbeddingDataHandler):
    def map_model(self, data_model: Query, embedding: list[float]) -> EmbeddedQuery:
        return EmbeddedQuery(
            id=data_model.id,
            content=data_model.content,
            embedding=embedding,
            metadata={
                "embedding_model_id": embedding_model.model_id,
                "embedding_size": embedding_model.embedding_size,
                "max_input_length": embedding_model.max_input_length,
            },
        )


class VideoClipEmbeddingHandler(EmbeddingDataHandler):
    def map_model(
        self, data_model: CleanedVideoClipDocument, embedding: list[float]
    ) -> EmbeddedVideoClipChunk:
        return EmbeddedVideoClipChunk(
            id=data_model.id,
            title=data_model.title,
            url=data_model.url,
            start_time=data_model.start_time,
            end_time=data_model.end_time,
            content=data_model.content,
            embedding=embedding,
            metadata={
                "embedding_model_id": embedding_model.model_id,
                "embedding_size": embedding_model.embedding_size,
                "max_input_length": embedding_model.max_input_length,
            },
        )


# ---------Delete this -----------


# class ArticleEmbeddingHandler(EmbeddingDataHandler):
#     def map_model(self, data_model: ArticleChunk, embedding: list[float]) -> EmbeddedArticleChunk:
#         return EmbeddedArticleChunk(
#             id=data_model.id,
#             content=data_model.content,
#             embedding=embedding,
#             platform=data_model.platform,
#             link=data_model.link,
#             document_id=data_model.document_id,
#             author_id=data_model.author_id,
#             author_full_name=data_model.author_full_name,
#             metadata={
#                 "embedding_model_id": embedding_model.model_id,
#                 "embedding_size": embedding_model.embedding_size,
#                 "max_input_length": embedding_model.max_input_length,
#             },
#         )
