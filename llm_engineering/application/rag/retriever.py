import concurrent.futures


from loguru import logger
from qdrant_client.models import FieldCondition, Filter, MatchValue
from collections import defaultdict
from typing import List

from llm_engineering.application import utils
from llm_engineering.application.preprocessing.dispatchers import EmbeddingDispatcher
from llm_engineering.domain.embedded_chunks import EmbeddedChunk, EmbeddedVideoClipChunk
from llm_engineering.domain.queries import EmbeddedQuery, Query

# from .query_expanison import QueryExpansion
from .reranking import Reranker
# from .self_query import SelfQuery


class ContextRetriever:
    def __init__(self, mock: bool = False) -> None:
        self._reranker = Reranker(mock=mock)

    def search(
        self,
        query: str,
        k: int = 3,
    ) -> list:
        query_model = Query.from_str(query)

        logger.info(
            f"Query  = {query_model.content}",
        )

        n_k_documents = self._search(query_model, k)

        n_k_documents = list(set(n_k_documents))

        logger.info(f"{len(n_k_documents)} documents retrieved successfully")

        if len(n_k_documents) > 0:
            k_documents = self.rerank(query, chunks=n_k_documents, keep_top_k=k)
        else:
            k_documents = []

        return k_documents

    def _search(self, query: Query, k: int = 3) -> list[EmbeddedVideoClipChunk]:
        assert k >= 3, "k should be >= 3"

        embedded_query: EmbeddedQuery = EmbeddingDispatcher.dispatch(query)

        retrieved_chunks = EmbeddedVideoClipChunk.search(
            query_vector=embedded_query.embedding,
            limit=k // 3,
        )

        return retrieved_chunks

    def rerank(
        self, query: str | Query, chunks: list[EmbeddedVideoClipChunk], keep_top_k: int
    ) -> list[EmbeddedVideoClipChunk]:
        if isinstance(query, str):
            query = Query.from_str(query)

        reranked_documents = self._reranker.generate(
            query=query, chunks=chunks, keep_top_k=keep_top_k
        )

        logger.info(f"{len(reranked_documents)} documents reranked successfully.")

        return reranked_documents

    def group_by_title(
        self, documents: list[EmbeddedVideoClipChunk]
    ) -> list[EmbeddedVideoClipChunk]:
        grouped = defaultdict(list)
        # Group chunks by title
        for doc in documents:
            grouped[doc.title].append(doc)

        result = []
        for title, group in grouped.items():
            min_start = min(c.start_time for c in group)
            max_end = max(c.end_time for c in group)
            combined_content = " ".join(c.content for c in group)

            # Take representative data from the first chunk
            first_chunk = group[0]

            grouped_chunk = EmbeddedVideoClipChunk(
                title=title,
                url=first_chunk.url,
                start_time=min_start,
                end_time=max_end,
                content=combined_content,
                embedding=None,  # You could optionally average embeddings here
                metadata={"source_chunks": len(group)},
            )
            result.append(grouped_chunk)

        return result
