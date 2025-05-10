from loguru import logger
from zenml import get_step_context, step
from llm_engineering.domain.documents import VideoClipDocument


from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
import spacy

nlp = spacy.load("en_core_web_sm")
filler_words = [
    "uh",
    "um",
    "like",
    "you",
    "know",
    "so",
    "actually",
    "basically",
    "definitely",
    "sorry",
]


def clean_text(text):
    doc = nlp(text)
    # Remove stopwords, punctuation, and non-alphabetic tokens
    return " ".join(
        [
            token.text
            for token in doc
            if not token.is_stop
            and not token.is_punct
            and token.text.lower() not in filler_words
        ]
    )


def timestamp_to_seconds(timestamp_str):
    h, m, s = timestamp_str.split(":")
    s, ms = s.split(".")
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000


def creating_sentences(total, spans):
    doc = nlp(total)
    sentence_chunks = []
    for sent in doc.sents:
        sentence_text = sent.text
        sentence_start = sent.start_char
        sentence_end = sent.end_char

        # Find all spans that overlap with this sentence
        overlapping_spans = [
            span
            for span in spans
            if not (sentence_end <= span["start"] or sentence_start >= span["end"])
        ]

        if overlapping_spans:
            start_time = overlapping_spans[0]["start_time"]
            end_time = overlapping_spans[-1]["end_time"]

            sentence_chunks.append(
                {"text": sentence_text, "start_time": start_time, "end_time": end_time}
            )


# @step
# def semantic_chunk(total, spans, title: str, url: str):
#     # Step 1: Extract just the text from your chunks
#     cleaned_texts = [clean_text(chunk["text"]) for chunk in chunks if chunk["text"].strip() != ""]

#     embedder = SentenceTransformer("all-MiniLM-L6-v2")
#     embeddings = embedder.encode(cleaned_texts, show_progress_bar=True)

#     # Step 2: Create and fit the BERTopic model
#     topic_model = BERTopic(min_topic_size = 2)#, embedding_model = embedder)
#     topics, probs = topic_model.fit_transform(cleaned_texts, embeddings)

#     model = VideoClipDocument

#     grouped_chunks = []
#     current_group = {
#         "text": "",
#         "start": chunks[0]["start"],
#         "end": chunks[0]["end"],
#         "topic": topics[0],
#         "topic_name": topic_model.get_topic(topics[0])[0][0] if topics[0] != -1 else "Outlier"
#     }
#     logger.info(f"Starting to  semantically chunk {len(chunks)} chunks")

#     for i in range(len(chunks)):
#         chunk = chunks[i]
#         topic = topics[i]

#         if topic == current_group["topic"]:
#             # Extend current group
#             current_group["text"] += " " + chunk["text"]
#             current_group["end"] = chunk["end"]
#         else:
#             # Save current group and start a new one
#             instance = model(
#                 title= title,
#                 url =  url,
#                 start_time= timestamp_to_seconds(current_group['start']),
#                 end_time = timestamp_to_seconds(chunk["end"]),
#                 content = current_group['text']
#             )
#             grouped_chunks.append(instance)
#             instance.save()
#             #start new group
#             current_group = {
#                 "text": chunk["text"],
#                 "start": chunk["start"],
#                 "end": chunk["end"],
#                 "topic": topic,
#                 "topic_name": topic_model.get_topic(topic)[0][0] if topic != -1 else "Outlier"
#             }
#     step_context = get_step_context()
#     metadata = {}
#     metadata = _add_to_metadata(metadata, grouped_chunks)
#     step_context.add_output_metadata(metadata = metadata)
#     logger.info(f"Successfully semantically chunked {len(grouped_chunks)} chunks")
#     return title


@step
def semantic_chunk(chunks, title: str, url: str):
    # Step 1: Extract just the text from your chunks
    # cleaned_texts = [clean_text(chunk["text"]) for chunk in chunks if chunk["text"].strip() != ""]

    # embedder = SentenceTransformer("all-MiniLM-L6-v2")
    # embeddings = embedder.encode(cleaned_texts, show_progress_bar=True)

    # # Step 2: Create and fit the BERTopic model
    # topic_model = BERTopic(min_topic_size = 2)#, embedding_model = embedder)
    # topics, probs = topic_model.fit_transform(cleaned_texts, embeddings)

    model = VideoClipDocument

    clips_list = []

    for chunk in chunks:
        if chunk["start"] == None:
            print(chunk)
        instance = model(
            title=title,
            url=url,
            start_time=timestamp_to_seconds(chunk["start"]),
            end_time=timestamp_to_seconds(chunk["end"]),
            content=chunk["text"],
        )
        instance.save()
        clips_list.append(instance)

    step_context = get_step_context()
    metadata = {}
    metadata = _add_to_metadata(metadata, chunks, title, url)
    step_context.add_output_metadata(metadata=metadata)
    logger.info(f"Successfully semantically chunked {len(chunks)} chunks")
    return title


def _add_to_metadata(metadata: dict, grouped_chunks: list[dict], title, url) -> dict:
    if grouped_chunks:
        metadata["title"] = title
        metadata["start"] = grouped_chunks[0]["start"]  # Start time of the first chunk
        metadata["url"] = url
        metadata["end"] = grouped_chunks[-1]["end"]  # End time of the last chunk
        metadata["total_chunks"] = len(grouped_chunks)
    return metadata  # Total number of chunks
