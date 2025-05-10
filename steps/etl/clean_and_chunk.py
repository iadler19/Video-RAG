from loguru import logger
from zenml import get_step_context, step

import difflib
import spacy


@step
def clean_and_chunk(captions: list[dict], token_limit=100):
    chunks = []
    current_chunk_tokens = []
    current_chunk_texts = []
    current_chunk_start = None
    current_chunk_end = None
    chunk_token_limit = token_limit
    nlp = spacy.load("en_core_web_sm")

    if captions:
        logger.info(f"Starting to chunk {len(captions)} captions.")
        total = ""
        for caption in captions:
            # Extract the start time, end time, and text
            start_time = caption.get("start", "N/A")
            end_time = caption.get("end", "N/A")
            text = caption.get("text", "").strip()

            # print(f"Start: {start_time} | End: {end_time}\n")

            # This combines the subtitles
            matcher = difflib.SequenceMatcher(None, total, text)
            match = matcher.find_longest_match(0, len(total), 0, len(text))

            if match.size > 0:
                new_addition = text[match.b + match.size :]
                # print(f"New addition: '{new_addition}'")
                total = total + new_addition

            else:
                new_addition = text
                # print(f"Caption: {text}")
                total = total + new_addition

            # building chunks of 100ish tokens
            doc = nlp(new_addition)
            # extracts non-whitespace
            tokens = [token.text for token in doc if not token.is_space]

            if not current_chunk_tokens:
                current_chunk_start = start_time

            # Add to current chunk
            current_chunk_tokens.extend(tokens)
            current_chunk_texts.append(new_addition)
            current_chunk_end = end_time

            if len(current_chunk_tokens) >= chunk_token_limit:
                chunk_text = " ".join(current_chunk_texts)
                chunks.append(
                    {
                        "text": chunk_text,
                        "start": current_chunk_start,
                        "end": current_chunk_end,
                        "token_count": len(current_chunk_tokens),
                    }
                )
                # print(chunks[-1])
                # Reset chunk buffers
                current_chunk_tokens = []
                current_chunk_texts = []
                current_chunk_start = None
                current_chunk_end = None

        # Final chunk if there's leftover( after loop)
        if current_chunk_tokens:
            if chunks:
                # Merge leftovers with last chunk
                chunks[-1]["text"] += " " + " ".join(current_chunk_texts)
                chunks[-1]["end"] = current_chunk_end
                chunks[-1]["token_count"] += len(current_chunk_tokens)
        else:
            # If no chunks exist yet, just make the first one
            chunk_text = " ".join(current_chunk_texts)
            chunks.append(
                {
                    "text": chunk_text,
                    "start": current_chunk_start,
                    "end": current_chunk_end,
                    "token_count": len(current_chunk_tokens),
                }
            )
        print(chunks[-1])

        logger.info(f"Successfully created {len(chunks)} chunks")
        return chunks

    else:
        logger.error(f"There are no captions for this clip")


def _add_to_metadata(metadata: dict, chunks: list[dict]) -> dict:
    # Initialize chunk summary
    if chunks:
        metadata["start_time"] = chunks[0]["start"]  # Start time of the first chunk
        metadata["end_time"] = chunks[-1]["end"]  # End time of the last chunk
        metadata["total_chunks"] = len(chunks)  # Total number of chunks

    return metadata
