from zenml import pipeline
from datasets import load_dataset

# from steps.etl import crawl_links, get_or_create_user
from steps.etl import clean_and_chunk, semantic_chunk


@pipeline
def digital_data_etl(streaming_link: str):
    dataset = load_dataset(
        "aegean-ai/ai-lectures-spring-24", streaming=True, token=True
    )
    invocation_ids = []

    for example in dataset["train"]:
        title = example.get("json").get("title")
        url = example["info.json"].get("webpage_url")
        captions = example.get("json").get("captions")

        chunks = clean_and_chunk(captions)

        last_step = semantic_chunk(chunks, title, url)

        invocation_ids.append(last_step.invocation_id)

    return invocation_ids
