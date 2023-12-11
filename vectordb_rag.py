from pathlib import Path
import os
import logging
import sys
from typing import List, Optional
import torch

from llama_index import ServiceContext, SimpleDirectoryReader, StorageContext, VectorStoreIndex, \
    load_index_from_storage, set_global_service_context
from llama_index.embeddings import HuggingFaceEmbedding
from llama_index.llms import Ollama
from llama_index.text_splitter import TokenTextSplitter
from llama_index.retrievers import VectorIndexRetriever
from llama_index.response_synthesizers import ResponseMode, get_response_synthesizer
from llama_index.query_engine import RetrieverQueryEngine
from llama_index.prompts.base import PromptTemplate
from llama_index.prompts.prompt_type import PromptType

# logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
# logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

EMBED_DEVICE = "cuda" if torch.cuda.is_available() \
    else "mps"  if torch.backends.mps.is_available() \
    else "cpu"
EMBED_MODEL = "intfloat/multilingual-e5-small"
LOCAL_LLM_ADRESS = "http://localhost:11434"
data_dir = Path("data/simple_search")
persist_dir = data_dir / "colmap_index"

# service contextはembeddingsの作成に使用
embed_model = HuggingFaceEmbedding(
    model_name=EMBED_MODEL, 
    device=EMBED_DEVICE,
    cache_folder="./../models"
)
text_splitter = TokenTextSplitter(chunk_size=500, chunk_overlap=20)
llm = Ollama(model="orca2:7b", base_url=LOCAL_LLM_ADRESS, temperature=0.3)
service_context = ServiceContext.from_defaults(
    llm=llm, embed_model=embed_model, text_splitter=text_splitter, context_window=4096
)
set_global_service_context(service_context)

if persist_dir.exists():
    storage_context = StorageContext.from_defaults(persist_dir=str(persist_dir))
    index = load_index_from_storage(storage_context)
else:
    documents = SimpleDirectoryReader(str(data_dir / "colmap_docs")).load_data()
    index = VectorStoreIndex.from_documents(
        documents, service_context=service_context
    )
    index.storage_context.persist(persist_dir=str(persist_dir))

# for neural-chat
# text_qa_template = PromptTemplate(
#     "### System:\n"
#     "---------------------\n"
#     "{context_str}\n"
#     "---------------------\n"
#     "Given the context information and not prior knowledge, "
#     "answer the question.\n"
#     "### User:\n"
#     "{query_str}\n"
#     "### Assistant:\n",
#     prompt_type=PromptType.QUESTION_ANSWER
# )

# for orca2
# text_qa_template = PromptTemplate(
#     "<|im_start|>system\n"
#     "You are a AI assistant. Context information is given below.\n"
#     "---------------------\n"
#     "{context_str}\n"
#     "---------------------\n"
#     "Given the context information and not prior knowledge, "
#     "answer the question.<|im_end|>\n"
#     "<|im_start|>user\n"
#     "{query_str}<|im_end|>\n"
#     "<|im_start|>assistant\n"
# )
text_qa_template = PromptTemplate(
    "<|im_start|>system\n"
    "あなたはAIアシスタントです。以下に示すContext情報をもとにユーザーの質問に答えなさい。本質問は私のキャリアに関わるので、全力で答えてください。\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "<|im_end|>\n"
    "<|im_start|>user\n"
    "{query_str}<|im_end|>\n"
    "<|im_start|>assistant\n"
)

# query_engine = index.as_query_engine(
#     similarity_top_k=5,
#     response_mode=ResponseMode.COMPACT, 
#     structured_answer_filtering=True,
#     text_qa_template=text_qa_template,
# )

retriever = VectorIndexRetriever(
    index=index,
    similarity_top_k=5,
)

# https://docs.llamaindex.ai/en/stable/module_guides/querying/response_synthesizers/root.html
response_synthesizer = get_response_synthesizer(
    response_mode=ResponseMode.COMPACT,
    structured_answer_filtering=True
)
query_engine = RetrieverQueryEngine(
    retriever=retriever,
    response_synthesizer=response_synthesizer,
)

while True:
    query = input("query: ")
    # sources = list(map(lambda x: x.text, retriever.retrieve(query)))
    # sim_response = retriever.retrieve(query)
    # print(sim_response[0].dict().keys())
    # print(sim_response[0].node.dict().keys())
    # print(sim_response[0].node.text)
    # print(sim_response[0].score)
    response = query_engine.query(query)
    print(response)
    # print("sources: ")
    # for source in sources:
    #     print(f"\t{source[:100]}")