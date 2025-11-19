from haystack import GeneratedAnswer, Pipeline
from haystack.components.builders import ChatPromptBuilder
from haystack.components.builders.answer_builder import AnswerBuilder
from haystack.components.builders.prompt_builder import PromptBuilder
from haystack.components.embedders import SentenceTransformersTextEmbedder
from haystack.components.generators import HuggingFaceAPIGenerator
from haystack.components.generators.chat import HuggingFaceAPIChatGenerator
from haystack.dataclasses import ChatMessage
from haystack.utils import Secret
from haystack.utils.hf import HFGenerationAPIType

from couchbase_haystack import CouchbasePasswordAuthenticator, CouchbaseSearchDocumentStore, CouchbaseSearchEmbeddingRetriever

# Load HF Token from environment variables.
HF_TOKEN = Secret.from_env_var("HF_API_TOKEN")

# Make sure you have a running couchbase database, e.g. with Docker:
# docker run \
#     --restart always \
#     --publish=8091-8096:8091-8096 --publish=11210:11210 \
#     --env COUCHBASE_ADMINISTRATOR_USERNAME=admin \
#     --env COUCHBASE_ADMINISTRATOR_PASSWORD=passw0rd \
#     couchbase:enterprise-7.6.2

document_store = CouchbaseSearchDocumentStore(
    cluster_connection_string=Secret.from_env_var("CB_CONNECTION_STRING"),
    authenticator=CouchbasePasswordAuthenticator(
        username=Secret.from_env_var("CB_USERNAME"),
        password=Secret.from_env_var("CB_PASSWORD"),
    ),
    bucket="haystack_bucket_name",
    scope="_default",
    collection="_default",
    vector_search_index="vector_search_index",
)

# Build a RAG pipeline with a Retriever to get relevant documents to the query and a HuggingFaceTGIGenerator
# interacting with LLMs using a custom prompt.
prompt_template = [
    ChatMessage.from_system(
        '''
        You are a precise, factual QA assistant.
        According to the following documents:
        {% for document in documents %}
        {{document.content}}
        {% endfor %}

        If an answer cannot be deduced from the documents, say "I don't know based on these documents".

        When answering:
        - be concise
        - write the documents that support your answer

        Answer the given question
        '''
    ),
    ChatMessage.from_user(
        '''
        {{query}}
        '''
    ),
    ChatMessage.from_system("Answer:")
]
rag_pipeline = Pipeline()
rag_pipeline.add_component(
    "query_embedder",
    SentenceTransformersTextEmbedder(model="sentence-transformers/all-MiniLM-L6-v2", progress_bar=False),
)
rag_pipeline.add_component("retriever", CouchbaseSearchEmbeddingRetriever(document_store=document_store))
rag_pipeline.add_component("chat_prompt_builder", ChatPromptBuilder(template=prompt_template, required_variables="*"))
rag_pipeline.add_component(
    "llm",
    HuggingFaceAPIChatGenerator(api_type=HFGenerationAPIType.SERVERLESS_INFERENCE_API,
                                api_params={"model": "Qwen/Qwen2.5-7B-Instruct",
                                            "provider": "together"}),
)
rag_pipeline.add_component("answer_builder", AnswerBuilder())

rag_pipeline.connect("query_embedder", "retriever.query_embedding")
rag_pipeline.connect("retriever.documents", "chat_prompt_builder.documents")
rag_pipeline.connect("chat_prompt_builder.prompt", "llm.messages")
rag_pipeline.connect("llm.replies", "answer_builder.replies")
rag_pipeline.connect("retriever.documents", "answer_builder.documents")

# Ask a question on the data you just added.
question = "Who created the Dothraki vocabulary?"
result = rag_pipeline.run(
    {
        "query_embedder": {"text": question},
        "retriever": {"top_k": 3},
        "chat_prompt_builder": {"query": question},
        "answer_builder": {"query": question},
    }
)

# For details, like which documents were used to generate the answer, look into the GeneratedAnswer object
answer: GeneratedAnswer = result["answer_builder"]["answers"][0]

# ruff: noqa: T201
print("Query: ", answer.query)
print("Answer: ", answer.data)
print("== Sources:")
for doc in answer.documents:
    print("-> ", doc.meta["file_path"])
