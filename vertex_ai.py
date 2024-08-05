from langchain_google_vertexai import VertexAI
from google.cloud import bigquery
from langchain_google_vertexai import VertexAIEmbeddings
from langchain_community.vectorstores import BigQueryVectorSearch
import os
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

PROJECT_ID = "solutions-data"
DATASET = "companyData"
TABLE = "company_data"
TABLEEMBED = "company_detail_embedding"
REGION = "asia-southeast1"
JSON_KEY_PATH = "credential/vertexAi.json"
# REGION = "US"


os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = JSON_KEY_PATH

embedding_model = VertexAIEmbeddings(
    model_name="text-multilingual-embedding-002", project=PROJECT_ID
)
bq_vector_datasource= BigQueryVectorSearch(
    project_id=PROJECT_ID,
    dataset_name=DATASET,
    table_name=TABLEEMBED,
    location=REGION,
    embedding=embedding_model,
)
retriever = bq_vector_datasource.as_retriever(search_type="mmr")



def ai_config(config_prompt, model_name):
    llm = VertexAI(model_name=model_name)
    if config_prompt != "":
        system_prompt = (config_prompt)
    else:
        system_prompt = (
            "you are an Ai name จิดริ้ด"
            "Use the given context to answer the question. "
            "If you don't know the answer, say you don't know. "
            "You can find closest anwser and give to user"
            "use all embedding data to anwser the question"
            "response thai language"
            "Context: {context}"
        )
    return (system_prompt, llm)


def promp_ai(query, config_prompt = "", model_name = "gemini-1.5-pro-001"):
    system_prompt, llm = ai_config(config_prompt, model_name)
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}"),
        ]
    )

    #Create a chain for passing a list of Documents to a model.
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    #Create retrieval chain that retrieves documents and then passes them on.
    chain = create_retrieval_chain(retriever, question_answer_chain)
    result = chain.invoke({"input": query})

    return result.get("answer")


