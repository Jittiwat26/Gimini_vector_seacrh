from langchain_google_vertexai import ChatVertexAI
from google.cloud import bigquery
from langchain_google_vertexai import VertexAIEmbeddings
from langchain_community.vectorstores import BigQueryVectorSearch
import os
from langchain.chains import create_retrieval_chain, create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory

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




def ai_config(config_prompt, model_name):
    llm = ChatVertexAI(model_name=model_name, max_tokens=None, max_retries=2)
    if config_prompt != "":
        system_prompt = (config_prompt)
    else:
        system_prompt = """
            "you are an Ai name จิดริ้ด"
            "Use the given context to answer the question. "
            "If you don't know the answer, say you don't know. "
            "You can find closest anwser and give to user"
            "use all embedding data to anwser the question"
            "response thai language"
            "Context: {context}"
        """
    return (system_prompt, llm)




store = {}
def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

# def promp_ai(query, config_prompt = "", model_name = "gemini-1.5-pro-001", session_id = ""):
#     system_prompt, llm = ai_config(config_prompt, model_name)
#     prompt = ChatPromptTemplate.from_messages(
#         [
#             ("system", system_prompt),
#             ("human", "{input}"),
#         ]
#     )
#     retriever = bq_vector_datasource.as_retriever(search_type="mmr")
#     history_aware_retriever = create_history_aware_retriever(
#     llm, retriever, prompt
#     )
#     #Create a chain for passing a list of Documents to a model.
#     question_answer_chain = create_stuff_documents_chain(llm, prompt)
#     #Create retrieval chain that retrieves documents and then passes them on.
#     chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
#     if session_id == "":
#         result = chain.invoke({"input": query})
#     else:
#
#         conversational_rag_chain = RunnableWithMessageHistory(
#             chain,
#             get_session_history,
#             input_messages_key="input",
#             history_messages_key="chat_history",
#             output_messages_key="answer",
#         )
#         result = conversational_rag_chain.invoke({"input": query},config={"configurable": {"session_id": session_id}})
#     print("**************************")
#     print(store)
#     print("**************************")
#
#     return result.get("answer")


def promp_ai(query, config_prompt="", model_name="gemini-1.5-pro-001", session_id=""):
    system_prompt, llm = ai_config(config_prompt, model_name)

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}"),
        ]
    )

    retriever = bq_vector_datasource.as_retriever(search_type="mmr")
    history_aware_retriever = create_history_aware_retriever(llm, retriever, prompt)

    # Create a chain for passing a list of Documents to a model.
    question_answer_chain = create_stuff_documents_chain(llm, prompt)

    # Create retrieval chain that retrieves documents and then passes them on.
    chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    # Here we need to fetch the context first.
    context = ""  # Initialize an empty context
    if session_id:  # Only fetch if session_id is provided
        message_history = get_session_history(session_id)
        context = " ".join([msg.content for msg in message_history.messages])
        conversational_rag_chain = RunnableWithMessageHistory(
            chain,
            get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer",
        )
    print(context)
    # If no session_id, just use the query directly
    result = conversational_rag_chain.invoke({"input": query, "context": context},config={"configurable": {"session_id": session_id}})

    print("**************************")
    print(store)
    print("**************************")

    return result.get("answer")
