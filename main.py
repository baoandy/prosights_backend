from typing import Iterable, List
from fastapi import FastAPI
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from fastapi.responses import StreamingResponse
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import HumanMessagePromptTemplate, MessagesPlaceholder
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from pydantic import BaseModel
from langchain_core.messages import AIMessageChunk

from docs import get_vector_store


def streaming_parse(chunks: Iterable[AIMessageChunk]) -> Iterable[str]:
    for chunk in chunks:
        if "answer" in chunk:
            yield chunk["answer"]


load_dotenv()
openai_api_key = os.environ.get("OPENAI_API_KEY")


prompt_string = """
You are an assistant to an expert investor at a top tier private equity firm. Given a user query, you need to provide a detailed response to the investor given the context.
User Question:
<input>{input}</input>

Context for Interaction:
<context>{context}</context>
"""

store = {}

def setup_chain():
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", prompt_string),
            MessagesPlaceholder(variable_name="history"),
            HumanMessagePromptTemplate.from_template("{input}"),
        ]
    )

    llm = ChatOpenAI(
        openai_api_key=openai_api_key,
        model="gpt-3.5-turbo",
        streaming=True,
        temperature=0.1,
    )
    vector_store = get_vector_store()
    retriever = vector_store.as_retriever()

    document_chain = create_stuff_documents_chain(llm, prompt)
    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    answer_question_llm_with_history = RunnableWithMessageHistory(
        retrieval_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="history",
        output_messages_key="answer",
    )

    final_chain = answer_question_llm_with_history | streaming_parse
    return final_chain


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]


chain = setup_chain()
app = FastAPI()


class ChatInput(BaseModel):
    input: str
    messages: List[str]


@app.post("/chat/")
def simple_test_chain(chat_input: ChatInput):
    input_string = chat_input.input

    vector_store = get_vector_store()
    retriever = vector_store.as_retriever()

    docs = retriever.invoke(input_string)
    print(docs)
    print("Number of docs: ", len(docs))

    response = chain.stream(
        {"input": input_string},
        config={"configurable": {"session_id": "123"}},
    )
    return StreamingResponse(response, media_type="text/event-stream")


@app.get("/")
def simple_test_chain():
    input_string = "What were Wells Fargo's key conclusions on Bumble?"
    
    vector_store = get_vector_store()
    retriever = vector_store.as_retriever()

    docs = retriever.invoke(input_string)
    print(docs)

    print("Number of docs: ", len(docs))

    response = chain.stream(
        {"input": input_string},
        config={"configurable": {"session_id": "123"}},
    )
    return StreamingResponse(response, media_type="text/event-stream")
