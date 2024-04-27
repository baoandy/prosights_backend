import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
import time
import concurrent.futures

load_dotenv()

pinecone_api_key = os.environ.get("PINECONE_API_KEY")
openai_embeddings = OpenAIEmbeddings(openai_api_key=os.environ.get("OPENAI_API_KEY"), model="text-embedding-3-large")

def load_user_docs():
    start_time = time.time()
    bumble_path = "files/"
    bumble_loader = PyPDFDirectoryLoader(bumble_path, extract_images=False)
    bumble_docs = bumble_loader.load()
    print("Loaded docs")
    print("Loaded docs: ", len(bumble_docs))
    # Split the documents
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=100)
    bumble_docs = text_splitter.split_documents(bumble_docs)
    print(f"Split docs in {time.time() - start_time} seconds")
    print("Finished splitting docs, num docs: ", len(bumble_docs))
    return bumble_docs


def get_vector_store():
    vector_store =  PineconeVectorStore(
        index_name="prosights", embedding=openai_embeddings, pinecone_api_key=pinecone_api_key
    )
    # Only need to do the below once
    # vector_store.delete(delete_all=True)
    # start_time = time.time()
    # docs = load_user_docs()
    # partitions = 10
    # partitioned_docs = [docs[i::partitions] for i in range(partitions)]
    # with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
    #     futures = [executor.submit(vector_store.add_documents, partition) for partition in partitioned_docs]
    #     for future in concurrent.futures.as_completed(futures):
    #         future.result()
    # print(f"Added docs in {time.time() - start_time} seconds")
    return vector_store
