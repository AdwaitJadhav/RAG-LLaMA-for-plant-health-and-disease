import chromadb
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext
from llama_index.vector_stores.chroma import ChromaVectorStore
import logging

def create_index(embed_model, persist_directory):
    reader = SimpleDirectoryReader(input_dir="./Knowledge base", recursive=True)
    documents = reader.load_data()
    
    logging.info("Creating index with `%d` documents", len(documents))
    
    chroma_client = chromadb.PersistentClient(path=persist_directory)
    chroma_collection = chroma_client.create_collection("grape")
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    
    index = VectorStoreIndex.from_documents(documents, storage_context=storage_context, embed_model=embed_model)
    index.storage_context.persist(persist_directory)
    return index

def load_index(embed_model, persist_directory):
    chroma_client = chromadb.PersistentClient(path=persist_directory)
    chroma_collection = chroma_client.get_collection("grape")
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    index = VectorStoreIndex.from_vector_store(vector_store, embed_model=embed_model)
    return index
