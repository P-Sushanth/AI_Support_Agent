import os
from src.retrieval.documents import build_knowledge_corpus_from_train
from src.retrieval.retriever import KnowledgeRetriever

def build_and_save_rag_index(train_path: str = "data/processed/train.jsonl", save_dir: str = "results/rag/index"):
    chunks = build_knowledge_corpus_from_train(train_path)
    retriever = KnowledgeRetriever()
    retriever.build_index(chunks)
    retriever.save_index(save_dir)
    print(f"RAG Knowledge Index built successfully with {len(chunks)} documents!")

if __name__ == "__main__":
    build_and_save_rag_index()
