import os
import json
import pickle
from typing import List, Dict, Any
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from src.retrieval.documents import DocumentChunk

class KnowledgeRetriever:
    def __init__(self, top_k: int = 3):
        self.top_k = top_k
        self.vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
        self.documents: List[DocumentChunk] = []
        self.tfidf_matrix = None
        self.is_indexed = False

    def build_index(self, documents: List[DocumentChunk]):
        self.documents = documents
        texts = [doc.text for doc in documents]
        if not texts:
            raise ValueError("Cannot index empty document corpus.")
        self.tfidf_matrix = self.vectorizer.fit_transform(texts)
        self.is_indexed = True

    def save_index(self, save_dir: str = "results/rag/index"):
        os.makedirs(save_dir, exist_ok=True)
        docs_data = [d.to_dict() for d in self.documents]
        with open(os.path.join(save_dir, "documents.json"), "w", encoding="utf-8") as f:
            json.dump(docs_data, f, ensure_ascii=False, indent=2)
            
        with open(os.path.join(save_dir, "vectorizer.pkl"), "wb") as f:
            pickle.dump(self.vectorizer, f)
            
        with open(os.path.join(save_dir, "tfidf_matrix.pkl"), "wb") as f:
            pickle.dump(self.tfidf_matrix, f)
            
        print(f"Index saved successfully to {save_dir}")

    def load_index(self, load_dir: str = "results/rag/index"):
        with open(os.path.join(load_dir, "documents.json"), "r", encoding="utf-8") as f:
            docs_data = json.load(f)
            self.documents = [
                DocumentChunk(
                    doc_id=d["doc_id"],
                    text=d["text"],
                    category=d["category"],
                    intent=d["intent"],
                    source=d["source"],
                    metadata=d.get("metadata", {})
                ) for d in docs_data
            ]
            
        with open(os.path.join(load_dir, "vectorizer.pkl"), "rb") as f:
            self.vectorizer = pickle.load(f)
            
        with open(os.path.join(load_dir, "tfidf_matrix.pkl"), "rb") as f:
            self.tfidf_matrix = pickle.load(f)
            
        self.is_indexed = True

    def retrieve(self, query: str, top_k: int = None, category_filter: str = None) -> List[Dict[str, Any]]:
        if not self.is_indexed:
            raise RuntimeError("Index has not been built or loaded.")
            
        k = top_k or self.top_k
        query_vec = self.vectorizer.transform([query])
        scores = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
        
        # Rank document indices by score descending
        ranked_indices = scores.argsort()[::-1]
        
        results = []
        for idx in ranked_indices:
            if len(results) >= k:
                break
            score = float(scores[idx])
            doc = self.documents[idx]
            
            if category_filter and doc.category != category_filter:
                continue
                
            results.append({
                "doc_id": doc.doc_id,
                "score": round(score, 4),
                "text": doc.text,
                "category": doc.category,
                "intent": doc.intent
            })
            
        return results
