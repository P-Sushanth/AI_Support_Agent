import json
from typing import List, Dict, Any

class DocumentChunk:
    def __init__(self, doc_id: str, text: str, category: str, intent: str, source: str, metadata: Dict[str, Any] = None):
        self.doc_id = doc_id
        self.text = text
        self.category = category
        self.intent = intent
        self.source = source
        self.metadata = metadata or {}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "doc_id": self.doc_id,
            "text": self.text,
            "category": self.category,
            "intent": self.intent,
            "source": self.source,
            "metadata": self.metadata
        }

def build_knowledge_corpus_from_train(train_jsonl_path: str = "data/processed/train.jsonl") -> List[DocumentChunk]:
    chunks = []
    with open(train_jsonl_path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            
            # Combine customer message context and agent resolution answer into a searchable document
            msg = rec.get("customer_message", "").strip()
            resp = rec.get("agent_response", "").strip()
            cat = rec.get("category", "GENERAL")
            intent = rec.get("intent", "general_query")
            
            if not resp and not msg:
                continue
                
            doc_id = f"DOC-{rec.get('ticket_id', f'TRAIN-{idx:05d}')}"
            content_text = f"Support Category: {cat}\nIntent: {intent}\nCustomer Issue: {msg}\nResolution Policy & Answer: {resp}"
            
            chunk = DocumentChunk(
                doc_id=doc_id,
                text=content_text,
                category=cat,
                intent=intent,
                source=rec.get("source_dataset", "train_split"),
                metadata={"should_escalate": rec.get("should_escalate", False)}
            )
            chunks.append(chunk)
            
    print(f"Built knowledge corpus of {len(chunks)} document chunks from {train_jsonl_path}")
    return chunks
