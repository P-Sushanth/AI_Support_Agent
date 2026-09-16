import os
import pytest
from src.retrieval.documents import DocumentChunk
from src.retrieval.retriever import KnowledgeRetriever
from src.agent.rag_agent import RAGSupportAgent
from src.agent.schemas import CustomerTicketInput

def test_document_chunk_creation():
    doc = DocumentChunk(
        doc_id="DOC-1",
        text="Support answer for order cancellation",
        category="ORDER",
        intent="cancel_order",
        source="bitext"
    )
    assert doc.doc_id == "DOC-1"
    assert doc.category == "ORDER"
    assert doc.to_dict()["source"] == "bitext"

def test_retriever_build_and_retrieve(tmp_path):
    docs = [
        DocumentChunk("D1", "How to request a refund within 30 days return policy", "REFUND", "refund", "test"),
        DocumentChunk("D2", "How to reset your password and manage 2FA settings", "ACCOUNT", "password", "test"),
        DocumentChunk("D3", "Tracking your shipping order via FedEx delivery", "SHIPPING", "shipping", "test")
    ]
    
    retriever = KnowledgeRetriever(top_k=2)
    retriever.build_index(docs)
    
    # Save & reload index test
    index_dir = str(tmp_path / "index")
    retriever.save_index(index_dir)
    
    reloaded = KnowledgeRetriever(top_k=2)
    reloaded.load_index(index_dir)
    
    results = reloaded.retrieve("password reset 2FA", top_k=2)
    assert len(results) == 2
    assert results[0]["doc_id"] == "D2"
    assert results[0]["score"] > 0.0

def test_rag_agent_source_propagation(tmp_path):
    docs = [
        DocumentChunk("D-PASS", "Password reset instructions: click forgot password.", "ACCOUNT", "reset_password", "test"),
        DocumentChunk("D-REFUND", "Refund policy: 30 days return window.", "BILLING", "refund", "test")
    ]
    retriever = KnowledgeRetriever(top_k=1)
    retriever.build_index(docs)
    
    agent = RAGSupportAgent(retriever=retriever, top_k=1)
    inp = CustomerTicketInput(ticket_id="T-RAG-1", customer_message="Forgot my password reset link")
    
    result = agent.process_ticket(inp, use_cache=False)
    assert "D-PASS" in result["output"]["sources"]
    assert len(result["retrieved_documents"]) == 1
