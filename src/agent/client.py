import os
import json
import time
import re
import hashlib
import urllib.request
import urllib.error
from typing import Dict, Any, Optional

class LLMClient:
    """
    Provider abstraction for LLM inference connecting directly to local Ollama LLMs
    (qwen3.5:2b, qwen3.5:9b, gemma4:12b) or mock provider for unit testing.
    """
    def __init__(self, provider: str = "ollama", model_name: str = "qwen3.5:2b", temperature: float = 0.0, cache_dir: str = "results/baseline/cache", ollama_host: str = "http://localhost:11434"):
        self.provider = provider
        self.model_name = model_name
        self.temperature = temperature
        self.cache_dir = cache_dir
        self.ollama_host = ollama_host.rstrip('/')
        os.makedirs(self.cache_dir, exist_ok=True)

    def _get_cache_key(self, system_prompt: str, user_message: str) -> str:
        raw = f"{self.provider}:{self.model_name}:{self.temperature}:{system_prompt}:{user_message}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def generate(self, system_prompt: str, user_message: str, use_cache: bool = True) -> Dict[str, Any]:
        cache_key = self._get_cache_key(system_prompt, user_message)
        cache_path = os.path.join(self.cache_dir, f"{cache_key}.json")
        
        if use_cache and os.path.exists(cache_path):
            with open(cache_path, "r", encoding="utf-8") as f:
                cached_res = json.load(f)
                cached_res["cached"] = True
                return cached_res

        start_time = time.time()
        
        if self.provider == "ollama":
            response_data = self._ollama_generate(system_prompt, user_message)
        else:
            response_data = self._mock_generate(user_message)
            
        latency_ms = round((time.time() - start_time) * 1000, 2)
        
        result = {
            "response": response_data,
            "latency_ms": latency_ms,
            "model_name": self.model_name,
            "cached": False
        }
        
        if use_cache:
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2)
                
        return result

    def _ollama_generate(self, system_prompt: str, user_message: str) -> Dict[str, Any]:
        url = f"{self.ollama_host}/api/generate"
        
        structured_prompt = (
            f"{system_prompt}\n\n"
            "Return valid JSON only matching the schema:\n"
            "{\n"
            '  "intent": "<short_intent>",\n'
            '  "response": "<answer_text>",\n'
            '  "should_escalate": <true|false>,\n'
            '  "escalation_reason": "<reason_or_null>",\n'
            '  "confidence": 0.95,\n'
            '  "sources": []\n'
            "}\n\n"
            f"User Query:\n{user_message}"
        )
        
        payload = {
            "model": self.model_name,
            "prompt": structured_prompt,
            "stream": False,
            "format": "json",
            "options": {
                "temperature": self.temperature
            }
        }
        
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"}
        )
        
        try:
            with urllib.request.urlopen(req, timeout=45) as response:
                res_data = json.loads(response.read().decode("utf-8"))
                raw_text = res_data.get("response", "").strip()
                
                parsed = self._extract_json(raw_text)
                if parsed:
                    return parsed
                else:
                    return {
                        "intent": "general_support",
                        "response": raw_text or "Thank you for reaching out to support.",
                        "should_escalate": False,
                        "escalation_reason": None,
                        "confidence": 0.85,
                        "sources": []
                    }
        except Exception as e:
            # Re-raise or handle connection errors cleanly
            raise RuntimeError(f"Ollama API call to model {self.model_name} failed: {e}")

    def _extract_json(self, text: str) -> Optional[Dict[str, Any]]:
        if not text:
            return None
        try:
            return json.loads(text)
        except Exception:
            pass
            
        match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except Exception:
                pass
                
        match = re.search(r"(\{.*\})", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except Exception:
                pass
                
        return None

    def _mock_generate(self, user_message: str) -> Dict[str, Any]:
        msg_lower = user_message.lower()
        
        if any(w in msg_lower for w in ["hacker", "2fa", "breach", "cyberattack", "locked out"]):
            return {
                "intent": "security_incident",
                "response": "Security alert detected. Your issue is being escalated immediately to our Security Escalation Team to freeze account activity.",
                "should_escalate": True,
                "escalation_reason": "Security incident / account takeover threat detected",
                "confidence": 0.98,
                "sources": []
            }
        elif any(w in msg_lower for w in ["vat", "tax-exempt", "wire credit", "4,250", "4250"]):
            return {
                "intent": "corporate_billing_tax",
                "response": "This retroactive corporate tax exemption wire adjustment request is escalated to our Billing Compliance team.",
                "should_escalate": True,
                "escalation_reason": "Corporate tax-exempt wire credit adjustment",
                "confidence": 0.95,
                "sources": []
            }
        elif any(w in msg_lower for w in ["90 days", "past the 30 day", "exception override", "medical emergency"]):
            return {
                "intent": "out_of_policy_return",
                "response": "Your return request is past our standard 30-day window and has been escalated to a Senior Support Supervisor for policy exception review.",
                "should_escalate": True,
                "escalation_reason": "Out-of-policy return exception request",
                "confidence": 0.90,
                "sources": []
            }
        elif any(w in msg_lower for w in ["human manager", "talk to human", "get me a human"]):
            return {
                "intent": "human_escalation_request",
                "response": "I am transferring your request directly to a human support representative as requested.",
                "should_escalate": True,
                "escalation_reason": "Customer explicitly requested human support",
                "confidence": 0.99,
                "sources": []
            }
        elif "cancel" in msg_lower or "cancelling" in msg_lower:
            return {
                "intent": "order_cancellation",
                "response": "To cancel your order, navigate to My Orders, select the item, and click 'Cancel Order' before it ships.",
                "should_escalate": False,
                "escalation_reason": None,
                "confidence": 0.92,
                "sources": []
            }
        elif "refund" in msg_lower or "charged twice" in msg_lower or "invoice" in msg_lower:
            return {
                "intent": "billing_refund",
                "response": "We have verified your invoice details. A refund has been initiated to your original payment method in 3-5 business days.",
                "should_escalate": False,
                "escalation_reason": None,
                "confidence": 0.94,
                "sources": []
            }
        else:
            return {
                "intent": "general_support",
                "response": "Thank you for contacting customer support. We are reviewing your inquiry and will provide detailed guidance shortly.",
                "should_escalate": False,
                "escalation_reason": None,
                "confidence": 0.85,
                "sources": []
            }
