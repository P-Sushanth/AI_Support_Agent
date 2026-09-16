import os

class PromptManager:
    def __init__(self, prompt_dir: str = "prompts"):
        self.prompt_dir = prompt_dir

    def load_prompt(self, version: str = "system_v1") -> str:
        file_path = os.path.join(self.prompt_dir, f"{version}.txt")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Prompt version {version} not found at {file_path}")
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()

    def format_user_message(self, customer_message: str, category: str = "", context_docs: list = None) -> str:
        formatted = f"Customer Query:\n{customer_message}\n"
        if category:
            formatted += f"Category: {category}\n"
        if context_docs:
            formatted += "\nRetrieved Knowledge Base Context:\n"
            for idx, doc in enumerate(context_docs, 1):
                formatted += f"[{idx}] {doc}\n"
        return formatted
