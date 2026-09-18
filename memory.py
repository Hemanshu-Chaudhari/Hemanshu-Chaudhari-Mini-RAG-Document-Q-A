from typing import List, Dict, Any, Optional
from google import genai
import config


class ConversationMemory:
    """
    Tracks multi-turn conversation history and generates standalone search queries
    for contextual follow-up questions.
    """

    def __init__(self, client: genai.Client, max_history_turns: int = 6):
        self.client = client
        self.max_history_turns = max_history_turns
        self.history: List[Dict[str, str]] = []

    def add_user_message(self, message: str):
        self.history.append({"role": "user", "content": message.strip()})
        self._trim_history()

    def add_assistant_message(self, message: str):
        self.history.append({"role": "model", "content": message.strip()})
        self._trim_history()

    def _trim_history(self):
        # Keep recent turns up to max_history_turns * 2 (user + model per turn)
        if len(self.history) > self.max_history_turns * 2:
            self.history = self.history[-(self.max_history_turns * 2):]

    def clear(self):
        self.history = []

    def get_history(self) -> List[Dict[str, str]]:
        return list(self.history)

    def get_formatted_history(self) -> str:
        if not self.history:
            return "No previous conversation."
        
        lines = []
        for msg in self.history:
            speaker = "User" if msg["role"] == "user" else "Assistant"
            lines.append(f"{speaker}: {msg['content']}")
        return "\n".join(lines)

    def reformulate_query(self, user_question: str) -> str:
        """
        If there is prior conversation history, reformulates the user question into a
        standalone search query to resolve references, pronouns, and implicit context.
        """
        if not self.history:
            return user_question.strip()

        recent_history = self.history[-6:]
        history_text = "\n".join(
            f"{'User' if m['role'] == 'user' else 'Assistant'}: {m['content']}" 
            for m in recent_history
        )

        prompt = f"""You are an expert search query reformulator for a Retrieval-Augmented Generation (RAG) system.
Given the conversation history and the latest user question, rephrase the question into a clear, standalone search query that can be used to search a document vector database.

Rules:
1. Resolve any pronouns (it, they, this, that, former, latter) using the conversation history.
2. If the user question is already standalone and specific, return it unchanged.
3. Do NOT attempt to answer the question.
4. Output ONLY the reformulated query and nothing else.

Conversation History:
{history_text}

Latest User Question:
{user_question}

Standalone Search Query:"""

        try:
            response = self.client.models.generate_content(
                model=config.GENERATION_MODEL,
                contents=prompt
            )
            reformulated = response.text.strip()
            # If the model returned quotes, strip them
            if reformulated.startswith('"') and reformulated.endswith('"'):
                reformulated = reformulated[1:-1].strip()
            return reformulated if reformulated else user_question.strip()
        except Exception:
            return user_question.strip()
