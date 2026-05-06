import os
import json
import requests
from langchain_community.llms import Ollama
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# ─────────────────────────────────────────────────────────────
# LLM Backend helper
# ─────────────────────────────────────────────────────────────

OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_MODEL    = "qwen2.5:3b"          # local model via Ollama


def _is_ollama_alive() -> bool:
    """Returns True if the local Ollama server is reachable."""
    try:
        r = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=3)
        return r.status_code == 200
    except Exception:
        return False


def _call_ollama(prompt: str, model: str = OLLAMA_MODEL, timeout: int = 180) -> str:
    """
    Directly calls the Ollama /api/generate endpoint (no LangChain).
    Returns the full response text.
    """
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.4
        }
    }
    r = requests.post(
        f"{OLLAMA_BASE_URL}/api/generate",
        json=payload,
        timeout=timeout
    )
    r.raise_for_status()
    return r.json().get("response", "").strip()


def _stream_ollama(prompt: str, model: str = OLLAMA_MODEL, timeout: int = 300):
    """
    Generator — streams tokens from Ollama's /api/generate (stream=True).
    Yields individual token strings as they arrive.
    """
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": True
    }
    with requests.post(
        f"{OLLAMA_BASE_URL}/api/generate",
        json=payload,
        stream=True,
        timeout=timeout
    ) as r:
        r.raise_for_status()
        for line in r.iter_lines():
            if line:
                try:
                    chunk = json.loads(line)
                    token = chunk.get("response", "")
                    if token:
                        yield token
                    if chunk.get("done", False):
                        break
                except json.JSONDecodeError:
                    continue


# ─────────────────────────────────────────────────────────────
# Main Service
# ─────────────────────────────────────────────────────────────

class LLMService:
    """
    Unified LLM service that tries Ollama (Qwen2.5:3b) first.
    Falls back to OpenRouter (Mistral-7B) if Ollama is unavailable.
    """

    def __init__(self):
        self.use_ollama = _is_ollama_alive()

        if self.use_ollama:
            print(f"[LLMService] ✅ Using local Ollama → {OLLAMA_MODEL}")
        else:
            # OpenRouter fallback
            api_key = os.getenv("OPENROUTER_API_KEY")
            if api_key:
                self.llm = ChatOpenAI(
                    model="mistralai/mistral-7b-instruct:free",
                    openai_api_key=api_key,
                    openai_api_base="https://openrouter.ai/api/v1",
                    temperature=0.7,
                    default_headers={
                        "HTTP-Referer": "http://localhost:8501",
                        "X-Title": "Vernacular AI Tutor"
                    }
                )
                self.output_parser = StrOutputParser()
                print("[LLMService] ⚠️  Ollama unavailable — using OpenRouter.")
            else:
                self.llm = None
                print("[LLMService] ❌ No LLM configured.")

    # ── internal dispatcher (blocking) ────────────────────────────
    def _invoke(self, prompt: str, timeout: int = 180) -> str:
        """Send a plain-text prompt to whatever backend is active."""
        if self.use_ollama:
            return _call_ollama(prompt, timeout=timeout)
        elif self.llm:
            from langchain_core.messages import HumanMessage
            response = self.llm.invoke([HumanMessage(content=prompt)])
            return response.content.strip()
        else:
            return "Error: No LLM backend configured."

    # ── internal dispatcher (streaming) ───────────────────────────
    def _stream(self, prompt: str, timeout: int = 300):
        """
        Returns a generator that yields tokens one by one.
        For Ollama: true streaming via iter_lines.
        For OpenRouter: yields the full response as a single chunk.
        """
        if self.use_ollama:
            yield from _stream_ollama(prompt, timeout=timeout)
        elif self.llm:
            from langchain_core.messages import HumanMessage
            response = self.llm.invoke([HumanMessage(content=prompt)])
            yield response.content.strip()
        else:
            yield "Error: No LLM backend configured."

    # ── 1. Question Generation ─────────────────────────────────────
    def generate_questions(self, context_chunk: str) -> dict:
        """
        Returns a dict with three questions:
          { "easy": "...", "medium": "...", "conceptual": "..." }
        """
        prompt = f"""You are an expert educational tutor. Based on the text below, generate exactly 3 questions to test student understanding.

Text:
\"\"\"
{context_chunk}
\"\"\"

Rules:
- Question 1 (Easy): A simple factual/recall question.
- Question 2 (Medium): Requires understanding and application.
- Question 3 (Conceptual): Requires deep thinking or reasoning about the concept.

Respond ONLY in this exact JSON format with no extra text:
{{
  "easy": "your easy question here",
  "medium": "your medium question here",
  "conceptual": "your conceptual question here"
}}"""

        raw = self._invoke(prompt)

        # Try parsing JSON
        try:
            # Find the JSON object inside the response
            start = raw.find("{")
            end   = raw.rfind("}") + 1
            if start != -1 and end > start:
                data = json.loads(raw[start:end])
                return {
                    "easy":       data.get("easy",       "What is the main topic of this text?"),
                    "medium":     data.get("medium",     "How does the concept described apply in practice?"),
                    "conceptual": data.get("conceptual", "Why is this concept important in the broader context?")
                }
        except Exception:
            pass

        # Fallback: split by newline heuristic
        lines = [l.strip() for l in raw.split("\n") if l.strip() and "?" in l]
        questions = (lines + ["What is this topic about?", "How is this applied?", "Why does this matter?"])[:3]
        return {"easy": questions[0], "medium": questions[1], "conceptual": questions[2]}

    # ── legacy helper for backward compat ─────────────────────────
    def generate_question(self, context_chunk: str) -> str:
        qs = self.generate_questions(context_chunk)
        return qs.get("medium", "What is the main idea of this passage?")

    # ── 2. Answer Evaluation ───────────────────────────────────────
    def evaluate_answer(self, question: str, user_answer: str, context: str) -> dict:
        """
        Returns:
          {
            "verdict":       "Correct" | "Partial" | "Wrong",
            "is_correct":    bool,
            "weak_concept":  str,   # detected weak area
            "explanation":   str    # short explanation
          }
        """
        prompt = f"""You are a strict but fair academic evaluator.

Reference Context (ground truth):
\"\"\"
{context}
\"\"\"

Question: {question}
Student Answer: {user_answer}

Evaluate the student's answer based on the reference context above.

Respond ONLY in this exact JSON format:
{{
  "verdict": "Correct" or "Partial" or "Wrong",
  "weak_concept": "one short phrase describing the concept the student is missing, or 'None' if correct",
  "explanation": "a concise 1-2 sentence explanation of the evaluation"
}}"""

        raw = self._invoke(prompt)

        try:
            start = raw.find("{")
            end   = raw.rfind("}") + 1
            if start != -1 and end > start:
                data = json.loads(raw[start:end])
                verdict      = data.get("verdict", "Wrong").strip()
                weak_concept = data.get("weak_concept", "General understanding")
                explanation  = data.get("explanation", raw)
                is_correct   = verdict.lower() == "correct"
                return {
                    "verdict":      verdict,
                    "is_correct":   is_correct,
                    "weak_concept": weak_concept,
                    "explanation":  explanation
                }
        except Exception:
            pass

        # Fallback
        lines      = raw.strip().split("\n")
        first_line = lines[0].upper()
        is_correct = "CORRECT" in first_line and "INCORRECT" not in first_line and "WRONG" not in first_line
        verdict    = "Correct" if is_correct else ("Partial" if "PARTIAL" in first_line else "Wrong")
        reason     = "\n".join(lines[1:]).strip() or raw
        return {
            "verdict":      verdict,
            "is_correct":   is_correct,
            "weak_concept": "General understanding",
            "explanation":  reason
        }

    # ── 3. Explanation (Tamil / Tanglish / English) ────────────────
    def _explain_prompt(self, weak_concept: str, context: str, language: str = "tamil") -> str:
        """Builds the explanation prompt for a given language mode."""
        trimmed_context = context[:1200].strip()

        if language == "tanglish":
            return f"""You are a friendly AI tutor who speaks in natural Tanglish (Tamil + English mix).

Rules:
- Use casual WhatsApp-style Tanglish
- DO NOT translate word-by-word into Tamil grammar
- Keep sentences simple, natural, and friendly
- Mix English technical terms naturally (e.g. "machine learning", "neural network")
- NEVER use Tamil scripts (அ, இ), only English alphabet (A-Z)
- Write normal words with no hyphens between letters.

Examples of Good Tanglish Style:
Q: What is AI?
A: AI na machine ku human madhiri think panna solli train panradhu bro.

Q: What is ML?
A: ML na data use panni machine ah patterns learn panna vekkaradhu thaan.

Bad Example (DO NOT DO THIS):
"AI yu nanal paththaippattinum illa" <- Too robotic and broken translation.

Now explain the concept below in 3-4 sentences of good, natural Tanglish. Use a real-life analogy.

Concept: "{weak_concept}"

Study Content (for reference):
\"\"\"
{trimmed_context}
\"\"\"

Explanation (in natural Tanglish):"""

        elif language == "english":
            return f"""You are a patient and friendly tutor. Read the study content below.

Study Content:
\"\"\"
{trimmed_context}
\"\"\"

The student did not understand: "{weak_concept}"

Explain this concept clearly in simple, everyday English with a real-life analogy or example. Keep it concise — 3-4 sentences only. Avoid technical jargon."""

        else:  # default: tamil script
            return f"""நீங்கள் ஒரு பொறுமையான ஆசிரியர். கீழே உள்ள பாடத்திட்ட உள்ளடக்கத்தை படியுங்கள்.

பாட உள்ளடக்கம்:
\"\"\"
{trimmed_context}
\"\"\"

மாணவர் புரிந்துகொள்ளாத கருத்து: "{weak_concept}"

இந்த கருத்தை மிகவும் எளிமையான தமிழில், ஒரு அன்றாட உதாரணத்துடன் விளக்கவும்.
கடினமான சொல்லாட்சி தவிர்க்கவும். 3-4 வாக்கியங்களில் மட்டும் சொல்லுங்கள்.

Explain this concept in simple Tamil script with one real-life example in 3-4 sentences only."""

    def explain_stream(self, weak_concept: str, context: str, language: str = "tamil"):
        """Streaming explanation — yields tokens for st.write_stream. Language: tamil/tanglish/english."""
        prompt = self._explain_prompt(weak_concept, context, language)
        try:
            yield from self._stream(prompt, timeout=300)
        except Exception as e:
            err = str(e)
            if "ReadTimeout" in err or "Timeout" in err or "timed out" in err.lower():
                yield (
                    f"⏱️ **Explanation timed out** — the model took too long.\n\n"
                    f"**Concept:** {weak_concept}\n\n"
                    "Please try again or check that Ollama is running with enough resources."
                )
            else:
                yield f"❌ Error: {err}"

    # kept for backward compat
    def explain_in_tamil_stream(self, weak_concept: str, context: str, language: str = "tamil"):
        yield from self.explain_stream(weak_concept, context, language)

    def explain_in_tamil(self, weak_concept: str, context: str, language: str = "tamil") -> str:
        return "".join(self.explain_stream(weak_concept, context, language))

    # ── 4. Follow-up Question ─────────────────────────────────────
    def generate_followup_question(self, weak_concept: str, context: str) -> str:
        """
        Generates a single targeted follow-up question on the weak concept.
        """
        prompt = f"""You are a tutor helping a student who struggled with: "{weak_concept}"

Based on this context:
\"\"\"
{context[:800]}
\"\"\"

Generate ONE short follow-up question to check if the student now understands "{weak_concept}".
The question should be simple, direct, and different from what was asked before.
Output ONLY the question, nothing else."""

        return self._invoke(prompt).strip().strip('"')

    def generate_followup_question_stream(self, weak_concept: str, context: str):
        """Streaming version of follow-up question generation."""
        prompt = f"""You are a tutor helping a student who struggled with: "{weak_concept}"

Based on this context:
\"\"\"
{context[:800]}
\"\"\"

Generate ONE short follow-up question to check if the student now understands "{weak_concept}".
The question should be simple, direct, and different from what was asked before.
Output ONLY the question, nothing else."""
        yield from self._stream(prompt)

    # ── 5. Follow-up Evaluation ───────────────────────────────────
    def evaluate_followup(self, question: str, user_answer: str, weak_concept: str) -> dict:
        """
        Quickly evaluate if the student improved on the weak concept.
        """
        prompt = f"""You are evaluating a student's follow-up response about "{weak_concept}".

Follow-up Question: {question}
Student Answer: {user_answer}

Did the student show improvement in understanding "{weak_concept}"?
Respond ONLY in this exact JSON format:
{{
  "improved": true or false,
  "feedback": "one short encouraging sentence"
}}"""

        raw = self._invoke(prompt)
        try:
            start = raw.find("{")
            end   = raw.rfind("}") + 1
            if start != -1 and end > start:
                data = json.loads(raw[start:end])
                return {
                    "improved": bool(data.get("improved", False)),
                    "feedback": data.get("feedback", "Keep practising!")
                }
        except Exception:
            pass
        improved = "true" in raw.lower() or "yes" in raw.lower() or "correct" in raw.lower()
        return {"improved": improved, "feedback": raw.strip()[:200]}

    # ── Backend info ───────────────────────────────────────────────
    def get_backend_info(self) -> str:
        if self.use_ollama:
            return f"🟢 Local Ollama — {OLLAMA_MODEL}"
        elif self.llm:
            return "🟡 OpenRouter — mistral-7b-instruct (free)"
        else:
            return "🔴 No LLM configured"
