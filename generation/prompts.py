"""
All prompts in one place for easy iteration and versioning.

Prompt engineering decisions:
  1. Strict grounding — "ONLY use the context"
  2. Source citation — traceable answers
  3. Explicit refusal — "say I don't have enough info" vs hallucinating
  4. Structured output — bullet points for complex answers
"""

SYSTEM_PROMPT = """\
You are a precise, helpful AI assistant that answers questions based
ONLY on the provided context documents.

Rules:
1. Use ONLY information from the context below. Never use prior knowledge.
2. Cite your sources using [Source: filename] tags from the context.
3. If the context does not contain enough information, say exactly:
   "I don't have enough information in the provided documents to answer this."
4. Be concise but thorough. Use bullet points for multi-part answers.
5. If sources conflict, acknowledge the discrepancy.
6. Do not start your answer with "Based on the context" or similar preamble.
"""

RAG_PROMPT = """\
Context Documents:
{context}

---

Question: {query}

Answer the question using ONLY the context documents above."""


REFINE_PROMPT = """\
You previously answered: {previous_answer}

Here is additional context:
{new_context}

Refine your answer if the new context adds relevant information.
If not, keep the original answer. Always cite sources."""


def build_rag_prompt(query: str, context: str) -> str:
    return RAG_PROMPT.format(context=context, query=query)


def build_context_block(chunks) -> str:
    """
    Format retrieved chunks into a context block for the LLM.
    Each chunk is tagged with its source for citation.
    """
    blocks = []
    for i, chunk in enumerate(chunks, 1):
        source = chunk.metadata.source if hasattr(chunk, "metadata") else "unknown"
        blocks.append(f"[Source: {source}]\n{chunk.content}")
    return "\n\n---\n\n".join(blocks)
