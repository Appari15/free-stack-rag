# Prompt Engineering for RAG Systems

## Why Prompts Matter in RAG

The quality of a RAG system depends heavily on how you instruct the LLM to use the retrieved context. A poorly crafted prompt can lead to:

- Ignoring the context and using training knowledge instead
- Cherry-picking from context while missing key information
- Failing to cite sources
- Verbose, unfocused answers

## Key Principles

### 1. Explicit Grounding Instructions

Tell the model exactly what to do with the context:

**Bad**: "Answer the question."
**Good**: "Answer the question using ONLY the provided context documents. If the context doesn't contain the answer, say so."

### 2. Source Citation

Include source tags in the context and instruct the model to reference them:

```
[Source: architecture.md]
RAG systems use vector databases for retrieval.
```

Then instruct: "Cite sources using [Source: filename] tags."

### 3. Graceful Refusal

The model should admit when it doesn't have enough information rather than hallucinating:

"If the context does not contain enough information to answer, respond with: I don't have enough information in the provided documents."

### 4. Output Structure

Guide the format:
- "Use bullet points for multi-part answers"
- "Be concise but thorough"
- "Do not start with 'Based on the context'"

## Temperature Settings

For RAG systems, lower temperature (0.0-0.2) is almost always better:

- **0.0-0.1**: Most factual, least creative. Best for question-answering.
- **0.2-0.5**: Slight variation in phrasing. Good for summaries.
- **0.7+**: Creative, but may deviate from context. Avoid for RAG.

## Common Failure Modes

1. **Context window overflow**: Too many chunks overwhelm the prompt. Solution: limit top_k and chunk size.
2. **Irrelevant context**: Low-quality retrieval leads to confused answers. Solution: increase minimum relevance score.
3. **Lost in the middle**: LLMs tend to focus on the beginning and end of long contexts. Solution: put the most relevant chunks first.
