# What is Retrieval-Augmented Generation (RAG)?

## Overview

Retrieval-Augmented Generation (RAG) is an AI architecture pattern that enhances Large Language Models (LLMs) by providing them with external knowledge at inference time. Instead of relying solely on information learned during training, RAG systems retrieve relevant documents from a knowledge base and include them in the LLM's context window.

## The Problem RAG Solves

LLMs have several well-known limitations:

- **Knowledge cutoff**: Training data has a fixed date. The model doesn't know about events after that date.
- **Hallucinations**: LLMs can generate plausible-sounding but factually incorrect information.
- **No source attribution**: Standard LLM responses don't cite where information came from.
- **Expensive updates**: Fine-tuning a model to incorporate new knowledge is costly and slow.

RAG addresses all of these by grounding the model's responses in retrieved evidence.

## How RAG Works

The RAG pipeline has three stages:

### 1. Ingestion (Offline)

Documents are processed and stored for later retrieval:

- **Extract**: Read text from PDFs, Markdown, HTML, etc.
- **Chunk**: Split documents into smaller pieces (typically 256-512 tokens) with overlap to prevent information loss at boundaries.
- **Embed**: Convert each chunk into a dense vector using an embedding model.
- **Store**: Save vectors in a vector database for fast similarity search.

### 2. Retrieval (Online)

When a user asks a question:

- The query is embedded using the same embedding model.
- The vector database performs approximate nearest neighbor (ANN) search.
- The top-K most similar chunks are returned as context.

### 3. Generation (Online)

The retrieved chunks are inserted into the LLM's prompt:

- A system prompt instructs the model to answer only from the provided context.
- The model generates a response grounded in the retrieved evidence.
- Sources can be cited because we know which chunks were used.

## Benefits

1. **Reduced hallucinations**: The model has factual context to reference.
2. **Up-to-date knowledge**: New documents can be added without retraining.
3. **Source attribution**: Every answer can cite its sources.
4. **Cost effective**: Much cheaper than fine-tuning for most use cases.
5. **Data privacy**: Sensitive documents stay in your own infrastructure.
