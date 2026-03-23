# Vector Databases for RAG Systems

## What is a Vector Database?

A vector database is a specialized database designed to store, index, and search high-dimensional vectors efficiently. In RAG systems, these vectors are numerical representations (embeddings) of text chunks.

## How Vector Search Works

### Embedding Space

When text is converted to a vector, semantically similar texts end up close together in the embedding space. "The cat sat on the mat" and "A feline rested on the rug" would have vectors pointing in nearly the same direction, even though they share few words.

### Similarity Metrics

- **Cosine Similarity**: Measures the angle between vectors. Most common for text embeddings.
- **Euclidean Distance**: Measures straight-line distance. Sensitive to vector magnitude.
- **Dot Product**: Fast to compute. Equivalent to cosine similarity for normalized vectors.

### Approximate Nearest Neighbor (ANN)

Exact nearest neighbor search is O(n) — too slow for millions of vectors. ANN algorithms trade a small amount of accuracy for dramatic speed improvements:

- **HNSW (Hierarchical Navigable Small World)**: Graph-based. Best overall quality/speed tradeoff.
- **IVF (Inverted File Index)**: Partition-based. Good for very large datasets.
- **Product Quantization**: Compression-based. Reduces memory usage.

## Popular Vector Databases

| Database          | Type              | Best For                    |
|-------------------|-------------------|-----------------------------|
| ChromaDB          | Embedded/Server   | Prototypes, small-medium    |
| Pinecone          | Managed Cloud     | Production, fully managed   |
| Weaviate          | Self-hosted/Cloud | Hybrid search, GraphQL API  |
| Qdrant            | Self-hosted/Cloud | Performance, Rust-based    |
| OpenSearch        | Self-hosted/AWS   | Existing OpenSearch users  |
| pgvector          | PostgreSQL ext    | Teams already using Postgres|

## ChromaDB in Detail

ChromaDB is used in this project because:

1. **Zero configuration**: Works out of the box with Docker.
2. **Built-in embedding**: Can embed text automatically (we manage our own for consistency).
3. **Metadata filtering**: Supports rich metadata queries alongside vector search.
4. **Persistent storage**: Data survives container restarts.
5. **Free and open source**: No API keys or cloud accounts needed.
