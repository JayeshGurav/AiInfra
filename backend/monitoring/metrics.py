from prometheus_client import Counter, Gauge, Histogram

TOTAL_QUERIES = Counter("rag_total_queries", "Number of accepted RAG queries")
TOTAL_INGESTIONS = Counter("rag_total_ingestions", "Number of ingestion requests")
INGESTED_DOCUMENTS = Counter("rag_ingested_documents", "Number of successfully ingested PDF documents")
EMBEDDING_LATENCY = Histogram("rag_embedding_latency_seconds", "Embedding API latency")
RETRIEVAL_LATENCY = Histogram("rag_retrieval_latency_seconds", "Qdrant retrieval latency")
RERANK_LATENCY = Histogram("rag_rerank_latency_seconds", "Reranker API latency")
GENERATION_LATENCY = Histogram("rag_generation_latency_seconds", "Generation API latency")
END_TO_END_LATENCY = Histogram("rag_end_to_end_latency_seconds", "End-to-end /ask latency")
QDRANT_HEALTHY = Gauge("rag_qdrant_healthy", "Whether Qdrant is reachable")
