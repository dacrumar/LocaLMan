from llama_index.embeddings.ollama import OllamaEmbedding

embed = OllamaEmbedding(model_name="mxbai-embed-large", base_url="http://localhost:11434")

v = embed.get_text_embedding("SAP HANA components")
print(f"Dimensiones: {len(v)}")        # 1024
print(f"Primeros 10 valores: {v[:10]}")
print(f"Min: {min(v):.4f}  Max: {max(v):.4f}")