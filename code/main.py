import asyncio
from EmbeddingModel import SciBERTEmbeddingModel  # Adjust the import based on your file structure

async def main():
    sci_bert_model = SciBERTEmbeddingModel()
    # Example texts to embed
    texts = ["This is a scientific sentence.", "Another example of a scientific sentence."]
    embeddings = await sci_bert_model.embed_documents(texts)
    print("Embeddings:", embeddings)

if __name__ == "__main__":
    asyncio.run(main())