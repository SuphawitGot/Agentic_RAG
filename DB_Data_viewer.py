from ragsale.rag.vector_store import get_collection

collection = get_collection()
print("Total records:", collection.count())

# Read five saved records.
data = collection.get(
    limit=5,
    include=["documents", "metadatas", "embeddings"],
)

for i, record_id in enumerate(data["ids"]):
    print("\nID:", record_id)
    print("Text:", data["documents"][i])
    print("Metadata:", data["metadatas"][i])
    print("Embedding preview:", data["embeddings"][i][:5])
    print("Dimensions:", len(data["embeddings"][i]))