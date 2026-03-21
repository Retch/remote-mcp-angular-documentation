import modules.retrieval as Retriever

retrieved_docs = Retriever.find_docs("ngClass")

print(len(retrieved_docs))