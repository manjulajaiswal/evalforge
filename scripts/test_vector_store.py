from app.rag_pipeline import retrieve_context

question = "How many days do customers have to request a refund?"

contexts, sources = retrieve_context(question, n_results=3)

print("Question:")
print(question)

print("\nRetrieved Context:")
for context in contexts:
    print(context)

print("\nSources:")
for source in sources:
    print(source["source"])