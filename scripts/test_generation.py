from app.rag_pipeline import retrieve_context, generate_answer


question = "How many days do customers have to request a refund?"

contexts, sources = retrieve_context(
    question,
    n_results=3,
)

answer = generate_answer(
    question,
    contexts,
)

print("Question:")
print(question)

print("\nAnswer:")
print(answer)

print("\nSources:")
for source in sources:
    print("-", source["source"])