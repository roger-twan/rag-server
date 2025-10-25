from app.services.retriever import retrieve
from app.utils.gemini_client import gemini_client


async def generate_answer(query: str):
    docs = await retrieve(query)
    content = "\n\n".join([doc for doc in docs])
    prompt = f"""
You are a helpful assistant.
Answer the question based on the context provided.

Context: {content}

Question: {query}
"""
    response = gemini_client.models.generate_content(
        model="gemini-2.5-pro",
        contents=prompt,
    )
    return response.text
