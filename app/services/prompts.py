from langchain_core.prompts import ChatPromptTemplate

prompt_template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Answer strictly from the provided context. If the context is insufficient, say you don't have enough information. Be concise and natural.",
        ),
        (
            "human",
            """Context:
{context}

Question: {question}

Answer:""",
        ),
    ]
)


fallback_prompt_template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "If the user is greeting you, thanking you, or making simple small talk, respond naturally and briefly. If the user asks for factual or personal knowledge that would require retrieved context, say you don't have enough information.",
        ),
        (
            "human",
            """Question: {question}

Answer:""",
        ),
    ]
)


rewrite_prompt_template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Rewrite the user's latest question into a standalone search query. Use chat history only to resolve references. Return only the rewritten query.",
        ),
        (
            "human",
            """Chat history:
{history}

Latest question: {question}

Standalone search query:""",
        ),
    ]
)
