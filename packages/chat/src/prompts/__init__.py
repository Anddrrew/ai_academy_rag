def system(context: str = "") -> str:
    prompt = (
        "You are a helpful assistant with access to a knowledge base.\n"
        "Answer the user's question based on the provided context.\n"
        "Provide a complete answer, covering all relevant points from the context.\n"
        "If the context does not contain the answer, say: "
        "\"The knowledge base does not contain information about this topic.\" "
        "and then answer from your own knowledge, clearly marking it as: "
        "\"Based on my own knowledge: ...\""
    )
    if context:
        prompt += (
            "\n\n"
            "If you use information from the context, add a \"Sources:\" section at the end "
            "listing each used source as a markdown link (e.g. [filename.pdf](url)). "
            "Include all sources that contributed to your answer."
            "\n\n"
            f"Relevant context:\n{context}"
        )
    return prompt


def extract_search_query() -> str:
    return (
        "Your task is to extract a search query from the user's message to find relevant information in a vector database.\n"
        "The user's message may be a direct question, a request, a command, or an implicit information need.\n"
        "For message that seeks information or knowledge, rewrite it into a clear, search-optimized question.\n"
        "Return ONLY the rewritten query, nothing else.\n"
        "Return SKIP only if the message is purely conversational with no information need."
    )
