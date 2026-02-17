def system(context: str = "") -> str:
    prompt = (
        "You are a helpful assistant with access to a knowledge base.\n"
        "Answer the user's question using ONLY the provided context.\n"
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
            "Only include sources you actually used."
            "\n\n"
            f"Relevant context:\n{context}"
        )
    return prompt


def extract_search_query() -> str:
    return (
        "Your task is to extract the search query from the user's question to find relevant information in a vector database. "
        "Rewrite the user's question into a clear, search-optimized query. "
        "Return ONLY the rewritten query, nothing else. If the user's question is already clear and search-optimized, return it as is. If the user's message does not have a question return SKIP." 
    )
