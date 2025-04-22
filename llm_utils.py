def generate_response(query, search_results):
    try:
        from llama_cpp import Llama
        import re

        model_path = r"C:\Users\navak\Downloads\WebSearchChatbot\WebSearchChatbot\models\capybarahermes-2.5-mistral-7b.Q5_K_S.gguf"

        if not search_results:
            return "I couldn't find any relevant information."

        context = "\n".join(f"{i+1}. {r['snippet']}" for i, r in enumerate(search_results[:5]))
        prompt = (
            f"You are a helpful assistant. Use the following search results to answer the user's question:\n\n"
            f"{context}\n\n"
            f"Question: {query}\nAnswer:"
        )

        llm = Llama(model_path=model_path, n_ctx=2048, verbose=False)
        output = llm(prompt, max_tokens=256, stop=["Question:", "Answer:"])
        text = output["choices"][0]["text"].strip()

        # Clean repeated lines and trim
        lines = text.splitlines()
        seen = set()
        cleaned = []
        for line in lines:
            if line not in seen:
                cleaned.append(line)
                seen.add(line)

        return "\n".join(cleaned).strip()

    except Exception as e:
        return f"⚠️ Error: {str(e)}"
