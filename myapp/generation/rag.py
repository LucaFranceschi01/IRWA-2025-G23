import os
from groq import Groq
from dotenv import load_dotenv
load_dotenv()  # take environment variables from .env


class RAGGenerator:

    PROMPT = """
        You are an expert product advisor helping users choose the best option from retrieved e-commerce products.

        ## User Query:
        "{user_query}"

        ## Retrieved Products (Top {num_results}):
        {retrieved_results}

        ## Instructions:
        1. Analyze the user's query to understand their intent.
        2. Select the single best recommendation from the list.
        3. Write a natural, conversational response recommending this product. Mention its name, price, and key features that match the query.
        4. Briefly mention 1-2 alternatives if available, comparing them naturally (e.g., "If you prefer a cheaper option...").
        5. Do NOT use bold headers like "Reasoning:", "Top Pick:", or "Alternatives:". Just write in plain paragraphs.
        6. Keep the response concise and helpful.
    """

    def generate_response(self, user_query: str, retrieved_results: list, top_N: int = 5) -> str:
        """
        Generate a response using the retrieved search results. 
        Returns:
            str: The generated suggestion.
        """
        DEFAULT_ANSWER = "RAG is not available. Check your credentials (.env file) or account limits."
        try:
            client = Groq(
                api_key=os.environ.get("GROQ_API_KEY"),
            )
            model_name = os.environ.get("GROQ_MODEL", "llama-3.1-8b-instant")

            # Format the retrieved results for the prompt
            # We need to handle the fact that retrieved_results contains dictionaries, not objects
            formatted_results_list = []
            
            # We limit to top_N (e.g., 5) to avoid exceeding token limits with the extra metadata
            for res in retrieved_results[:top_N]:
                # Safely get values from dict
                pid = res.get('pid', 'N/A')
                title = res.get('title', 'Unknown Title')
                price = res.get('selling_price', 'N/A')
                rating = res.get('average_rating', 'N/A')
                brand = res.get('brand', 'N/A')
                
                # Truncate description to save tokens
                desc = res.get('description', '')
                if desc and len(desc) > 150:
                    desc = desc[:150] + "..."
                
                item_str = (
                    f"- ID: {pid}\n"
                    f"  Name: {title}\n"
                    f"  Brand: {brand}\n"
                    f"  Price: {price}\n"
                    f"  Rating: {rating}\n"
                    f"  Description: {desc}"
                )
                formatted_results_list.append(item_str)

            formatted_results = "\n\n".join(formatted_results_list)

            prompt = self.PROMPT.format(
                retrieved_results=formatted_results,
                user_query=user_query,
                num_results=len(formatted_results_list)
            )

            chat_completion = client.chat.completions.create(
                messages=[
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
                model=model_name,
            )

            generation = chat_completion.choices[0].message.content
            return generation
        except Exception as e:
            print(f"Error during RAG generation: {e}")
            return DEFAULT_ANSWER
