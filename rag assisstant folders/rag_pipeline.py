import retriever
import openai

class RAGPipeline:
    def __init__(self, api_key, model_name="gpt-4o-mini", vector_dim=384):
        openai.api_key = api_key
        self.retriever = retriever.Retriever(vector_dim=vector_dim)
        self.model_name = model_name

    def generate_answer(self, question, k=5):
        # Step 1: Retrieve relevant chunks
        results = self.retriever.get_relevant_chunks(question, k=k)
        context = "\n\n".join([chunk for _, chunk in results])

        # Step 2: Build prompt
        prompt = f"""
You are an AI assistant. Answer the question using only the context provided.

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:
"""

        # Step 3: Call OpenAI API
        try:
            response = openai.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
                max_tokens=500
            )
            # Extract text
            answer = response.choices[0].message.content.strip()
        except Exception as e:
            answer = f"Error from OpenAI API: {e}"

        return answer
