from google import genai
from configs import settings

client = genai.Client(api_key=settings["google_gemini_api_key"])

def call_llm(prompt: str) -> str:
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )
    return response.text.strip()
