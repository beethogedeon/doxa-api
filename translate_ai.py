from deep_translator import GoogleTranslator
import pandas as pd
import numpy as np
from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(
  # base_url="https://openrouter.ai/api/v1",
  api_key=os.getenv("OPENROUTER_API_KEY"),
)


def translate_text(text, source_language="auto", target_language="fr"):
    translated = GoogleTranslator(source=source_language, target=target_language).translate(text)
    return translated

def ask_ai(text: str):

    completion = client.chat.completions.create(
    #extra_headers={
    #    "HTTP-Referer": "<YOUR_SITE_URL>", # Optional. Site URL for rankings on openrouter.ai.
    #    "X-Title": "<YOUR_SITE_NAME>", # Optional. Site title for rankings on openrouter.ai.
    #},
    model="gpt-5-mini",
    messages=[
        {
        "role": "system",
        "content": """Tu es Doxa, une intelligence artificielle textuelle et vocale, multilingue et capable de s’exprimer aussi en langues locales.
Doxa aide les utilisatrices à réaliser un auto-diagnostic pour mieux comprendre leurs symptômes et identifier d’éventuels troubles.
Elle informe, conseille et sensibilise sur la santé mentale des femmes béninoises et africaines, en favorisant la prévention, l’éducation et le bien-être émotionnel."""
        },
        {
        "role": "user",
        "content": text
        }
    ]
    )
    return completion.choices[0].message.content


def translate_and_ask_ai(text: str, output_language: str):
    translated_text = translate_text(text)
    answer = ask_ai(translated_text)
    return translate_text(answer, source_language="fr", target_language=output_language)