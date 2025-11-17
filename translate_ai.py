from deep_translator import GoogleTranslator
import pandas as pd
import numpy as np
from openai import OpenAI
import os
from dotenv import load_dotenv
import logging
import time
from logging.handlers import RotatingFileHandler

# Configuration du logging
os.makedirs("logs", exist_ok=True)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

log_format = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

file_handler = RotatingFileHandler(
    'logs/translate_ai.log',
    maxBytes=10*1024*1024,
    backupCount=5,
    encoding='utf-8'
)
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(log_format)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(log_format)

logger.addHandler(file_handler)
logger.addHandler(console_handler)

# Chargement des variables d'environnement
load_dotenv()

logger.info("Initialisation du client OpenAI...")
try:
    client = OpenAI(
        # base_url="https://openrouter.ai/api/v1",
        api_key=os.getenv("OPENROUTER_API_KEY"),
    )
    logger.info("✓ Client OpenAI initialisé avec succès")
except Exception as e:
    logger.error(f"✗ Erreur lors de l'initialisation du client OpenAI: {str(e)}", exc_info=True)
    raise


def translate_text(text, source_language="auto", target_language="fr"):
    """Traduit un texte d'une langue à une autre"""
    start_time = time.time()
    text_preview = text[:50] + "..." if len(text) > 50 else text
    
    logger.info(f"Traduction {source_language} → {target_language}")
    logger.info(f"Texte à traduire: {text_preview}")
    
    try:
        translated = GoogleTranslator(source=source_language, target=target_language).translate(text)
        
        duration = time.time() - start_time
        translated_preview = translated[:50] + "..." if len(translated) > 50 else translated
        
        logger.info(f"✓ Traduction réussie en {duration:.2f}s")
        logger.info(f"Texte traduit: {translated_preview}")
        
        return translated
    
    except Exception as e:
        duration = time.time() - start_time
        logger.error(f"✗ Erreur lors de la traduction après {duration:.2f}s: {str(e)}", exc_info=True)
        raise


def ask_ai(text: str):
    """Envoie une requête à l'IA et récupère la réponse"""
    start_time = time.time()
    text_preview = text[:100] + "..." if len(text) > 100 else text
    
    logger.info(f"=" * 60)
    logger.info(f"Envoi de la requête à l'IA")
    logger.info(f"Prompt utilisateur: {text_preview}")
    
    try:
        logger.info("Appel de l'API OpenAI en cours...")
        api_call_start = time.time()
        
        completion = client.chat.completions.create(
            model="gpt-4.1-mini-2025-04-14",
            #reasoning_effort= "none",
            #verbosity = "low",
            messages=[
                {
                    "role": "system",
                    "content": """Tu es Doxa, une intelligence artificielle textuelle et vocale, multilingue et capable de s'exprimer aussi en langues locales.
                    Doxa aide les utilisatrices à réaliser un auto-diagnostic pour mieux comprendre leurs symptômes et identifier d'éventuels troubles.
                    Elle informe, conseille et sensibilise sur la santé mentale des femmes béninoises et africaines, en favorisant la prévention, l'éducation et le bien-être émotionnel. Soit succinct dans tes réponses."""
                },
                {
                    "role": "user",
                    "content": text
                }
            ],
            max_completion_tokens=100
            
        )
        
        api_call_duration = time.time() - api_call_start
        logger.info(f"✓ Réponse de l'API reçue en {api_call_duration:.2f}s")
        
        response_content = completion.choices[0].message.content
        response_preview = response_content[:100] + "..." if len(response_content) > 100 else response_content
        
        # Statistiques de la réponse
        logger.info(f"Modèle utilisé: {completion.model}")
        if hasattr(completion, 'usage'):
            logger.info(f"Tokens utilisés: {completion.usage.total_tokens} (prompt: {completion.usage.prompt_tokens}, completion: {completion.usage.completion_tokens})")
        
        total_duration = time.time() - start_time
        logger.info(f"Réponse de l'IA: {response_preview}")
        logger.info(f"✓ Requête IA complétée en {total_duration:.2f}s")
        logger.info(f"=" * 60)
        
        return response_content
    
    except Exception as e:
        duration = time.time() - start_time
        logger.error(f"✗ Erreur lors de l'appel à l'IA après {duration:.2f}s: {str(e)}", exc_info=True)
        logger.info(f"=" * 60)
        raise


def translate_and_ask_ai(text: str, output_language: str):
    """Pipeline complet: traduction vers français, IA, traduction vers langue cible"""
    start_time = time.time()
    logger.info(f"=" * 80)
    logger.info(f"DÉBUT DU PIPELINE TRADUCTION + IA")
    logger.info(f"Langue de sortie souhaitée: {output_language}")
    
    try:
        # ÉTAPE 1: Traduction vers le français
        step_start = time.time()
        logger.info(f"ÉTAPE 1/3: Traduction vers le français...")
        translated_text = translate_text(text, target_language="fr")
        step_duration = time.time() - step_start
        logger.info(f"✓ ÉTAPE 1 terminée en {step_duration:.2f}s")
        
        # ÉTAPE 2: Génération de la réponse IA
        step_start = time.time()
        logger.info(f"ÉTAPE 2/3: Génération de la réponse IA...")
        answer = ask_ai(translated_text)
        step_duration = time.time() - step_start
        logger.info(f"✓ ÉTAPE 2 terminée en {step_duration:.2f}s")
        
        # ÉTAPE 3: Traduction de la réponse vers la langue cible
        step_start = time.time()
        logger.info(f"ÉTAPE 3/3: Traduction vers {output_language}...")
        final_response = translate_text(answer, source_language="fr", target_language=output_language)
        step_duration = time.time() - step_start
        logger.info(f"✓ ÉTAPE 3 terminée en {step_duration:.2f}s")
        
        total_duration = time.time() - start_time
        logger.info(f"✓✓ PIPELINE TERMINÉ avec succès en {total_duration:.2f}s")
        logger.info(f"=" * 80)
        
        return final_response
    
    except Exception as e:
        duration = time.time() - start_time
        logger.error(f"✗ Erreur dans le pipeline après {duration:.2f}s: {str(e)}", exc_info=True)
        logger.info(f"=" * 80)
        raise