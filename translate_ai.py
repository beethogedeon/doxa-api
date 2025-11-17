from deep_translator import GoogleTranslator
import pandas as pd
import numpy as np
from openai import OpenAI
import os
from dotenv import load_dotenv
import logging
import time
from logging.handlers import RotatingFileHandler
import re
from typing import Generator

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
        
        completion = client.responses.create(
            model="gpt-4.1-mini",
            #reasoning_effort= "none",
            #verbosity = "low",
            instructions="""Tu es Doxa, une intelligence artificielle textuelle et vocale, multilingue et capable de s'exprimer aussi en langues locales.
                    Doxa aide les utilisatrices à réaliser un auto-diagnostic pour mieux comprendre leurs symptômes et identifier d'éventuels troubles.
                    Elle informe, conseille et sensibilise sur la santé mentale des femmes béninoises et africaines, en favorisant la prévention, l'éducation et le bien-être émotionnel. Soit succinct dans tes réponses.""",
            input=text,
            max_outpu_tokens=100
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


def split_text_into_sentences(text: str) -> list:
    """Divise un texte en phrases pour traitement par chunks"""
    # Diviser par points, points d'exclamation, points d'interrogation
    sentences = re.split(r'([.!?]+)', text)
    # Recombiner les phrases avec leur ponctuation
    result = []
    for i in range(0, len(sentences) - 1, 2):
        if i + 1 < len(sentences):
            result.append(sentences[i] + sentences[i + 1])
        else:
            result.append(sentences[i])
    # Filtrer les phrases vides
    return [s.strip() for s in result if s.strip()]


def translate_text_stream(text: str, source_language: str = "auto", target_language: str = "fr") -> Generator[str, None, None]:
    """Traduit un texte par chunks (phrases) en streaming"""
    start_time = time.time()
    logger.info(f"Traduction streaming {source_language} → {target_language}")
    
    try:
        # Diviser le texte en phrases
        sentences = split_text_into_sentences(text)
        logger.info(f"Texte divisé en {len(sentences)} phrases")
        
        translator = GoogleTranslator(source=source_language, target=target_language)
        
        for sentence in sentences:
            if sentence:
                translated = translator.translate(sentence)
                yield translated + " "
        
        duration = time.time() - start_time
        logger.info(f"✓ Traduction streaming réussie en {duration:.2f}s")
    
    except Exception as e:
        duration = time.time() - start_time
        logger.error(f"✗ Erreur lors de la traduction streaming après {duration:.2f}s: {str(e)}", exc_info=True)
        raise


def ask_ai_stream(text: str) -> Generator[str, None, None]:
    """Envoie une requête à l'IA et stream la réponse"""
    start_time = time.time()
    text_preview = text[:100] + "..." if len(text) > 100 else text
    
    logger.info(f"=" * 60)
    logger.info(f"Envoi de la requête à l'IA (streaming)")
    logger.info(f"Prompt utilisateur: {text_preview}")
    
    try:
        logger.info("Appel de l'API OpenAI en streaming...")
        api_call_start = time.time()
        
        stream = client.responses.create(
            model="gpt-4.1-mini",
            instructions="""Tu es Doxa, une intelligence artificielle textuelle et vocale, multilingue et capable de s'exprimer aussi en langues locales.
                    Doxa aide les utilisatrices à réaliser un auto-diagnostic pour mieux comprendre leurs symptômes et identifier d'éventuels troubles.
                    Elle informe, conseille et sensibilise sur la santé mentale des femmes béninoises et africaines, en favorisant la prévention, l'éducation et le bien-être émotionnel. Soit succinct dans tes réponses.""",
            input=text,
            max_output_tokens=100,
            stream=True
        )
        
        accumulated_text = ""
        for chunk in stream:
            # Gérer différentes structures de réponse streaming
            try:
                # Structure standard OpenAI streaming
                if hasattr(chunk, 'choices') and chunk.choices:
                    delta = chunk.choices[0].delta
                    if hasattr(delta, 'content') and delta.content:
                        content = delta.content
                        accumulated_text += content
                        yield content
                # Structure alternative (si l'API responses a une structure différente)
                elif hasattr(chunk, 'content') and chunk.content:
                    content = chunk.content
                    accumulated_text += content
                    yield content
                # Structure avec message direct
                elif hasattr(chunk, 'message') and hasattr(chunk.message, 'content') and chunk.message.content:
                    content = chunk.message.content
                    accumulated_text += content
                    yield content
            except Exception as e:
                logger.warning(f"Erreur lors du traitement d'un chunk: {str(e)}")
                continue
        
        api_call_duration = time.time() - api_call_start
        logger.info(f"✓ Réponse de l'API streaming reçue en {api_call_duration:.2f}s")
        
        total_duration = time.time() - start_time
        logger.info(f"Réponse de l'IA (streaming): {accumulated_text[:100]}..." if len(accumulated_text) > 100 else f"Réponse de l'IA (streaming): {accumulated_text}")
        logger.info(f"✓ Requête IA streaming complétée en {total_duration:.2f}s")
        logger.info(f"=" * 60)
    
    except Exception as e:
        duration = time.time() - start_time
        logger.error(f"✗ Erreur lors de l'appel à l'IA streaming après {duration:.2f}s: {str(e)}", exc_info=True)
        logger.info(f"=" * 60)
        raise


def translate_and_ask_ai_stream(text: str, output_language: str) -> Generator[str, None, None]:
    """Pipeline complet en streaming: traduction vers français, IA, traduction vers langue cible"""
    start_time = time.time()
    logger.info(f"=" * 80)
    logger.info(f"DÉBUT DU PIPELINE TRADUCTION + IA (STREAMING)")
    logger.info(f"Langue de sortie souhaitée: {output_language}")
    
    try:
        # ÉTAPE 1: Traduction vers le français (non-streaming car nécessaire avant l'IA)
        step_start = time.time()
        logger.info(f"ÉTAPE 1/3: Traduction vers le français...")
        translated_text = translate_text(text, target_language="fr")
        step_duration = time.time() - step_start
        logger.info(f"✓ ÉTAPE 1 terminée en {step_duration:.2f}s")
        
        # ÉTAPE 2: Génération de la réponse IA en streaming
        step_start = time.time()
        logger.info(f"ÉTAPE 2/3: Génération de la réponse IA (streaming)...")
        
        # Accumuler la réponse IA pour la traduction
        ai_response_chunks = []
        for chunk in ask_ai_stream(translated_text):
            ai_response_chunks.append(chunk)
            # Streamer directement les chunks traduits
            # Mais on doit d'abord accumuler pour traduire correctement
            pass
        
        # Reconstruire la réponse complète
        full_ai_response = "".join(ai_response_chunks)
        step_duration = time.time() - step_start
        logger.info(f"✓ ÉTAPE 2 terminée en {step_duration:.2f}s")
        
        # ÉTAPE 3: Traduction de la réponse vers la langue cible en streaming
        step_start = time.time()
        logger.info(f"ÉTAPE 3/3: Traduction vers {output_language} (streaming)...")
        logger.info(f"Réponse IA complète à traduire: {full_ai_response[:100]}..." if len(full_ai_response) > 100 else f"Réponse IA complète: {full_ai_response}")
        
        if not full_ai_response or not full_ai_response.strip():
            logger.warning("Réponse IA vide, rien à traduire!")
            return
        
        translated_chunks_count = 0
        for translated_chunk in translate_text_stream(full_ai_response, source_language="fr", target_language=output_language):
            if translated_chunk and translated_chunk.strip():
                translated_chunks_count += 1
                logger.debug(f"Chunk traduit {translated_chunks_count}: {translated_chunk[:50]}...")
                yield translated_chunk
        
        step_duration = time.time() - step_start
        logger.info(f"✓ ÉTAPE 3 terminée en {step_duration:.2f}s ({translated_chunks_count} chunks traduits)")
        
        if translated_chunks_count == 0:
            logger.warning("Aucun chunk traduit généré!")
        
        total_duration = time.time() - start_time
        logger.info(f"✓✓ PIPELINE STREAMING TERMINÉ avec succès en {total_duration:.2f}s")
        logger.info(f"=" * 80)
    
    except Exception as e:
        duration = time.time() - start_time
        logger.error(f"✗ Erreur dans le pipeline streaming après {duration:.2f}s: {str(e)}", exc_info=True)
        logger.info(f"=" * 80)
        raise