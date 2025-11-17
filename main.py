from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse, Response, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np
from speechbrain.inference import EncoderASR
from .asr import transcribe
from transformers import pipeline
import io
import tempfile
import os
from .translate_ai import translate_and_ask_ai, translate_and_ask_ai_stream
from .tts import generate_speech, generate_speech_stream
import logging
import uvicorn
import time
from datetime import datetime

# Configuration du logging avec rotation de fichiers
from logging.handlers import RotatingFileHandler

# Créer le dossier de logs s'il n'existe pas
os.makedirs("logs", exist_ok=True)

# Configuration du logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Format détaillé pour les logs
log_format = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# Handler pour le fichier avec rotation (max 10MB, garde 5 fichiers)
file_handler = RotatingFileHandler(
    'logs/doxa_api.log',
    maxBytes=10*1024*1024,  # 10MB
    backupCount=5,
    encoding='utf-8'
)
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(log_format)

# Handler pour la console
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(log_format)

# Ajouter les handlers
logger.addHandler(file_handler)
logger.addHandler(console_handler)

app = FastAPI(title="Doxa API", description="API pour transcription et synthèse vocale")

# Configuration CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # En production, remplacer par des origines spécifiques
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def stream_transcribe_pipeline(audio_path: str, lang: str, request_id: str):
    """Pipeline de streaming complet: transcription -> traduction+IA -> TTS -> audio"""
    try:
        # ÉTAPE 1: Transcription de l'audio (non-streaming, nécessaire avant)
        step_start = time.time()
        logger.info(f"[REQUEST {request_id}] ÉTAPE 1/3: Transcription de l'audio...")
        
        transcription, detected_lang = transcribe(audio_path, lang)
        if lang is None:
            lang = detected_lang
        
        step_duration = time.time() - step_start
        logger.info(f"[REQUEST {request_id}] ✓ ÉTAPE 1 terminée en {step_duration:.2f}s")
        logger.info(f"[REQUEST {request_id}] Langue détectée: {lang}")
        logger.info(f"[REQUEST {request_id}] Transcription: {transcription[:100]}..." if len(transcription) > 100 else f"[REQUEST {request_id}] Transcription: {transcription}")
        
        # ÉTAPE 2: Traduction et réponse IA en streaming
        step_start = time.time()
        logger.info(f"[REQUEST {request_id}] ÉTAPE 2/3: Traduction et génération de réponse IA (streaming)...")
        
        # Obtenir le stream de texte traduit
        text_stream = translate_and_ask_ai_stream(transcription, output_language=lang)
        
        # ÉTAPE 3: Génération de la synthèse vocale en streaming
        logger.info(f"[REQUEST {request_id}] ÉTAPE 3/3: Génération de la synthèse vocale (streaming)...")
        
        # Streamer l'audio directement depuis le stream de texte
        audio_stream = generate_speech_stream(text_stream, lang, audio_path)
        
        step_duration = time.time() - step_start
        logger.info(f"[REQUEST {request_id}] ✓ ÉTAPE 2-3 terminées en {step_duration:.2f}s")
        
        # Yielder les chunks audio
        for audio_chunk in audio_stream:
            yield audio_chunk
        
        total_duration = time.time() - step_start
        logger.info(f"[REQUEST {request_id}] ✓✓✓ STREAMING TERMINÉ avec succès en {total_duration:.2f}s ✓✓✓")
        logger.info(f"=" * 80)
    
    except Exception as e:
        logger.error(f"[REQUEST {request_id}] ✗✗✗ ERREUR dans le pipeline streaming ✗✗✗")
        logger.error(f"[REQUEST {request_id}] Type d'erreur: {type(e).__name__}")
        logger.error(f"[REQUEST {request_id}] Message: {str(e)}", exc_info=True)
        logger.info(f"=" * 80)
        raise


@app.post("/transcribe")
async def transcribe_endpoint(audio: UploadFile = File(...), lang:str = None):
    """
    Endpoint pour transcrire un fichier audio.
    Reçoit un fichier audio et retourne la transcription en audio en streaming.
    """
    # Timestamp de début
    request_start = time.time()
    request_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{int(time.time() * 1000) % 1000}"
    
    logger.info(f"=" * 80)
    logger.info(f"[REQUEST {request_id}] Nouvelle requête de transcription (STREAMING) reçue")
    logger.info(f"[REQUEST {request_id}] Fichier: {audio.filename}, Type: {audio.content_type}")
    
    tmp_input_path = None
    
    try:
        # ÉTAPE 1: Sauvegarde du fichier audio
        step_start = time.time()
        logger.info(f"[REQUEST {request_id}] ÉTAPE 0/3: Sauvegarde du fichier audio...")
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_input:
            content = await audio.read()
            file_size = len(content) / 1024  # Taille en KB
            tmp_input.write(content)
            tmp_input_path = tmp_input.name
        
        step_duration = time.time() - step_start
        logger.info(f"[REQUEST {request_id}] ✓ ÉTAPE 0 terminée en {step_duration:.2f}s - Taille: {file_size:.2f} KB")
        logger.info(f"[REQUEST {request_id}] Fichier temporaire créé: {tmp_input_path}")
        
        # Créer le générateur de streaming
        def generate():
            try:
                for chunk in stream_transcribe_pipeline(tmp_input_path, lang, request_id):
                    yield chunk
            finally:
                # Nettoyer le fichier temporaire
                if tmp_input_path and os.path.exists(tmp_input_path):
                    os.unlink(tmp_input_path)
                    logger.info(f"[REQUEST {request_id}] Fichier d'entrée supprimé: {tmp_input_path}")
        
        # Durée totale
        total_duration = time.time() - request_start
        logger.info(f"[REQUEST {request_id}] Début du streaming après {total_duration:.2f}s")
        
        # Retourner le stream audio
        return StreamingResponse(
            generate(),
            media_type="audio/wav",
            headers={
                "Content-Type": "audio/wav",
                "X-Request-ID": request_id,
                "X-Request-Duration": f"{total_duration:.2f}s"
            }
        )
    
    except Exception as e:
        error_duration = time.time() - request_start
        logger.error(f"[REQUEST {request_id}] ✗✗✗ ERREUR après {error_duration:.2f}s ✗✗✗")
        logger.error(f"[REQUEST {request_id}] Type d'erreur: {type(e).__name__}")
        logger.error(f"[REQUEST {request_id}] Message: {str(e)}", exc_info=True)
        logger.info(f"=" * 80)
        
        # Nettoyer en cas d'erreur
        if tmp_input_path and os.path.exists(tmp_input_path):
            os.unlink(tmp_input_path)
        
        raise HTTPException(status_code=500, detail=f"Erreur lors de l'API: {str(e)}")

@app.get("/")
async def root():
    logger.info("Accès à la route racine")
    return {"message": "Doxa API - Transcription et synthèse vocale"}

@app.on_event("startup")
async def startup_event():
    logger.info("=" * 80)
    logger.info("🚀 Démarrage de l'application Doxa API")
    logger.info("=" * 80)

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("=" * 80)
    logger.info("🛑 Arrêt de l'application Doxa API")
    logger.info("=" * 80)

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)