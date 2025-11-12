from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse, Response
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np
from speechbrain.inference import EncoderASR
from .asr import transcribe
from transformers import pipeline
import io
import tempfile
import os
from .translate_ai import translate_and_ask_ai
from .tts import generate_speech
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

@app.post("/transcribe")
async def transcribe_endpoint(audio: UploadFile = File(...)):
    """
    Endpoint pour transcrire un fichier audio.
    Reçoit un fichier audio et retourne la transcription en audio.
    """
    # Timestamp de début
    request_start = time.time()
    request_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{int(time.time() * 1000) % 1000}"
    
    logger.info(f"=" * 80)
    logger.info(f"[REQUEST {request_id}] Nouvelle requête de transcription reçue")
    logger.info(f"[REQUEST {request_id}] Fichier: {audio.filename}, Type: {audio.content_type}")
    
    tmp_input_path = None
    tmp_output_path = None
    
    try:
        # ÉTAPE 1: Sauvegarde du fichier audio
        step_start = time.time()
        logger.info(f"[REQUEST {request_id}] ÉTAPE 1/4: Sauvegarde du fichier audio...")
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_input:
            content = await audio.read()
            file_size = len(content) / 1024  # Taille en KB
            tmp_input.write(content)
            tmp_input_path = tmp_input.name
        
        step_duration = time.time() - step_start
        logger.info(f"[REQUEST {request_id}] ✓ ÉTAPE 1 terminée en {step_duration:.2f}s - Taille: {file_size:.2f} KB")
        logger.info(f"[REQUEST {request_id}] Fichier temporaire créé: {tmp_input_path}")
        
        # ÉTAPE 2: Création du fichier de sortie
        step_start = time.time()
        logger.info(f"[REQUEST {request_id}] ÉTAPE 2/4: Création du fichier de sortie...")
        
        tmp_output_fd, tmp_output_path = tempfile.mkstemp(suffix=".wav")
        os.close(tmp_output_fd)
        
        step_duration = time.time() - step_start
        logger.info(f"[REQUEST {request_id}] ✓ ÉTAPE 2 terminée en {step_duration:.2f}s")
        logger.info(f"[REQUEST {request_id}] Fichier de sortie: {tmp_output_path}")
        
        # ÉTAPE 3: Transcription de l'audio
        step_start = time.time()
        logger.info(f"[REQUEST {request_id}] ÉTAPE 3/4: Transcription de l'audio...")
        
        transcription, lang = transcribe(tmp_input_path)
        
        step_duration = time.time() - step_start
        logger.info(f"[REQUEST {request_id}] ✓ ÉTAPE 3 terminée en {step_duration:.2f}s")
        logger.info(f"[REQUEST {request_id}] Langue détectée: {lang}")
        logger.info(f"[REQUEST {request_id}] Transcription: {transcription[:100]}..." if len(transcription) > 100 else f"[REQUEST {request_id}] Transcription: {transcription}")
        
        # ÉTAPE 3.1: Traduction et réponse IA
        step_start = time.time()
        logger.info(f"[REQUEST {request_id}] ÉTAPE 3.1/4: Traduction et génération de réponse IA...")
        
        ai_response = translate_and_ask_ai(transcription, output_language=lang)
        
        step_duration = time.time() - step_start
        logger.info(f"[REQUEST {request_id}] ✓ ÉTAPE 3.1 terminée en {step_duration:.2f}s")
        logger.info(f"[REQUEST {request_id}] Réponse IA: {ai_response[:100]}..." if len(ai_response) > 100 else f"[REQUEST {request_id}] Réponse IA: {ai_response}")
        
        # ÉTAPE 4: Génération de la synthèse vocale
        step_start = time.time()
        logger.info(f"[REQUEST {request_id}] ÉTAPE 4/4: Génération de la synthèse vocale...")
        
        generate_speech(ai_response, lang, tmp_input_path, tmp_output_path)
        
        step_duration = time.time() - step_start
        logger.info(f"[REQUEST {request_id}] ✓ ÉTAPE 4 terminée en {step_duration:.2f}s")
        
        # Lecture et envoi du fichier audio
        logger.info(f"[REQUEST {request_id}] Lecture du fichier audio généré...")
        with open(tmp_output_path, "rb") as f:
            audio_content = f.read()
        
        output_size = len(audio_content) / 1024  # Taille en KB
        logger.info(f"[REQUEST {request_id}] Taille du fichier de sortie: {output_size:.2f} KB")
        
        # Durée totale
        total_duration = time.time() - request_start
        logger.info(f"[REQUEST {request_id}] ✓✓✓ REQUÊTE TERMINÉE avec succès en {total_duration:.2f}s ✓✓✓")
        logger.info(f"=" * 80)
        
        # Retourner le fichier audio comme réponse
        return Response(
            content=audio_content,
            media_type="audio/wav",
            headers={
                "Content-Disposition": "attachment; filename=response.wav",
                "X-Request-Duration": f"{total_duration:.2f}s"
            }
        )
    
    except Exception as e:
        error_duration = time.time() - request_start
        logger.error(f"[REQUEST {request_id}] ✗✗✗ ERREUR après {error_duration:.2f}s ✗✗✗")
        logger.error(f"[REQUEST {request_id}] Type d'erreur: {type(e).__name__}")
        logger.error(f"[REQUEST {request_id}] Message: {str(e)}", exc_info=True)
        logger.info(f"=" * 80)
        
        raise HTTPException(status_code=500, detail=f"Erreur lors de l'API: {str(e)}")
    
    finally:
        # Nettoyer les fichiers temporaires
        logger.info(f"[REQUEST {request_id}] Nettoyage des fichiers temporaires...")
        if tmp_input_path and os.path.exists(tmp_input_path):
            os.unlink(tmp_input_path)
            logger.info(f"[REQUEST {request_id}] Fichier d'entrée supprimé: {tmp_input_path}")
        if tmp_output_path and os.path.exists(tmp_output_path):
            os.unlink(tmp_output_path)
            logger.info(f"[REQUEST {request_id}] Fichier de sortie supprimé: {tmp_output_path}")

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