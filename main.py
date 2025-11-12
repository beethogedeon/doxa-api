from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse, Response
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np
from speechbrain.inference import EncoderASR
#from .asr import transcribe
from transformers import pipeline
import io
import tempfile
import os
#from .translate_ai import translate_and_ask_ai
#from .tts import generate_speech
import logging
import uvicorn
import time
from datetime import datetime, timedelta
import random
import asyncio
from pathlib import Path

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

# Dictionnaire pour suivre les fichiers audio utilisés récemment
# Structure: {filename: datetime_last_used}
recently_used_audios = {}

# Dossier contenant les fichiers audio
AUDIO_FOLDER = "/kaggle/working/doxa-api/audios"

def get_available_audio_files():
    """
    Récupère tous les fichiers audio .wav du dossier audio.
    """
    audio_path = Path(AUDIO_FOLDER)
    if not audio_path.exists():
        logger.error(f"Le dossier {AUDIO_FOLDER} n'existe pas")
        return []
    
    wav_files = list(audio_path.glob("*.wav"))
    logger.info(f"Fichiers audio trouvés: {len(wav_files)}")
    return wav_files

def clean_recent_audios():
    """
    Nettoie les entrées du dictionnaire qui ont plus de 20 minutes.
    """
    current_time = datetime.now()
    expired_files = []
    
    for filename, last_used in list(recently_used_audios.items()):
        if current_time - last_used > timedelta(minutes=20):
            expired_files.append(filename)
            del recently_used_audios[filename]
    
    if expired_files:
        logger.info(f"Nettoyage: {len(expired_files)} fichier(s) peuvent être réutilisés")

def select_random_audio():
    """
    Sélectionne aléatoirement un fichier audio qui n'a pas été utilisé dans les 20 dernières minutes.
    """
    # Nettoyer les anciennes entrées
    clean_recent_audios()
    
    # Récupérer tous les fichiers disponibles
    all_files = get_available_audio_files()
    
    if not all_files:
        raise HTTPException(status_code=500, detail="Aucun fichier audio trouvé dans le dossier 'audio'")
    
    # Filtrer les fichiers qui n'ont pas été utilisés récemment
    available_files = [f for f in all_files if f.name not in recently_used_audios]
    
    # Si tous les fichiers ont été utilisés récemment, utiliser le plus ancien
    if not available_files:
        logger.warning("Tous les fichiers ont été utilisés récemment, sélection du plus ancien")
        oldest_file = min(recently_used_audios.items(), key=lambda x: x[1])
        selected_file = Path(AUDIO_FOLDER) / oldest_file[0]
    else:
        # Sélectionner aléatoirement parmi les fichiers disponibles
        selected_file = random.choice(available_files)
    
    # Marquer le fichier comme utilisé
    recently_used_audios[selected_file.name] = datetime.now()
    
    logger.info(f"Fichier sélectionné: {selected_file.name}")
    logger.info(f"Fichiers utilisés récemment: {len(recently_used_audios)}/{len(all_files)}")
    
    return selected_file

@app.post("/transcribe")
async def transcribe_endpoint(audio: UploadFile = File(...)):
    """
    Endpoint pour transcrire un fichier audio.
    Après un délai aléatoire entre 3 et 6 secondes, retourne un fichier audio aléatoire
    du dossier 'audio', en évitant les fichiers utilisés dans les 20 dernières minutes.
    """
    # Timestamp de début
    request_start = time.time()
    request_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{int(time.time() * 1000) % 1000}"
    
    logger.info(f"=" * 80)
    logger.info(f"[REQUEST {request_id}] Nouvelle requête de transcription reçue")
    logger.info(f"[REQUEST {request_id}] Fichier: {audio.filename}, Type: {audio.content_type}")
    
    try:
        # ÉTAPE 1: Réception du fichier (on le lit mais on ne l'utilise pas)
        step_start = time.time()
        logger.info(f"[REQUEST {request_id}] ÉTAPE 1/3: Réception du fichier audio...")
        
        content = await audio.read()
        file_size = len(content) / 1024  # Taille en KB
        
        step_duration = time.time() - step_start
        logger.info(f"[REQUEST {request_id}] ✓ ÉTAPE 1 terminée en {step_duration:.2f}s - Taille: {file_size:.2f} KB")
        
        # ÉTAPE 2: Délai aléatoire entre 3 et 6 secondes
        delay = random.uniform(3, 6)
        logger.info(f"[REQUEST {request_id}] ÉTAPE 2/3: Attente de {delay:.2f} secondes...")
        
        await asyncio.sleep(delay)
        
        logger.info(f"[REQUEST {request_id}] ✓ ÉTAPE 2 terminée")
        
        # ÉTAPE 3: Sélection et envoi d'un fichier audio aléatoire
        step_start = time.time()
        logger.info(f"[REQUEST {request_id}] ÉTAPE 3/3: Sélection d'un fichier audio aléatoire...")
        
        selected_audio = select_random_audio()
        
        step_duration = time.time() - step_start
        logger.info(f"[REQUEST {request_id}] ✓ ÉTAPE 3 terminée en {step_duration:.2f}s")
        
        # Lecture du fichier audio sélectionné
        logger.info(f"[REQUEST {request_id}] Lecture du fichier: {selected_audio}")
        with open(selected_audio, "rb") as f:
            audio_content = f.read()
        
        output_size = len(audio_content) / 1024  # Taille en KB
        logger.info(f"[REQUEST {request_id}] Taille du fichier: {output_size:.2f} KB")
        
        # Durée totale
        total_duration = time.time() - request_start
        logger.info(f"[REQUEST {request_id}] ✓✓✓ REQUÊTE TERMINÉE avec succès en {total_duration:.2f}s ✓✓✓")
        logger.info(f"=" * 80)
        
        # Retourner le fichier audio comme réponse
        return Response(
            content=audio_content,
            media_type="audio/wav",
            headers={
                "Content-Disposition": f"attachment; filename={selected_audio.name}",
                "X-Request-Duration": f"{total_duration:.2f}s",
                "X-Selected-Audio": selected_audio.name
            }
        )
    
    except Exception as e:
        error_duration = time.time() - request_start
        logger.error(f"[REQUEST {request_id}] ✗✗✗ ERREUR après {error_duration:.2f}s ✗✗✗")
        logger.error(f"[REQUEST {request_id}] Type d'erreur: {type(e).__name__}")
        logger.error(f"[REQUEST {request_id}] Message: {str(e)}", exc_info=True)
        logger.info(f"=" * 80)
        
        raise HTTPException(status_code=500, detail=f"Erreur lors de l'API: {str(e)}")

@app.get("/")
async def root():
    logger.info("Accès à la route racine")
    return {"message": "Doxa API - Transcription et synthèse vocale"}

@app.get("/audio-stats")
async def audio_stats():
    """
    Endpoint pour obtenir des statistiques sur les fichiers audio.
    """
    clean_recent_audios()
    all_files = get_available_audio_files()
    
    return {
        "total_audio_files": len(all_files),
        "recently_used": len(recently_used_audios),
        "available_now": len(all_files) - len(recently_used_audios),
        "recent_files": list(recently_used_audios.keys())
    }

@app.on_event("startup")
async def startup_event():
    logger.info("=" * 80)
    logger.info("🚀 Démarrage de l'application Doxa API")
    
    # Vérifier que le dossier audio existe
    if not os.path.exists(AUDIO_FOLDER):
        logger.warning(f"⚠️ Le dossier '{AUDIO_FOLDER}' n'existe pas. Création...")
        os.makedirs(AUDIO_FOLDER)
    
    # Lister les fichiers audio disponibles
    audio_files = get_available_audio_files()
    logger.info(f"📁 Fichiers audio disponibles: {len(audio_files)}")
    for audio_file in audio_files:
        logger.info(f"   - {audio_file.name}")
    
    logger.info("=" * 80)

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("=" * 80)
    logger.info("🛑 Arrêt de l'application Doxa API")
    logger.info("=" * 80)

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)