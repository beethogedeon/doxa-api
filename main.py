from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import Response
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np
from speechbrain.inference import EncoderASR
from transformers import pipeline
import io
import tempfile
import os

app = FastAPI(title="Doxa API", description="API pour transcription et synthèse vocale")

# Configuration CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # En production, remplacer par des origines spécifiques
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialisation des modèles (lazy loading)
speechbrain_model = None
tts_pipe_fon = None

def get_speechbrain_model():
    """Charge le modèle SpeechBrain de manière lazy"""
    global speechbrain_model
    if speechbrain_model is None:
        speechbrain_model = EncoderASR.from_hparams(
            source="speechbrain/asr-wav2vec2-dvoice-fongbe",
            savedir="pretrained_models/asr-wav2vec2-dvoice-fongbe",
            run_opts={"device": "cuda:0"}
        )
    return speechbrain_model

def get_tts_pipe_fon():
    """Charge le pipeline TTS pour le fon de manière lazy"""
    global tts_pipe_fon
    if tts_pipe_fon is None:
        tts_pipe_fon = pipeline(
            "text-to-speech",
            model="facebook/mms-tts-fon",
            device_map="auto",
            framework="pt"
        )
    return tts_pipe_fon

speechbrain_model = get_speechbrain_model()

@app.post("/transcribe")
async def transcribe(audio: UploadFile = File(...)):
    """
    Endpoint pour transcrire un fichier audio.
    Reçoit un fichier audio et retourne la transcription.
    """
    try:
        # Sauvegarder temporairement le fichier audio
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
            content = await audio.read()
            tmp_file.write(content)
            tmp_path = tmp_file.name
        
        try:
            transcription = speechbrain_model.transcribe_file(tmp_path)
            
            return {"transcription": transcription}
        finally:
            # Nettoyer le fichier temporaire
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors de la transcription: {str(e)}")

@app.post("/generate_speech")
async def generate_speech(text: str, lang: str = "fon"):
    """
    Endpoint pour générer de la parole à partir de texte.
    
    Parameters:
    - text: Le texte à synthétiser
    - lang: La langue de destination ("fon" pour Fongbe, "yor" pour Yoruba)
    """
    if lang != "fon":
        raise HTTPException(
            status_code=400,
            detail=f"Langue '{lang}' non supportée pour le moment. Seul 'fon' est disponible."
        )
    
    try:
        if lang == "fon":
            pipe = get_tts_pipe_fon()
            audio_output = pipe(text)
            
            # Extraire l'audio et normaliser
            audio_array = audio_output["audio"]
            audio_array = audio_array.squeeze()
            
            # Normaliser l'audio si nécessaire
            if audio_array.dtype != np.int16:
                # Si l'audio est en float32, normaliser entre -1 et 1 puis convertir en int16
                if audio_array.dtype == np.float32 or audio_array.dtype == np.float64:
                    # Normaliser vers [-1, 1] si nécessaire
                    max_val = np.abs(audio_array).max()
                    if max_val > 1.0:
                        audio_array = audio_array / max_val
                    
                    # Convertir en int16 (plage -32768 à 32767)
                    audio_array = (audio_array * 32767).astype(np.int16)
                else:
                    audio_array = audio_array.astype(np.int16)
            
            # Convertir en bytes
            audio_bytes = audio_array.tobytes()
            
            # Retourner l'audio en tant que réponse binaire
            return Response(
                content=audio_bytes,
                media_type="audio/wav",
                headers={
                    "Content-Disposition": f"attachment; filename=speech_{lang}.wav"
                }
            )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors de la génération vocale: {str(e)}")

@app.get("/")
async def root():
    return {"message": "Doxa API - Transcription et synthèse vocale"}

