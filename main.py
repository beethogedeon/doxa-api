from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse, Response
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np
from speechbrain.inference import EncoderASR
from doxa_api.asr import transcribe
from transformers import pipeline
import io
import tempfile
import os
from doxa_api.translate_ai import translate_and_ask_ai
from doxa_api.tts import generate_speech

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
    tmp_input_path = None
    tmp_output_path = None
    
    try:
        # Créer un fichier temporaire pour l'input
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_input:
            content = await audio.read()
            tmp_input.write(content)
            tmp_input_path = tmp_input.name
        
        # Créer un fichier temporaire pour l'output
        tmp_output_fd, tmp_output_path = tempfile.mkstemp(suffix=".wav")
        os.close(tmp_output_fd)  # Fermer le descripteur de fichier
        
        # Traitement
        transcription, lang = transcribe(tmp_input_path)
        ai_response = translate_and_ask_ai(transcription, output_language=lang)
        generate_speech(ai_response, lang, tmp_input_path, tmp_output_path)
        
        # Lire le contenu du fichier audio généré
        with open(tmp_output_path, "rb") as f:
            audio_content = f.read()
        
        # Retourner le fichier audio comme réponse
        return Response(
            content=audio_content,
            media_type="audio/wav",
            headers={
                "Content-Disposition": "attachment; filename=response.wav"
            }
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors de l'API: {str(e)}")
    
    finally:
        # Nettoyer les fichiers temporaires
        if tmp_input_path and os.path.exists(tmp_input_path):
            os.unlink(tmp_input_path)
        if tmp_output_path and os.path.exists(tmp_output_path):
            os.unlink(tmp_output_path)

@app.get("/")
async def root():
    return {"message": "Doxa API - Transcription et synthèse vocale"}