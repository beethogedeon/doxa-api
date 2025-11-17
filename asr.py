import whisper
from speechbrain.inference import EncoderASR
from torch import cuda
import logging
import time
from logging.handlers import RotatingFileHandler
import os

# Configuration du logging
os.makedirs("logs", exist_ok=True)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

log_format = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

file_handler = RotatingFileHandler(
    'logs/asr.log',
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


def get_speechbrain_model():
    """Charge le modèle SpeechBrain de manière lazy"""
    start_time = time.time()
    logger.info("Chargement du modèle SpeechBrain pour le Fongbe...")
    
    device = "cuda" if cuda.is_available() else "cpu"
    logger.info(f"Device utilisé: {device}")

    try:
        fon_asr_model = EncoderASR.from_hparams(
            source="speechbrain/asr-wav2vec2-dvoice-fongbe",
            savedir="pretrained_models/asr-wav2vec2-dvoice-fongbe",
            run_opts={"device": device}
        )
        
        duration = time.time() - start_time
        logger.info(f"✓ Modèle SpeechBrain chargé avec succès en {duration:.2f}s")
        return fon_asr_model
    
    except Exception as e:
        logger.error(f"✗ Erreur lors du chargement du modèle SpeechBrain: {str(e)}", exc_info=True)
        raise

logger.info("Initialisation des modèles ASR...")
init_start = time.time()

fon_asr_model = get_speechbrain_model()

logger.info("Chargement du modèle Whisper...")
whisper_start = time.time()
device = "cuda" if cuda.is_available() else "cpu"
model = whisper.load_model("small", device=device)
whisper_duration = time.time() - whisper_start
logger.info(f"✓ Modèle Whisper chargé avec succès en {whisper_duration:.2f}s")

total_init_duration = time.time() - init_start
logger.info(f"✓✓ Tous les modèles ASR initialisés en {total_init_duration:.2f}s")


def detect_language(audio_path: str):
    """Détecte la langue de l'audio"""
    start_time = time.time()
    logger.info(f"Détection de la langue pour: {audio_path}")
    
    try:
        # load audio and pad/trim it to fit 30 seconds
        logger.info("Chargement de l'audio...")
        audio = whisper.load_audio(audio_path)
        audio = whisper.pad_or_trim(audio)
        
        logger.info("Création du spectrogramme mel...")
        # make log-Mel spectrogram and move to the same device as the model
        mel = whisper.log_mel_spectrogram(audio, n_mels=model.dims.n_mels).to(model.device)
        
        logger.info("Détection de la langue en cours...")
        # detect the spoken language
        _, probs = model.detect_language(mel)
        detected_lang = max(probs, key=probs.get)
        confidence = probs[detected_lang]

        if confidence < 0.8 :
            raise ValueError("La confiance est trop faible pour détecter la langue")
        
        duration = time.time() - start_time
        logger.info(f"✓ Langue détectée: {detected_lang} (confiance: {confidence:.2%}) en {duration:.2f}s")
        
        # Afficher les 3 langues les plus probables
        top_3 = sorted(probs.items(), key=lambda x: x[1], reverse=True)[:3]
        logger.info(f"Top 3 des langues détectées: {[(lang, f'{prob:.2%}') for lang, prob in top_3]}")
        
        return detected_lang, mel
    
    except Exception as e:
        duration = time.time() - start_time
        logger.error(f"✗ Erreur lors de la détection de langue après {duration:.2f}s: {str(e)}", exc_info=True)
        raise


def transcribe(audio_path: str, lang: str):
    """Transcrit l'audio en fonction de la langue détectée"""
    start_time = time.time()
    logger.info(f"=" * 60)
    logger.info(f"Début de la transcription: {audio_path}")
    
    try:
        # Détection de la langue
        if lang is None:
            lang, mel = detect_language(audio_path)
        else:
            mel = None
        
        if lang == "yo":
            logger.info("Utilisation de Whisper pour le Yoruba...")
            transcribe_start = time.time()
            
            # decode the audio
            options = whisper.DecodingOptions()
            result = whisper.decode(model, mel, options)
            
            transcribe_duration = time.time() - transcribe_start
            total_duration = time.time() - start_time
            
            logger.info(f"✓ Transcription Yoruba terminée en {transcribe_duration:.2f}s")
            logger.info(f"Résultat (Yoruba): {result.text[:100]}..." if len(result.text) > 100 else f"Résultat (Yoruba): {result.text}")
            logger.info(f"✓ Transcription totale terminée en {total_duration:.2f}s")
            logger.info(f"=" * 60)
            
            return result.text, lang
        
        else:
            logger.info(f"Langue détectée ({lang}) non supportée par Whisper, utilisation de SpeechBrain pour le Fongbe...")
            transcribe_start = time.time()
            
            transcription = fon_asr_model.transcribe_file(audio_path)
            
            transcribe_duration = time.time() - transcribe_start
            total_duration = time.time() - start_time
            
            logger.info(f"✓ Transcription Fongbe terminée en {transcribe_duration:.2f}s")
            logger.info(f"Résultat (Fongbe): {transcription[:100]}..." if len(transcription) > 100 else f"Résultat (Fongbe): {transcription}")
            logger.info(f"✓ Transcription totale terminée en {total_duration:.2f}s")
            logger.info(f"=" * 60)
            
            return transcription, "fon"
    
    except Exception as e:
        duration = time.time() - start_time
        logger.error(f"✗ Erreur lors de la transcription après {duration:.2f}s: {str(e)}", exc_info=True)
        logger.info(f"=" * 60)
        raise