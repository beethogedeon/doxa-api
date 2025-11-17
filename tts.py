import tensorflow
import torch
import numpy as np
import logging
import soundfile as sf
import os
from openvoice import se_extractor
from openvoice.api import ToneColorConverter
import time
from logging.handlers import RotatingFileHandler
from typing import Generator
import tempfile
import io
import re

# Configuration du logging
os.makedirs("logs", exist_ok=True)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

log_format = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

file_handler = RotatingFileHandler(
    'logs/tts.log',
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


from transformers import VitsModel, AutoTokenizer, set_seed

set_seed(42)


logger.info("=" * 80)
logger.info("INITIALISATION DES MODÈLES TTS")
logger.info("=" * 80)

# Chargement du modèle Fongbe
logger.info("Chargement du modèle TTS Fongbe...")
fon_start = time.time()
try:
    fon_tts = VitsModel.from_pretrained("facebook/mms-tts-fon", device_map="auto", dtype=torch.float16)
    fon_tokenizer = AutoTokenizer.from_pretrained("facebook/mms-tts-fon")
    fon_duration = time.time() - fon_start
    logger.info(f"✓ Modèle TTS Fongbe chargé avec succès en {fon_duration:.2f}s")
except Exception as e:
    logger.error(f"✗ Erreur lors du chargement du modèle Fongbe: {str(e)}", exc_info=True)
    raise

# Chargement du modèle Yoruba
logger.info("Chargement du modèle TTS Yoruba...")
yor_start = time.time()
try:
    yor_tts = VitsModel.from_pretrained("facebook/mms-tts-yor",device_map="auto", dtype=torch.float16)
    yor_tokenizer = AutoTokenizer.from_pretrained("facebook/mms-tts-yor")
    yor_duration = time.time() - yor_start
    logger.info(f"✓ Modèle TTS Yoruba chargé avec succès en {yor_duration:.2f}s")
except Exception as e:
    logger.error(f"✗ Erreur lors du chargement du modèle Yoruba: {str(e)}", exc_info=True)
    raise

# Configuration du convertisseur de ton
ckpt_converter = './checkpoints_v2/converter'
device = "cuda" if torch.cuda.is_available() else "cpu"
output_dir = 'outputs_v2'

logger.info(f"Device utilisé: {device}")
logger.info("Chargement du convertisseur de couleur de ton...")
converter_start = time.time()
try:
    tone_color_converter = ToneColorConverter(f'{ckpt_converter}/config.json', device=device)
    tone_color_converter.load_ckpt(f'{ckpt_converter}/checkpoint.pth')
    converter_duration = time.time() - converter_start
    logger.info(f"✓ Convertisseur chargé avec succès en {converter_duration:.2f}s")
except Exception as e:
    logger.error(f"✗ Erreur lors du chargement du convertisseur: {str(e)}", exc_info=True)
    raise

# Extraction du speaker de référence
reference_speaker = './consolas_voice.wav'
logger.info(f"Extraction du speaker de référence: {reference_speaker}")
ref_start = time.time()
try:
    target_se, audio_name = se_extractor.get_se(reference_speaker, tone_color_converter, vad=True)
    ref_duration = time.time() - ref_start
    logger.info(f"✓ Speaker extrait avec succès en {ref_duration:.2f}s")
except Exception as e:
    logger.error(f"✗ Erreur lors de l'extraction du speaker: {str(e)}", exc_info=True)
    raise

os.makedirs(output_dir, exist_ok=True)

total_init = fon_duration + yor_duration + converter_duration + ref_duration
logger.info(f"✓✓ TOUS LES MODÈLES TTS INITIALISÉS en {total_init:.2f}s")
logger.info("=" * 80)


def clone_voice(input_audio_path: str, output_audio_path: str):
    """Clone la voix de l'entrée vers la sortie"""
    start_time = time.time()
    logger.info(f"Clonage de voix: {input_audio_path} → {output_audio_path}")
    
    try:
        logger.info("Extraction des caractéristiques de la source...")
        extract_start = time.time()
        source_se = tone_color_converter.extract_se(input_audio_path)
        extract_duration = time.time() - extract_start
        logger.info(f"✓ Extraction terminée en {extract_duration:.2f}s")
        
        logger.info("Conversion de la couleur de ton...")
        convert_start = time.time()
        tone_color_converter.convert(
            audio_src_path=input_audio_path,
            src_se=source_se,
            tgt_se=target_se,
            output_path=output_audio_path,
            message="@Doxa AI"
        )
        convert_duration = time.time() - convert_start
        logger.info(f"✓ Conversion terminée en {convert_duration:.2f}s")
        
        total_duration = time.time() - start_time
        logger.info(f"✓ Clonage de voix réussi en {total_duration:.2f}s")
        return True
    
    except Exception as e:
        duration = time.time() - start_time
        logger.error(f"✗ Erreur lors du clonage de voix après {duration:.2f}s: {str(e)}", exc_info=True)
        return False


def pad_array(array, sr):
    """Pad l'array audio pour supprimer les silences"""
    logger.debug("Padding de l'array audio...")
    
    if isinstance(array, list):
        array = np.array(array)

    if not array.shape[0]:
        logger.error("L'audio généré ne contient aucune donnée")
        raise ValueError("The generated audio does not contain any data")

    valid_indices = np.where(np.abs(array) > 0.001)[0]

    if len(valid_indices) == 0:
        logger.warning(f"Aucun indice valide trouvé dans l'array")
        return array

    try:
        pad_indice = int(0.1 * sr)
        start_pad = max(0, valid_indices[0] - pad_indice)
        end_pad = min(len(array), valid_indices[-1] + 1 + pad_indice)
        padded_array = array[start_pad:end_pad]
        
        logger.debug(f"Array paddé: {len(array)} → {len(padded_array)} échantillons")
        return padded_array
    
    except Exception as error:
        logger.error(f"Erreur lors du padding: {str(error)}")
        return array


def write_chunked(file, data, samplerate, subtype=None, endian=None, format=None, closefd=True):
    """Écrit l'audio dans un fichier"""
    logger.debug(f"Écriture de l'audio dans: {file}")
    
    data = np.asarray(data)
    if data.ndim == 1:
        channels = 1
    else:
        channels = data.shape[1]

    logger.debug(f"Paramètres: {samplerate}Hz, {channels} canal(aux), {len(data)} échantillons")

    with sf.SoundFile(
        file, 'w', samplerate, channels,
        subtype, endian, format, closefd
    ) as f:
        f.write(data)
    
    logger.debug(f"✓ Audio écrit avec succès")


def generate_speech(text: str, lang: str, input_audio_path: str, output_audio_path: str):
    """Génère la synthèse vocale à partir du texte"""
    start_time = time.time()
    logger.info(f"=" * 80)
    logger.info(f"GÉNÉRATION DE SYNTHÈSE VOCALE")
    logger.info(f"Langue: {lang}")
    logger.info(f"Texte: {text[:100]}..." if len(text) > 100 else f"Texte: {text}")
    logger.info(f"Input: {input_audio_path}")
    logger.info(f"Output: {output_audio_path}")
    
    try:
        if lang == "fon":
            logger.info("Utilisation du modèle TTS Fongbe...")
            
            # Tokenization
            token_start = time.time()
            logger.info("Tokenization du texte...")
            logger.info(text)
            inputs = fon_tokenizer(text, return_tensors="pt").to(fon_tts.device)
            sampling_rate = fon_tts.config.sampling_rate
            token_duration = time.time() - token_start
            logger.info(f"✓ Tokenization terminée en {token_duration:.2f}s (sample rate: {sampling_rate}Hz)")
            
            # Génération de la parole
            gen_start = time.time()
            logger.info("Génération de la forme d'onde...")
            with torch.no_grad():
                speech_output = fon_tts(**inputs).waveform
            gen_duration = time.time() - gen_start
            logger.info(f"✓ Génération terminée en {gen_duration:.2f}s")
            
            # Padding
            pad_start = time.time()
            logger.info("Padding de l'audio...")
            data_tts = pad_array(
                speech_output.cpu().numpy().squeeze().astype(np.float32),
                sampling_rate,
            )
            pad_duration = time.time() - pad_start
            logger.info(f"✓ Padding terminé en {pad_duration:.2f}s")
            
            # Écriture du fichier
            write_start = time.time()
            logger.info("Écriture du fichier audio...")
            write_chunked(
                file=output_audio_path,
                samplerate=sampling_rate,
                data=data_tts,
                format="wav",
            )
            write_duration = time.time() - write_start
            logger.info(f"✓ Fichier écrit en {write_duration:.2f}s")
            
            # Clonage de voix
            clone_start = time.time()
            logger.info("Clonage de la voix...")
            clone_success = clone_voice(output_audio_path, output_audio_path)
            clone_duration = time.time() - clone_start
            
            if clone_success:
                logger.info(f"✓ Clonage réussi en {clone_duration:.2f}s")
            else:
                logger.warning(f"⚠ Clonage échoué après {clone_duration:.2f}s")
            
            total_duration = time.time() - start_time
            logger.info(f"✓✓ SYNTHÈSE VOCALE FONGBE TERMINÉE en {total_duration:.2f}s")
            logger.info(f"Détails: token={token_duration:.2f}s, gen={gen_duration:.2f}s, pad={pad_duration:.2f}s, write={write_duration:.2f}s, clone={clone_duration:.2f}s")
            logger.info(f"=" * 80)
            
            return output_audio_path
        
        elif lang == "yor":
            logger.info("Utilisation du modèle TTS Yoruba...")
            
            # Tokenization
            token_start = time.time()
            logger.info("Tokenization du texte...")
            inputs = yor_tokenizer(text, return_tensors="pt").to(yor_tts.device)
            sampling_rate = yor_tts.config.sampling_rate
            token_duration = time.time() - token_start
            logger.info(f"✓ Tokenization terminée en {token_duration:.2f}s (sample rate: {sampling_rate}Hz)")
            
            # Génération de la parole
            gen_start = time.time()
            logger.info("Génération de la forme d'onde...")
            with torch.no_grad():
                speech_output = yor_tts(**inputs).waveform
            gen_duration = time.time() - gen_start
            logger.info(f"✓ Génération terminée en {gen_duration:.2f}s")
            
            # Padding
            pad_start = time.time()
            logger.info("Padding de l'audio...")
            data_tts = pad_array(
                speech_output.cpu().numpy().squeeze().astype(np.float32),
                sampling_rate,
            )
            pad_duration = time.time() - pad_start
            logger.info(f"✓ Padding terminé en {pad_duration:.2f}s")
            
            # Écriture du fichier
            write_start = time.time()
            logger.info("Écriture du fichier audio...")
            write_chunked(
                file=output_audio_path,
                samplerate=sampling_rate,
                data=data_tts,
                format="wav",
            )
            write_duration = time.time() - write_start
            logger.info(f"✓ Fichier écrit en {write_duration:.2f}s")
            
            # Clonage de voix
            clone_start = time.time()
            logger.info("Clonage de la voix...")
            clone_success = clone_voice(input_audio_path, output_audio_path)
            clone_duration = time.time() - clone_start
            
            if clone_success:
                logger.info(f"✓ Clonage réussi en {clone_duration:.2f}s")
            else:
                logger.warning(f"⚠ Clonage échoué après {clone_duration:.2f}s")
            
            total_duration = time.time() - start_time
            logger.info(f"✓✓ SYNTHÈSE VOCALE YORUBA TERMINÉE en {total_duration:.2f}s")
            logger.info(f"Détails: token={token_duration:.2f}s, gen={gen_duration:.2f}s, pad={pad_duration:.2f}s, write={write_duration:.2f}s, clone={clone_duration:.2f}s")
            logger.info(f"=" * 80)
            
            return output_audio_path
        
        else:
            logger.error(f"✗ Langue non supportée: {lang}")
            raise ValueError(f"Langue non supportée: {lang}")
    
    except Exception as e:
        duration = time.time() - start_time
        logger.error(f"✗ Erreur lors de la génération vocale après {duration:.2f}s: {str(e)}", exc_info=True)
        logger.info(f"=" * 80)
        raise


def split_text_into_chunks(text: str, max_chunk_length: int = 50) -> list:
    """Divise un texte en chunks pour traitement TTS"""
    # Diviser par phrases d'abord
    sentences = re.split(r'([.!?]+)', text)
    result = []
    current_chunk = ""
    
    for i in range(0, len(sentences) - 1, 2):
        if i + 1 < len(sentences):
            sentence = sentences[i] + sentences[i + 1]
        else:
            sentence = sentences[i]
        
        sentence = sentence.strip()
        if not sentence:
            continue
        
        # Si le chunk actuel + la phrase dépasse la limite, sauvegarder le chunk
        if current_chunk and len(current_chunk) + len(sentence) > max_chunk_length:
            result.append(current_chunk.strip())
            current_chunk = sentence
        else:
            current_chunk += " " + sentence if current_chunk else sentence
    
    if current_chunk:
        result.append(current_chunk.strip())
    
    return [chunk for chunk in result if chunk]


def generate_speech_chunk(text_chunk: str, lang: str):
    """Génère l'audio pour un chunk de texte"""
    try:
        if lang == "fon":
            inputs = fon_tokenizer(text_chunk, return_tensors="pt").to(fon_tts.device)
            sampling_rate = fon_tts.config.sampling_rate
            
            with torch.no_grad():
                speech_output = fon_tts(**inputs).waveform
            
            data_tts = pad_array(
                speech_output.cpu().numpy().squeeze().astype(np.float32),
                sampling_rate,
            )
            
            return data_tts, sampling_rate
        
        elif lang == "yor":
            inputs = yor_tokenizer(text_chunk, return_tensors="pt").to(yor_tts.device)
            sampling_rate = yor_tts.config.sampling_rate
            
            with torch.no_grad():
                speech_output = yor_tts(**inputs).waveform
            
            data_tts = pad_array(
                speech_output.cpu().numpy().squeeze().astype(np.float32),
                sampling_rate,
            )
            
            return data_tts, sampling_rate
        
        else:
            raise ValueError(f"Langue non supportée: {lang}")
    
    except Exception as e:
        logger.error(f"✗ Erreur lors de la génération d'un chunk: {str(e)}", exc_info=True)
        raise


def convert_audio_chunk_to_wav_bytes(audio_data: np.ndarray, sampling_rate: int) -> bytes:
    """Convertit un chunk audio en bytes WAV"""
    buffer = io.BytesIO()
    sf.write(buffer, audio_data, sampling_rate, format='WAV')
    return buffer.getvalue()


def concatenate_audio_chunks(chunks: list, sampling_rate: int) -> np.ndarray:
    """Concatène plusieurs chunks audio"""
    if not chunks:
        return np.array([])
    return np.concatenate(chunks)


def generate_speech_stream(text_stream: Generator[str, None, None], lang: str, input_audio_path: str) -> Generator[bytes, None, None]:
    """Génère la synthèse vocale en streaming à partir d'un stream de texte"""
    start_time = time.time()
    logger.info(f"=" * 80)
    logger.info(f"GÉNÉRATION DE SYNTHÈSE VOCALE (STREAMING)")
    logger.info(f"Langue: {lang}")
    logger.info(f"Input: {input_audio_path}")
    
    accumulated_text = ""
    chunk_count = 0
    audio_chunks = []
    sampling_rate = None
    
    try:
        # Accumuler le texte jusqu'à avoir des chunks significatifs
        text_buffer = ""
        
        for text_chunk in text_stream:
            text_buffer += text_chunk
            accumulated_text += text_chunk
            
            # Diviser en chunks pour TTS
            chunks = split_text_into_chunks(text_buffer, max_chunk_length=50)
            
            # Traiter tous les chunks sauf le dernier (qui peut être incomplet)
            for chunk in chunks[:-1]:
                if chunk:
                    chunk_count += 1
                    logger.info(f"Génération audio chunk {chunk_count}: {chunk[:50]}...")
                    
                    # Générer l'audio pour ce chunk
                    audio_data, sr = generate_speech_chunk(chunk, lang)
                    if sampling_rate is None:
                        sampling_rate = sr
                    audio_chunks.append(audio_data)
                    
                    # Si on a assez de chunks, envoyer un batch
                    if len(audio_chunks) >= 2:  # Envoyer par batch de 2 chunks
                        concatenated = concatenate_audio_chunks(audio_chunks, sampling_rate)
                        wav_bytes = convert_audio_chunk_to_wav_bytes(concatenated, sampling_rate)
                        yield wav_bytes
                        audio_chunks = []
            
            # Garder le dernier chunk dans le buffer (peut être incomplet)
            text_buffer = chunks[-1] if chunks else ""
        
        # Traiter le dernier chunk restant
        if text_buffer.strip():
            chunk_count += 1
            logger.info(f"Génération audio chunk final {chunk_count}: {text_buffer[:50]}...")
            
            audio_data, sr = generate_speech_chunk(text_buffer, lang)
            if sampling_rate is None:
                sampling_rate = sr
            audio_chunks.append(audio_data)
        
        # Envoyer les chunks restants
        if audio_chunks:
            concatenated = concatenate_audio_chunks(audio_chunks, sampling_rate)
            wav_bytes = convert_audio_chunk_to_wav_bytes(concatenated, sampling_rate)
            yield wav_bytes
        
        # Note: Le clonage de voix est complexe en streaming
        # On pourrait l'appliquer par chunks, mais cela nécessiterait des fichiers temporaires
        # Pour l'instant, on génère l'audio sans clonage en streaming
        # Le clonage pourrait être fait côté client ou en post-traitement
        
        total_duration = time.time() - start_time
        logger.info(f"✓✓ SYNTHÈSE VOCALE STREAMING TERMINÉE en {total_duration:.2f}s ({chunk_count} chunks)")
        logger.info(f"=" * 80)
    
    except Exception as e:
        duration = time.time() - start_time
        logger.error(f"✗ Erreur lors de la génération vocale streaming après {duration:.2f}s: {str(e)}", exc_info=True)
        logger.info(f"=" * 80)
        raise