import tensorflow
import torch
import numpy as np
import logging
import soundfile as sf
import os
from openvoice import se_extractor
from openvoice.api import ToneColorConverter


# Configure the basic logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Get a logger instance
logger = logging.getLogger(__name__)



# Use a pipeline as a high-level helper
from transformers import VitsModel, AutoTokenizer

fon_tts = VitsModel.from_pretrained("facebook/mms-tts-fon")
fon_tokenizer = AutoTokenizer.from_pretrained(
    "facebook/mms-tts-fon"
)

yor_tts = VitsModel.from_pretrained("facebook/mms-tts-yor")
yor_tokenizer = AutoTokenizer.from_pretrained(
    "facebook/mms-tts-yor"
)

ckpt_converter = './checkpoints_v2/converter'
device = "cuda" if torch.cuda.is_available() else "cpu"
output_dir = 'outputs_v2'




tone_color_converter = ToneColorConverter(f'{ckpt_converter}/config.json', device=device)
tone_color_converter.load_ckpt(f'{ckpt_converter}/checkpoint.pth')


reference_speaker = './consolas_voice.wav' # This is the voice you want to clone
target_se, audio_name = se_extractor.get_se(reference_speaker, tone_color_converter, vad=True)


os.makedirs(output_dir, exist_ok=True)

def clone_voice(input_audio_path: str, output_audio_path: str):
    try:
        source_se = tone_color_converter.extract_se(input_audio_path)
        tone_color_converter.convert(
            audio_src_path=input_audio_path,
            src_se=source_se,
            tgt_se=target_se,
            output_path=output_audio_path,
            message="@Doxa AI"
            )
        return True
    except Exception as e:
        logger.error(f"Error cloning voice: {e}")
        return False

def pad_array(array, sr):

    if isinstance(array, list):
        array = np.array(array)

    if not array.shape[0]:
        raise ValueError("The generated audio does not contain any data")

    valid_indices = np.where(np.abs(array) > 0.001)[0]

    if len(valid_indices) == 0:
        logger.debug(f"No valid indices: {array}")
        return array

    try:
        pad_indice = int(0.1 * sr)
        start_pad = max(0, valid_indices[0] - pad_indice)
        end_pad = min(len(array), valid_indices[-1] + 1 + pad_indice)
        padded_array = array[start_pad:end_pad]
        return padded_array
    except Exception as error:
        logger.error(str(error))
        return array

def write_chunked( file, data, samplerate, subtype=None, endian=None, format=None, closefd=True):
  data = np.asarray(data)
  if data.ndim == 1:
      channels = 1
  else:
      channels = data.shape[1]

  with sf.SoundFile(
      file, 'w', samplerate, channels,
      subtype, endian, format, closefd
  ) as f:
          f.write(data)

def generate_speech(text: str, lang: str, input_audio_path: str, output_audio_path: str):
    if lang == "fon":
        inputs = fon_tokenizer(text, return_tensors="pt")
        sampling_rate = fon_tts.config.sampling_rate
        with torch.no_grad():
            speech_output = fon_tts(**inputs).waveform

        data_tts = pad_array(
            speech_output.cpu().numpy().squeeze().astype(np.float32),
            sampling_rate,
        )

        write_chunked(
            file=output_audio_path,
            samplerate=sampling_rate,
            data=data_tts,
            format="wav",
            #subtype="vorbis",
        )

        clone_voice(input_audio_path, output_audio_path)

        return output_audio_path
    
    elif lang == "yor":
        inputs = yor_tokenizer(text, return_tensors="pt")
        sampling_rate = yor_tts.config.sampling_rate
        with torch.no_grad():
            speech_output = yor_tts(**inputs).waveform

        data_tts = pad_array(
            speech_output.cpu().numpy().squeeze().astype(np.float32),
            sampling_rate,
        )

        write_chunked(
            file=output_audio_path,
            samplerate=sampling_rate,
            data=data_tts,
            format="wav",
            #subtype="vorbis",
        )

        clone_voice(input_audio_path, output_audio_path)

        return output_audio_path