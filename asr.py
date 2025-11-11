import whisper
from speechbrain.inference import EncoderASR


def get_speechbrain_model():
    """Charge le modèle SpeechBrain de manière lazy"""

    fon_asr_model = EncoderASR.from_hparams(
        source="speechbrain/asr-wav2vec2-dvoice-fongbe",
        savedir="pretrained_models/asr-wav2vec2-dvoice-fongbe",
        run_opts={"device": "cuda:0"}
    )
    
    return fon_asr_model

fon_asr_model = get_speechbrain_model()

model = whisper.load_model("turbo")

def detect_language(audio_path: str):

    # load audio and pad/trim it to fit 30 seconds
    audio = whisper.load_audio(audio_path)
    audio = whisper.pad_or_trim(audio)

    # make log-Mel spectrogram and move to the same device as the model
    mel = whisper.log_mel_spectrogram(audio, n_mels=model.dims.n_mels).to(model.device)

    # detect the spoken language
    _, probs = model.detect_language(mel)
    return max(probs, key=probs.get), mel


def transcribe(audio_path: str):
    lang, mel = detect_language(audio_path)

    if lang == "yor":
        
        # decode the audio
        options = whisper.DecodingOptions()
        result = whisper.decode(model, mel, options)

        return result.text, lang
    else :

        transcription = fon_asr_model.transcribe_file(audio_path)
        return transcription, "fon"
