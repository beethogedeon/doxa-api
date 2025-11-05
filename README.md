# Doxa API

API FastAPI pour la transcription audio et la synthèse vocale.

## Installation

```bash
pip install -r requirements.txt
```

## Utilisation

### Démarrer le serveur

```bash
uvicorn main:app --reload
```

L'API sera accessible sur `http://localhost:8000`

### Documentation interactive

Accédez à la documentation Swagger sur `http://localhost:8000/docs`

## Endpoints

### POST /transcribe

Transcrit un fichier audio en utilisant SpeechBrain.

**Paramètres:**
- `audio`: Fichier audio (formats supportés par SpeechBrain)

**Réponse:**
```json
{
  "transcription": "texte transcrit"
}
```

### POST /generate_speech

Génère de la parole à partir de texte.

**Paramètres:**
- `text`: Texte à synthétiser
- `lang`: Langue de destination ("fon" pour Fongbe, "yor" pour Yoruba - à venir)

**Réponse:**
Fichier audio en bytes (format WAV)

## Exemple d'utilisation

### Transcription

```bash
curl -X POST "http://localhost:8000/transcribe" \
  -F "audio=@mon_fichier.wav"
```

### Génération vocale

```bash
curl -X POST "http://localhost:8000/generate_speech?text=Bonjour&lang=fon" \
  --output output.wav
```

