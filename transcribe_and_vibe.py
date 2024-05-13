from whisper_live.client import TranscriptionClient
client = TranscriptionClient("0.0.0.0", 9090, is_multilingual=True, lang="en", translate=True)
client()