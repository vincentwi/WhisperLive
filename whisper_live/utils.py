import os
import requests
# os.system("pip3 install -r requirements.txt -q")
# print("downloaded")
import re 
from elevenlabs import generate, play, voices, clone, save, stream, set_api_key, api
set_api_key("")
from pathlib import Path
import random 
import openai
import time
from pydub import AudioSegment
from difflib import SequenceMatcher
# Model = 'tiny' #@param ['tiny.en', 'tiny', 'base.en', 'base', 'small.en', 'small', 'medium.en', 'medium', 'large']  

def mp3_to_wav(file_name: str, stretch_factor: float = 0.25) -> bytes:
    """
    Converts an MP3 file to WAV format, optionally applying a speed stretch.
    
    Args:
        file_name (str): Name of the MP3 file (without extension).
        stretch_factor (float): Factor by which to stretch the audio speed.
        
    Returns:
        bytes: The WAV file data as a byte array.
    """
    # Generate file paths
    mp3_file_path = f'{file_name}.mp3'
    wav_file_path = f'{file_name}.wav'

    # Load and process MP3 file
    audio = AudioSegment.from_mp3(mp3_file_path)
    new_frame_rate = int(audio.frame_rate * stretch_factor)
    stretched_audio = audio._spawn(audio.raw_data, overrides={'frame_rate': new_frame_rate})
    stretched_audio.set_frame_rate(audio.frame_rate).export(wav_file_path, format="wav")

    # Read and return WAV data
    with open(wav_file_path, 'rb') as file:
        wav_bytes = file.read()
    return wav_bytes

class Transcriber():
    def __init__(self, p2audio, uploaded_file=None):
        self.p2audio = p2audio
        self.uploaded_file = uploaded_file
    def get_transcipt(self):     
        try:
            if self.uploaded_file is None: self.uploaded_file = self.p2audio
            transcript = openai.Audio.transcribe("whisper-1", self.uploaded_file)
        except: 
            transcript = {"text": f"error getting content from {self.p2audio}"}
        return transcript 

class Translation():
    def __init__(self, p2audio):
        self.p2audio = p2audio
    def get_translation(self):     
        try: 
            audio= open(Path(self.p2audio), "rb")
            translated = openai.Audio.translate("whisper-1", audio) 
        except: 
            translated = {"text": f"error getting content from {self.p2audio}"}
        return translated


def get_voices(): 
    voicez = pd.DataFrame(voices()) 
    voice_names = [voicez.iloc[i,1][1] for i in range(len(voicez))]
    return voice_names

class eLabs():
    def __init__(self, voice="Adam", stability=0.5, similarity=0.75, chunk_size=1024): 
        self.voice = voice 
        self.CHUNK_SIZE = chunk_size
        self.stability = stability
        self.similarity = similarity
        self.style = 0
        self.use_speaker_boost = True

    def TTS(self, text, save_file=True, play_from_shell=False):  

        # audio = stream(generate(text, voice=self.voice, model="eleven_multilingual_v2", 
        #                  stream=True, stream_chunk_size=self.CHUNK_SIZE)
        # tts = api.tts.TTS()
        # tts.generate
        # audio = generate(text, 
        #                  voice=self.voice, 
        #                  model="eleven_multilingual_v2", 
        #                  stream=False, 
        #                  stream_chunk_size=self.CHUNK_SIZE,
        #                  latency=4)




        CHUNK_SIZE = 1024
        url = f"https://api.elevenlabs.io/v1/text-to-speech/{self.voice}?output_format=mp3_44100&optimize_streaming_latency=4"

        headers = {
        "Accept": "audio/mpeg",
        "Content-Type": "application/json",
        "xi-api-key": "4d50427fecd16dcc5c5bb7f2d3dfc258"
        }

        data = {
        "text": text,
        "model_id": "eleven_multilingual_v2",
        "voice_settings": {
            "stability": self.stability,
            "similarity_boost": 0.99,
            "style": self.style,
            "use_speaker_boost": self.use_speaker_boost,
        }
        }

        audio = requests.post(url, json=data, headers=headers)
        # audio_data = list(audio.iter_content(CHUNK_SIZE))
        audio_data = audio.content
        name = f'/Users/vinceroy/Desktop/live_whisper/WhisperLive/whisper_live/data/translated/{text[:5]}{time.time()}.mp3'
        if save_file:
            save(audio_data, name)
        if play_from_shell: 
            play(audio_data) 
        return name[:-4]
    
    def update_voice(self, name):
        self.voice = name

    def clone_voice(self, name, p2file):
        print("* cloning voice")
        if type(p2file) != type([]): p2file = [p2file]
        voice = clone(name=name, files=p2file) 
        self.update_voice(name)
 