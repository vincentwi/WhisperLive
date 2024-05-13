import os
import time
import numpy as np
import sounddevice as sd
import wave 
from .utils import eLabs, mp3_to_wav 
import threading
from deep_translator import DeeplTranslator
from fuzzywuzzy import fuzz
from openai import OpenAI

OUTPUT_DEVICE = "toby speakers" # "TOBY"
DEEPL_API_KEY = ""
client = OpenAI(api_key="")
VOICE = {
    "Vincent": "7I3OeCq0m0NAOQs37WrV",
    "Alice": "rWgsVLtrjGcO2pWNsNuS"
}
xi = eLabs(voice=VOICE["Alice"], chunk_size=1024)
Language = {
    "Spanish": "es",
    "French": "fr",
    "English": "en", 
    "Chinese": "zh"
}

def get_device_index(device_name):
    # Query the list of devices
    devices = sd.query_devices()
    
    # Iterate through the devices to find the one that matches the given name
    for idx, device in enumerate(devices):
        if device_name in device['name']:
            return idx

    # Return None if the device is not found
    return None

def write_audio_to_stream(synthesized_speech: bytes, start_time, block_size=1024, channels=4, sample_rate=44100, sample_size_bytes=2):
    """
    Writes audio data to a sounddevice stream.

    Args:
        synthesized_speech (bytes): The audio data to be written.
        block_size (int): The block size for streaming.
        channels (int): The number of audio channels.
        sample_rate (int): The sample rate of the audio data.
    """

    # Ensure the audio data is divisible by the frame size
    frame_size = channels * sample_size_bytes
    remainder = len(synthesized_speech) % frame_size
    if remainder != 0:
        padding_size = frame_size - remainder
        synthesized_speech += b'\0' * padding_size

    # Write the audio data to the stream in chunks
    with sd.RawOutputStream(device= get_device_index(OUTPUT_DEVICE), samplerate=sample_rate, channels=channels, dtype=np.int16) as output_stream:
        end_time = time.time()
        print(f"\n>>Total Time: {end_time - start_time}s")
        for i in range(0, len(synthesized_speech), block_size):
            output_stream.write(synthesized_speech[i:i + block_size])
    


def text_to_speech(text: str, translation_start):
    if not text: return write_audio_to_stream(b'\0', start_time=translation_start)
    if text=="" or text == "42" or text==42 or text==" 42": return write_audio_to_stream(b'\0', start_time=translation_start)

    # TTS conversion
    tts_start = time.time()
    synthesized_speech = xi.TTS(text)
    tts_end = time.time()

    synthesized_speech_wav = mp3_to_wav(synthesized_speech)
    wav_conversion_end = time.time()

    # Write audio to stream
    write_audio_to_stream(synthesized_speech=synthesized_speech_wav, start_time=translation_start)

    # Print timestamps  
    print(f"TTS Time: {tts_end - tts_start}s")
    print(f"WAV Conversion Time: {wav_conversion_end - tts_end}s \n")


def translate_new_words_with_context(src_transcript, target_transcript, new_segment, src_lang, target_lang):

    translation_start = time.time()
    response = client.chat.completions.create(
        model= "gpt-3.5-turbo-1106" , #"gpt-4-1106-preview",
        messages=[
            {"role": "system", "content": f"""You are an expert translator and linguist with years of experience translating from {src_lang} to {target_lang}.
                                            You will be given a transcript in the source language ({src_lang}), and a transcript in the target language ({target_lang})
                                            as well as an incoming segment of text. 
                                            Your role is to identify the most precise and context aware translation for the incoming segment of text.
                                            Your response will be in two parts, seperated by a | symbol. You will respond with nothing else. 
                                            On the left of the | symbol, respond with the NEW information of the incoming text. AND ONLY THE NEW TEXT.
                                            On the right of the | symbol, respond the TRANSLATED version of the new segment from {src_lang} to {target_lang}. AND ONLY THE TRANSLATION.
                                            DO NOT add superflous text. RESPOND ONLY WITH THE FORMAT: NEW INFORMATION | TRANSLATED TEXT 
                                            Keep your answer context aware and consistent with the tone, formality and theme of the {src_lang} and {target_lang} transcripts.
                                            This new text may or may not completely overlap in meaning with the previous transcript. 
                                            In such a case, DO NOT REPEAT THE TEXT. RESPOND ONLY WITH THE NUMBER 42. 
                                            Below are successful examples from language pairs you have done in the past. 

                                            - EXAMPLE 1:
                                            Source Transcript: ['Hello, my name is']
                                            Target Transcript: ['Bonjour, mon nom est']
                                            New Segment: Vincent, what is yours?
                                            Source Language: English
                                            Target Language: French
                                            Output: Vincent, what is yours? | Vincent, et toi?

                                            - EXAMPLE 2:
                                            Source Transcript: ['Hello, my name is', 'Vincent, what is yours?']
                                            Target Transcript: ['Bonjour, mon nom est', 'Vincent, et toi?']
                                            New Segment: My name is Toby. Nice to meet you.
                                            Source Language: en
                                            Target Language: French
                                            Output: My name is Toby. Nice to meet you. | Mon nom est Toby. Ravi de faire votre connaissance. 

                                            - EXAMPLE 3:
                                            Source Transcript: ['Hello, my name is', 'Vincent, what is yours?', 'My name is Toby. Nice to meet you.']
                                            Target Transcript: ['Bonjour, mon nom est', 'Vincent, et toi?', 'Mon nom est Toby. Ravi de faire votre connaissance.']
                                            New Segment: It's nice to meet you, where are you now? 
                                            Source Language: English
                                            Target Language: fr
                                            Output: where are you now? | ou est tu maintenant?

                                            - EXAMPLE 4:
                                            Source Transcript: ['Hello, my name is', 'Vincent, what is yours?', 'My name is Toby. Nice to meet you.', 'where are you now?']
                                            Target Transcript: ['Bonjour, mon nom est', 'Vincent, et toi?', 'Mon nom est Toby. Ravi de faire votre connaissance.', 'ou est tu maintenant?']
                                            New Segment: where are you based nowadays?
                                            Source Language: en
                                            Target Language: fr
                                            Output: 42 | 42 

                                            - EXAMPLE 5:
                                            Source Transcript: ['Hola, ¿has probado el nuevo producto?']
                                            Target Transcript: ['你好，你试过新产品吗？']
                                            New Segment: Sí, lo probé ayer.
                                            Source Language: Spanish
                                            Target Language: Chinese
                                            Output: Sí, lo probé ayer. | 是的，我昨天试了。

                                            - EXAMPLE 6:
                                            Source Transcript: ['Hola, ¿has probado el nuevo producto?', 'Sí, lo probé ayer.']
                                            Target Transcript: ['你好，你试过新产品吗？', '是的，我昨天试了。']
                                            New Segment: ¿Y qué te pareció?
                                            Source Language: es
                                            Target Language: Chinese
                                            Output: ¿Y qué te pareció? | 你觉得怎么样？

                                            - EXAMPLE 7:
                                            Source Transcript: ['Hola, ¿has probado el nuevo producto?', 'Sí, lo probé ayer.', '¿Y qué te pareció?']
                                            Target Transcript: ['你好，你试过新产品吗？', '是的，我昨天试了。', '你觉得怎么样？']
                                            New Segment: Me gustó mucho.
                                            Source Language: Spanish
                                            Target Language: zh
                                            Output: Me gustó mucho. | 我非常喜欢。

                                            - EXAMPLE 8:
                                            Source Transcript: ['Este producto es realmente útil.']
                                            Target Transcript: ['这个产品真的很有用。']
                                            New Segment: Es realmente útil y eficiente.
                                            Source Language: Spanish 
                                            Target Language: Chinese
                                            Output: y eficiente. | 而且效率很高。

                                            - EXAMPLE 9:
                                            Source Transcript: ['Me gusta cómo funciona este dispositivo.']
                                            Target Transcript: ['我喜欢这个设备的运作方式。']
                                            New Segment: Me gusta cómo funciona y su diseño.
                                            Source Language: es
                                            Target Language: zh
                                            Output: y su diseño. | 和它的设计。

                                            - EXAMPLE 10:
                                            Source Transcript: ['Hola, ¿has probado el nuevo producto?', 'Sí, lo probé ayer.', '¿Y qué te pareció?', 'Me gustó mucho.', '¿Qué características te gustaron más?']
                                            Target Transcript: ['你好，你试过新产品吗？', '是的，我昨天试了。', '你觉得怎么样？', '我非常喜欢。', '你最喜欢哪些特点？']
                                            New Segment: Me gustaron más. Es innovador.
                                            Source Language: Spanish
                                            Target Language: Chinese
                                            Output: Es innovador. | 它很创新。

                                            - EXAMPLE 10:
                                            Source Transcript: ['Es un producto innovador.']
                                            Target Transcript: ['这是一个创新的产品。']
                                            New Segment: Realmente es un producto innovador.
                                            Source Language: es
                                            Target Language: Chinese
                                            Output: 42 | 42

                                            """},
            {"role": "user", "content": f"""Source Transcript: {src_transcript}
                                            Target Transcript: {target_transcript}
                                            New Segment: {new_segment}
                                            Source Language: {src_lang}
                                            Target Language: {target_lang}
                                            Output: """}
        ],
        stream=True  
    )
    result = ""
    for chunk in response: 
        chunk_message = chunk.choices[0].delta.content
        if chunk_message:
            result += chunk_message
    translation_end = time.time()
    print(result)
    print(f"Translation Time: {translation_end - translation_start}s")
    
    try:
        src_new, target_new = result.split("|")

        if len(target_transcript) == 0:
            t = threading.Thread(
                    target=text_to_speech,
                    args=(
                        target_new[1:],
                        translation_start,
                    ),
                )
            t.start()  
            return src_new[:-1], target_new[1:] #dealing with the    spaces 
        elif fuzz.ratio(target_new[1:], target_transcript[-1]) < 80: #redundancy check
            t = threading.Thread(
                    target=text_to_speech,
                    args=(
                        target_new[1:],
                        translation_start,
                    ),
                )
            t.start()  
            return src_new[:-1], target_new[1:] #dealing with the    spaces 

    except Exception as e: 
        print(e)
        translate_new_words_with_context(src_transcript, target_transcript, new_segment, src_lang, target_lang) #try again lmfao
        pass 

    return "", ""
    
