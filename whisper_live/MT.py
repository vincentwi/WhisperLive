# MT.py 
from openai import OpenAI
import time
import threading
from fuzzywuzzy import fuzz

client = OpenAI(api_key="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")


async def translate_new_words_with_context(SRC, TGT, new_sentence, SRC_LANG, TGT_LANG):
    translation_start = time.time()
    response = client.chat.completions.create(
        model=  "gpt-3.5-turbo-1106", #"gpt-4o-2024-05-13",
        messages=[
            {"role": "system", "content": f"""You are an expert translator and linguist with years of experience translating from {SRC_LANG} to {TGT_LANG}.
                                            You will be given a transcript in the source language ({SRC_LANG}), and a transcript in the target language ({TGT_LANG})
                                            as well as an incoming segment of text. 
                                            Your role is to identify the most precise and context aware translation for the incoming segment of text.
                                            Your response will be in two parts, seperated by a | symbol. You will respond with nothing else. 
                                            On the left of the | symbol, respond with the NEW information of the incoming text. AND ONLY THE NEW TEXT.
                                            On the right of the | symbol, respond the TRANSLATED version of the new segment from {SRC_LANG} to {TGT_LANG}. AND ONLY THE TRANSLATION.
                                            DO NOT add superflous text. RESPOND ONLY WITH THE FORMAT: NEW INFORMATION | TRANSLATED TEXT 
                                            Keep your answer context aware and consistent with the tone, formality and theme of the {SRC_LANG} and {TGT_LANG} transcripts.
                                            This new text may or may not completely overlap in meaning with the previous transcript. 
                                            In such a case, DO NOT REPEAT THE TEXT. RESPOND ONLY WITH THE NUMBER 42. 
                                            Below are successful examples from language pairs you have done in the past. 

                                            - EXAMPLE 1:
                                                Source Transcript: 'Hello, my name is'
                                                Target Transcript: 'Hola, mi nombre es'
                                                New Segment: Vincent, what is yours?
                                                Source Language: English
                                                Target Language: Spanish
                                                Output: Vincent, what is yours? | Vincent, ¿cuál es el tuyo?

                                            - EXAMPLE 2:
                                                Source Transcript: 'Hello, my name is', 'Vincent, what is yours?'
                                                Target Transcript: 'Hola, mi nombre es', 'Vincent, ¿cuál es el tuyo?'
                                                New Segment: My name is Toby. Nice to meet you.
                                                Source Language: English
                                                Target Language: Spanish
                                                Output: My name is Toby. Nice to meet you. | Me llamo Toby. Encantado de conocerte.

                                            - EXAMPLE 3:
                                                Source Transcript: 'Hello, my name is', 'Vincent, what is yours?', 'My name is Toby. Nice to meet you.'
                                                Target Transcript: 'Hola, mi nombre es', 'Vincent, ¿cuál es el tuyo?', 'Me llamo Toby. Encantado de conocerte.'
                                                New Segment: It's nice to meet you, where are you now?
                                                Source Language: English
                                                Target Language: Spanish
                                                Output: where are you now? | ¿dónde estás ahora?

                                            - EXAMPLE 4:
                                                Source Transcript: 'Hello, my name is', 'Vincent, what is yours?', 'My name is Toby. Nice to meet you.', 'where are you now?'
                                                Target Transcript: 'Hola, mi nombre es', 'Vincent, ¿cuál es el tuyo?', 'Me llamo Toby. Encantado de conocerte.', '¿dónde estás ahora?'
                                                New Segment: where are you based nowadays?
                                                Source Language: English
                                                Target Language: Spanish
                                                Output: 42 | 42 

                                            - EXAMPLE 5:
                                                Source Transcript: 'Hi, have you tried the new product?'
                                                Target Transcript: 'Hola, ¿has probado el nuevo producto?'
                                                New Segment: Yes, I tried it yesterday.
                                                Source Language: English
                                                Target Language: Spanish
                                                Output: Yes, I tried it yesterday. | Sí, lo probé ayer.

                                            - EXAMPLE 6:
                                                Source Transcript: 'Hi, have you tried the new product?', 'Yes, I tried it yesterday.'
                                                Target Transcript: 'Hola, ¿has probado el nuevo producto?', 'Sí, lo probé ayer.'
                                                New Segment: And what did you think?
                                                Source Language: English
                                                Target Language: Spanish
                                                Output: And what did you think? | ¿Y qué te pareció?

                                            - EXAMPLE 7:
                                                Source Transcript: 'Hi, have you tried the new product?', 'Yes, I tried it yesterday.', 'And what did you think?'
                                                Target Transcript: 'Hola, ¿has probado el nuevo producto?', 'Sí, lo probé ayer.', '¿Y qué te pareció?'
                                                New Segment: I liked it a lot.
                                                Source Language: English
                                                Target Language: Spanish
                                                Output: I liked it a lot. | Me gustó mucho.

                                            - EXAMPLE 8:
                                                Source Transcript: 'This product is really useful.'
                                                Target Transcript: 'Este producto es realmente útil.'
                                                New Segment: It is really useful and efficient.
                                                Source Language: English
                                                Target Language: Spanish
                                                Output: and efficient. | y eficiente.

                                            - EXAMPLE 9:
                                                Source Transcript: 'I like how this device works.'
                                                Target Transcript: 'Me gusta cómo funciona este dispositivo.'
                                                New Segment: I like how it works and its design.
                                                Source Language: English
                                                Target Language: Spanish
                                                Output: and its design. | y su diseño.

                                            - EXAMPLE 10:
                                                Source Transcript: 'Hi, have you tried the new product?', 'Yes, I tried it yesterday.', 'And what did you think?', 'I liked it a lot.', 'What features did you like the most?'
                                                Target Transcript: 'Hola, ¿has probado el nuevo producto?', 'Sí, lo probé ayer.', '¿Y qué te pareció?', 'Me gustó mucho.', '¿Qué características te gustaron más?'
                                                New Segment: I liked them more. It's innovative.
                                                Source Language: English
                                                Target Language: Spanish
                                                Output: It's innovative. | Es innovador



                                            """},
            {"role": "user", "content": f"""Source Transcript: {SRC}
                                            Target Transcript: {TGT}
                                            New Segment: {new_sentence}
                                            Source Language: {SRC_LANG}
                                            Target Language: {TGT_LANG}
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
    print("GPT result: ", result)
    print(f"Translation Time: {translation_end - translation_start}s \n")
    
    try:
        SRCn, TGTn = result.split(" | ")
        # print(SRCn, TGTn)
        # # push pop
        # if len(TGT) == 0:
        #     t = threading.Thread(
        #             target=text_to_speech,
        #             args=(
        #                 TGTn,
        #                 translation_start,
        #             ),
        #         )
        #     t.start()  
        #     return SRCn, TGTn #dealing with the    spaces 
        # elif fuzz.ratio(TGTn, TGT) < 80: #redundancy check
        #     t = threading.Thread(
        #             target=text_to_speech,
        #             args=(
        #                 TGTn,
        #                 translation_start,
        #             ),
        #         )
        #     t.start()  
            # return SRCn, TGTn #dealing with the    spaces 
        return SRCn, TGTn
    except Exception as e: 
        print(e)
        translate_new_words_with_context(SRC, TGT, new_sentence, SRC_LANG, TGT_LANG) #try again lmfao
        pass 

    return "", ""

