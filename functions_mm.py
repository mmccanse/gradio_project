from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os
from langchain.schema import AIMessage, HumanMessage
import gradio as gr
from transformers import pipeline, AutoTokenizer, TFAutoModelForSeq2SeqLM
import subprocess
import torch
#Google Text to Speech
from gtts import gTTS
import tempfile
from langdetect import detect
from transformers import MarianMTModel, MarianTokenizer
import boto3
# Additional imports for loading PDF documents and QA chain.
from langchain_community.document_loaders import PyPDFLoader
# Additional imports for loading Wikipedia content and QA chain
from langchain_community.document_loaders import WikipediaLoader
from langchain.chains.question_answering import load_qa_chain
# Import RegEx for translate function, to split sentences in avoiding token limits
import re

#Get keys #########################################################################################
load_dotenv()

# Set the model name for our LLMs.
OPENAI_MODEL = "gpt-3.5-turbo"
# Store the API key in a variable.
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

#Define language variables ###################################################################################

#Define voice map
voice_map = {
    "ar": "Hala",
    "en": "Gregory",
    "es": "Mia",
    "fr": "Liam",
    "de": "Vicki",
    "it": "Bianca",
    "zh": "Hiujin",
    "hi": "Kajal",
    "jap": "Tomoko",
    "trk": "Burcu", 
    "vi": "no audio available for this language"
    
    }

#Define language map from full names to ISO codes
language_map = {
    "Arabic (Gulf)": "ar",
    "Chinese (Cantonese)": "zh",
    "English": "en",
    "French": "fr",
    "German": "de",
    "Hindi": "hi",
    "Italian": "it",
    "Japanese": "jap",
    "Spanish": "es",
    "Turkish": "trk",
    "Vietnamese (no audio available)": "vi"
    
}

# list of languages and their codes for dropdown
languages = gr.Dropdown(
    label="Click in the middle of the dropdown bar to select translation language", 
    choices=list(language_map.keys()))

#Define default language
default_language = "English"

#Setting the Chatbot Model #################################################################################

# Set the model name for our LLMs.
OPENAI_MODEL = "gpt-3.5-turbo"
# Store the API key in a variable.
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

#Instantiating the llm we'll use and the arguments to pass
llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY, model_name=OPENAI_MODEL, temperature=0.0)

# Define the wikipedia topic as a string.
wiki_topic = "diabetes"

# Load the wikipedia results as documents, using a max of 2.
#included error handling- unable to load documents
try:
    documents = WikipediaLoader(query=wiki_topic, load_max_docs=2, load_all_available_meta=True).load()
except Exception as e:
    print("Failed to load documents:", str(e))
    documents = []
# Create the QA chain using the LLM.
chain = load_qa_chain(llm)

##############################################################################################################

#Define the function to call the OpenAI chat LLM
def handle_query(user_query):
    if not documents:
        return "Source not loading info; please try again later."

    if user_query.lower() == 'quit':
        return "Goodbye!"

    try:
        # Pass the documents and the user's query to the chain, and return the result.
        result = chain.invoke({"input_documents": documents, "question": user_query})
        return result["output_text"] if result["output_text"].strip() else "No answer found, try a different question."
        
    except Exception as e:
        return "An error occurred while searching for the answer: " + str(e)
    
#Language models and functions ############################################################################

#Setup cache mechanism to initialize translation model at module level to improve app speed.

#Define global variables for tokenizer and model
helsinki_model_cache = {}

def get_helsinki_model_and_tokenizer(src_lang, target_lang):
    helsinki_model_name =f"Helsinki-NLP/opus-mt-{src_lang}-{target_lang}"
    if helsinki_model_name not in helsinki_model_cache:
        tokenizer = MarianTokenizer.from_pretrained(helsinki_model_name)
        model = MarianMTModel.from_pretrained(helsinki_model_name)
        helsinki_model_cache[helsinki_model_name] = (tokenizer, model)
    return helsinki_model_cache[helsinki_model_name]

#Define function to transcribe audio to text and then translate it into the specified language
def translate(transcribed_text, target_lang="es"):
    try:
        #Define the model and tokenizer
        src_lang = detect(transcribed_text)
        tokenizer, model = get_helsinki_model_and_tokenizer(src_lang, target_lang)
        max_length = tokenizer.model_max_length

        # Split text based on sentence endings to better manage translation segments
        sentences = re.split(r'(?<=[.!?]) +', transcribed_text)
        full_translation = ""

        # Process each sentence individually
        for sentence in sentences:
            tokens = tokenizer.encode(sentence, return_tensors="pt", truncation=True, max_length=max_length)
            if tokens.size(1) > max_length:
                continue  # optionally handle long sentences longer than max # tokens for model
            translated_tokens = model.generate(tokens)
            segment_translation = tokenizer.decode(translated_tokens[0], skip_special_tokens=True)
            full_translation += segment_translation + " "

        return full_translation.strip()

    except Exception as e:
        print(f"An error occurred: {e}")
        return "Error in transcription or translation"
        

#Initialize Whisper model at the module level to be used across different calls
transcription_pipeline = None

def initialize_transcription_model():
    global transcription_pipeline
    if transcription_pipeline is None:
        transcription_pipeline = pipeline("automatic-speech-recognition", model="openai/whisper-large")

#Define function to transcribes audio to text using Whisper in the original language it was spoken
def transcribe_audio_original(audio_filepath):
    try:
        if transcription_pipeline is None:
            initialize_transcription_model()
        transcription_result = transcription_pipeline(audio_filepath)
        transcribed_text = transcription_result['text']
        return transcribed_text
    except Exception as e:
        print(f"an error occured: {e}")
        return "Error in transcription"


# Define function to translate text to speech for output
# Using Google Text-to-speech

def text_to_speech(text):
    tts = gTTS(text, lang='en')
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp3')
    tts.save(temp_file.name)
    return temp_file.name



#Initialize Polly client at module level
polly_client = boto3.client('polly')

# Define text-to-speech function using Amazon Polly
def polly_text_to_speech(text, lang_code):
    
    try:
    
        #get the appropriate voice ID from the mapping
        voice_id = voice_map[lang_code]
        
        #request speech synthesis
        response = polly_client.synthesize_speech(
            Engine = 'neural',
            Text=text,
            OutputFormat='mp3',
            VoiceId=voice_id
        )
        
        # Save the audio to a temporary file and return its path
        if "AudioStream" in response:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as audio_file:
                audio_file.write(response['AudioStream'].read())
                return audio_file.name
    except boto3.exceptions.Boto3Error as e:
        print(f"Error accessing Polly: {e}")
    return None  # Return None if there was an error



#Define function to submit query to Wikipedia to feed into Gradio app
def submit_question (audio_filepath=None, typed_text=None, target_lang=default_language):
    
    #Determine source of text: audio transctiption or direct text input
    # if audio_filepath and typed_text:
    #     return "Please use only one input method at a time", None
    
    if not audio_filepath and not typed_text:
        return "Please provide input by typing or speaking", None
    
    response_speech = None
    response_text = None
    
    if typed_text:
        query_text = typed_text
    elif audio_filepath:
        query_text = transcribe_audio_original(audio_filepath)
    else:
        return "No valid input provided", None, None
    
    # Query calls OPEN AI chat model
    detected_lang_code = detect(query_text)
    response_text = handle_query(query_text)
        
    # Check if detected language is supported for audio
    voice_id = voice_map.get(detected_lang_code, None)
    if voice_id == "no audio available for this language":
        no_audio_message = voice_id
    else:
        response_speech = polly_text_to_speech(response_text, detected_lang_code)
        no_audio_message = None if response_speech else "Error in generating audio"
        
    return response_text, response_speech, no_audio_message

#Define function to transcribe audio and provide output in text and speech
def transcribe_and_speech(audio_filepath=None, typed_text=None, target_lang=default_language):
    
    #Determine source of text: audio transctiption or direct text input
    if audio_filepath and typed_text:
        return "Please use only one input method at a time", None
    
    query_text = None
    detected_lang_code = None
    original_speech = None
    
    if typed_text:
        #convert typed text to speech
        query_text = typed_text
        detected_lang_code = detect(query_text)
        original_speech = polly_text_to_speech(query_text, detected_lang_code)
        return None, original_speech
    
    elif audio_filepath:
        #transcribe audio to text
        query_text = transcribe_audio_original(audio_filepath)
        detected_lang_code = detect(query_text)
        original_speech = polly_text_to_speech(query_text, detected_lang_code)
        return query_text, original_speech
    
    if not query_text:
        return "Please provide input by typing or speaking.", None
    
    #Check if the language is specified. Default to English if not.
    target_lang_code = language_map.get(target_lang, "en")
    
    #Map detected language code to language name
    detected_lang = [key for key, value in language_map.items() if value == detected_lang_code][0]
    
    
    return query_text, original_speech


#Define function to translate query into target language in text and audio
def translate_and_speech(response_text=None, target_lang=default_language):
    
        
    #Detect language of input text
    detected_lang_code = detect(response_text)
    detected_lang = [key for key, value in language_map.items() if value == detected_lang_code][0]
    
    #Check if the language is specified. Default to English if not.
    target_lang_code = language_map.get(target_lang, "en")
    
    #Process text: translate 
    #Check if the detected language and target language are the same
    if detected_lang == target_lang:
        translated_response = response_text
    else:
        translated_response = translate(response_text, target_lang_code)
    
    #convert to speech
    voice_id = voice_map[target_lang_code]
    if voice_id == "no audio available for this language":
        translated_speech = None
        no_audio_message = voice_id
    else:
        translated_speech = polly_text_to_speech(translated_response, target_lang_code)
        no_audio_message = None if translated_speech else "Error in generating audio"
    
    return  translated_response, translated_speech, no_audio_message


# Function to clear out all inputs
def clear_inputs():
    return None, None, None, None, None, None, None, None