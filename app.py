# %%
# pip install langdetect
# pip install sentencepiece
# pip install boto3
# pip install awscli
# pip install sacremoses

# %%
import gradio as gr
from transformers import pipeline, AutoTokenizer, TFAutoModelForSeq2SeqLM
from dotenv import load_dotenv
import os
import subprocess
import torch
import tempfile
from langdetect import detect
from transformers import MarianMTModel, MarianTokenizer
import re
import boto3

# %%
# import functions from functions file

from functions_mm import handle_query, transcribe_audio_original, submit_question, polly_text_to_speech, translate, translate_and_speech, clear_inputs, voice_map, language_map, default_language, languages



# %%
# gr.themes.builder()

# %%
import gradio as gr

theme = gr.themes.Soft(
    secondary_hue="teal",
    neutral_hue="slate",
    font=[gr.themes.GoogleFont('Source Sans Pro'), 'ui-sans-serif', 'system-ui', 'sans-serif'],
).set(
    background_fill_primary='white',
    
    #change background color here
    body_background_fill='*primary_100',
    
    shadow_drop='rgba(0,0,0,0.05) 0px 1px 2px 0px',
    shadow_drop_lg='0 1px 3px 0 rgb(0 0 0 / 0.1), 0 1px 2px -1px rgb(0 0 0 / 0.1)',
    shadow_spread='3px',
    block_background_fill='*background_fill_primary',
    block_border_width='1px',
    block_border_width_dark='1px',
    block_label_background_fill='*background_fill_primary',
    block_label_background_fill_dark='*background_fill_secondary',
    block_label_text_color='*neutral_500',
    block_label_text_color_dark='*neutral_200',
    block_label_margin='0',
    block_label_padding='*spacing_sm *spacing_lg',
    block_label_radius='calc(*radius_lg - 1px) 0 calc(*radius_lg - 1px) 0',
    block_label_text_size='*text_sm',
    block_label_text_weight='400',
    block_title_background_fill='none',
    block_title_background_fill_dark='none',
    block_title_text_color='*neutral_500',
    block_title_text_color_dark='*neutral_200',
    block_title_padding='0',
    block_title_radius='none',
    block_title_text_weight='400',
    
    #panel border width change here
    panel_border_width='1px',
    
    #try panel border here
    panel_border_color='*neutral_900',
    
    panel_border_width_dark='0',
    checkbox_background_color_selected='*secondary_600',
    checkbox_background_color_selected_dark='*secondary_600',
    checkbox_border_color='*neutral_300',
    checkbox_border_color_dark='*neutral_700',
    checkbox_border_color_focus='*secondary_500',
    checkbox_border_color_focus_dark='*secondary_500',
    checkbox_border_color_selected='*secondary_600',
    checkbox_border_color_selected_dark='*secondary_600',
    checkbox_border_width='*input_border_width',
    checkbox_shadow='*input_shadow',
    checkbox_label_background_fill_selected='*checkbox_label_background_fill',
    checkbox_label_background_fill_selected_dark='*checkbox_label_background_fill',
    checkbox_label_shadow='none',
    checkbox_label_text_color_selected='*checkbox_label_text_color',
    input_background_fill='*neutral_100',
    input_border_color='*border_color_primary',
    input_shadow='none',
    input_shadow_dark='none',
    input_shadow_focus='*input_shadow',
    input_shadow_focus_dark='*input_shadow',
    slider_color='#2563eb',
    slider_color_dark='#2563eb',
    button_shadow='none',
    button_shadow_active='none',
    button_shadow_hover='none',
    
    #change button color here
    button_primary_background_fill='linear-gradient(45deg, *primary_500, *secondary_200)',
    # *primary_300',
    button_primary_background_fill_hover='*button_primary_background_fill',
    button_primary_background_fill_hover_dark='*button_primary_background_fill',
    
    #change button text color here
    button_primary_text_color='*neutral_900',
    
    #change button color here
    button_secondary_background_fill='*secondary_500',
    button_secondary_background_fill_hover='*button_secondary_background_fill',
    button_secondary_background_fill_hover_dark='*button_secondary_background_fill',
    button_secondary_text_color='*neutral_700',
    button_cancel_background_fill_hover='*button_cancel_background_fill',
    button_cancel_background_fill_hover_dark='*button_cancel_background_fill'
)




# %%
instructions="""
# Diabetes Chatbot! 
## Ask a question through audio recording/text. Receive a response. Translate to selected language."""


with gr.Blocks(theme=theme) as app2:
    
    with gr.Row():
        gr.Markdown(instructions)
        
    with gr.Row():
        with gr.Column(variant="panel", scale=1):
            # gr.Markdown("## Step 1: Enter question by recording audio")
            input_audio = gr.Audio(
                label="Click the microphone to record audio",
                type="filepath",
                waveform_options=gr.WaveformOptions(
                    waveform_color="#01C6FF",
                    waveform_progress_color="#0066B4",
                    skip_length=2,
                    show_controls=False,)
            )
            # gr.Markdown("## Step 2: Transcribe audio into text")
            transcribe_button = gr.Button("Transcribe audio", variant="primary")
            # gr.Markdown("## Step 3: Transcribed audio appears here. Or, type your question here.")
            query_text = gr.Textbox(label="Transcribed audio appears here. Or type a question here.")
            # gr.Markdown("## Submit your question")
            submit_button = gr.Button("Submit your question", variant="primary")
            
        with gr.Column(variant="panel", scale=2): 
            response_text = gr.Textbox(label="Chatbot response")
            response_speech = gr.Audio(label="Chatbot response speech",
                                       waveform_options=gr.WaveformOptions(
                                           waveform_color="#01C6FF",
                                           waveform_progress_color="#0066B4",
                                           skip_length=2,
                                           show_controls=False,))
            output_no_audio_message = gr.Textbox(label="message if audio not available")
            
        with gr.Column(variant="panel", scale=2):
            language_dropdown = gr.Dropdown(label="Click the middle of the dropdown bar to select translation language",
                                            choices=list(language_map.keys()), value=default_language, type='value')
            translate_button = gr.Button("Translate the response", variant="primary")
            output_translated = gr.Textbox(label="Translated text")
            output_translated_speech = gr.Audio(label="Translated speech",
                                                waveform_options=gr.WaveformOptions(
                                                    waveform_color="#01C6FF",
                                                    waveform_progress_color="#0066B4",
                                                    skip_length=2,
                                                    show_controls=False,))
            output_no_audio_available = gr.Textbox(label="message if audio not available")
    
    with gr.Row():
        clear_button = gr.Button("Clear All", variant="primary") 
                

    # Audio transcription
    transcribe_button.click(
        fn=transcribe_audio_original,
        inputs=[input_audio],
        outputs=[query_text]
    )
    
    submit_button.click(
        fn=submit_question,
        inputs=[input_audio, query_text, language_dropdown],
        outputs=[response_text, response_speech, output_no_audio_message]
    )
        
    # Translation
    translate_button.click(
        fn=translate_and_speech,
        inputs=[response_text, language_dropdown],
        outputs=[output_translated, output_translated_speech, output_no_audio_available]
    )
        
    #Clearing all inputs and outputs
    clear_button.click(
    fn=clear_inputs,
    inputs=[],
    outputs=[input_audio, query_text, response_text, response_speech, output_translated, output_translated_speech, output_no_audio_message, output_no_audio_available]
    )

app2.launch(show_error=True)



