import gradio as gr
import torch
import pytube as pt
from transformers import pipeline
from huggingface_hub import model_info


transcribe_model_ckpt = "openai/whisper-small"
lang = "en"
device = 0 if torch.cuda.is_available() else "cpu"
transcribe_pipe = pipeline(
    task="automatic-speech-recognition",
    model=transcribe_model_ckpt,
    chunk_length_s=30,
    device=device,
)
transcribe_pipe.model.config.forced_decoder_ids = transcribe_pipe.tokenizer.get_decoder_prompt_ids(language=lang, task="transcribe")

def _return_yt_html_embed(yt_url):
    video_id = yt_url.split("?v=")[-1]
    HTML_str = (
        f'<center> <iframe width="500" height="320" src="https://www.youtube.com/embed/{video_id}"> </iframe>'
        " </center>"
    )
    return HTML_str

def yt_transcribe(yt_url):
    yt = pt.YouTube(yt_url)
    html_embed_str = _return_yt_html_embed(yt_url)
    stream = yt.streams.filter(only_audio=True)[0]
    stream.download(filename="audio.mp3")

    text = transcribe_pipe("audio.mp3")["text"]

    return html_embed_str, text

qa_model_ckpt = "deepset/tinyroberta-squad2"
qa_pipe = pipeline('question-answering', model=qa_model_ckpt, tokenizer=qa_model_ckpt)

def get_answer(query,context):
    QA_input = {
    'question': query,
    'context': context
    }
    res = qa_pipe(QA_input)["answer"]
    return res



with gr.Blocks() as demo:
    gr.Markdown("<h1><center>Youtube-QA</center></h1>")
    gr.Markdown("<h3>Ask questions from youtube video, it takes sometime to run so grabbing a coffee is a good option ☕️</h3>")
    gr.Markdown("""
                Youtube-audio --> openai-whisper --> Transcription + Query --> QA-model --> Answer
                """)
    with gr.Row():
        with gr.Column():
            in_yt = gr.inputs.Textbox(lines=1, placeholder="Enter Youtube URL", label="YouTube URL")
            transcribe_btn = gr.Button("Transcribe")
        with gr.Column():
            out_yt_html = gr.outputs.HTML()
            out_yt_text = gr.Textbox(label="Transcription")
        with gr.Column():
            in_query = gr.Textbox(lines=1, placeholder="What's your Question", label="Query")
            ans_btn = gr.Button("Answer")
            out_query = gr.outputs.Textbox(label="Answer")
            
    
    transcribe_btn.click(fn=yt_transcribe, inputs=in_yt, outputs=[out_yt_html,out_yt_text])
    ans_btn.click(fn=get_answer, inputs=[in_query,out_yt_text], outputs=out_query)

demo.launch()