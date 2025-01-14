import gradio as gr
from transformers import pipeline, Pipeline


languages = ["greek", "galician"]
model_ids = [
    "kostissz/whisper-tiny-gl",
    "kostissz/whisper-tiny-el",
    "openai/whisper-small",
    "openai/whisper-tiny",
]


def load_model(model_id: str, language: str) -> Pipeline:
    return pipeline(
        "automatic-speech-recognition",
        model=model_id,
        generate_kwargs={"language": language},
    )


def transcribe(pipe: Pipeline, audio: gr.Audio) -> str:
    text = pipe(audio)["text"]
    return text


def setup_gradio_demo():
    with gr.Blocks() as demo:
        selected_lang = gr.Dropdown(
            choices=languages, value=None, label="Select a language"
        )
        selected_model = gr.Dropdown(
            choices=model_ids, value=None, label="Select a model"
        )
        audio_input = gr.Audio(
            sources="microphone", type="filepath", label="Record a message"
        )
        transcribe_button = gr.Button("Transcribe")
        transcribe_output = gr.Text(label="Output")

        model = gr.State()

        selected_model.change(
            fn=load_model, inputs=[selected_model, selected_lang], outputs=model
        )
        transcribe_button.click(
            fn=transcribe, inputs=[model, audio_input], outputs=transcribe_output
        )

    demo.launch()


if __name__ == "__main__":
    setup_gradio_demo()
