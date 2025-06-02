
import gradio as gr
from diffusers import StableDiffusionPipeline
import torch

model_id = "runwayml/stable-diffusion-v1-5"
pipe = StableDiffusionPipeline.from_pretrained(
    model_id,
    torch_dtype=torch.float32
).to("cpu")

def generate_image(prompt):
    image = pipe(prompt).images[0]
    return image

interface = gr.Interface(
    fn=generate_image,
    inputs=gr.Textbox(label="Enter your prompt"),
    outputs="image",
    title="🎨 Real-Time AI Art Generator",
    description="Turn your imagination into visuals using Stable Diffusion!"
)

interface.launch()
