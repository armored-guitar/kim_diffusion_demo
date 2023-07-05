import json

import torch
import yaml
from diffusers import EulerDiscreteScheduler, StableDiffusionPipeline
from flask import Flask, request, send_file

from src import consts
from src.utils import consctruct_prompt, serve_pil_image

app = Flask(__name__)
pipe = StableDiffusionPipeline.from_ckpt(consts.model_path, safety_checker=None)
pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)
pipe.load_textual_inversion(consts.ti_vector_path, token=consts.KIM_TOKEN)
if torch.cuda.is_available():
    pipe.to("cuda")


@app.route("/generate_image")
def generate_image():
    content = json.load(request.files["json"])
    additional_prompt = content["prompt"]
    num_steps = int(content["num_steps"])
    cfg_scale = float(content["cfg_scale"])
    prompt = consctruct_prompt(additional_prompt)
    image = pipe(
        prompt=prompt,
        negative_prompt=consts.NEGATIVE_PROMPT,
        num_inference_steps=num_steps,
        guidance_scale=cfg_scale,
    ).images[0]
    return serve_pil_image(image)


if __name__ == "__main__":
    with open("./config.yaml", "r") as f:
        yaml_config = yaml.safe_load(f)
    app.config["CACHE_TYPE"] = "null"
    app.run(
        "0.0.0.0",
        port=yaml_config["flask"]["port"],
        debug=False,
        processes=1,
        threaded=False,
    )
