import io
import json

import yaml
from flask import Flask, jsonify, request, send_file

from src import consts
from src.utils import consctruct_prompt, get_model

app = Flask(__name__)
pipe = get_model()


def run_inference(additional_prompt: str, num_steps: int, cfg_scale: float):
    prompt = consctruct_prompt(additional_prompt)
    image = pipe(
        prompt=prompt,
        negative_prompt=consts.NEGATIVE_PROMPT,
        num_inference_steps=num_steps,
        guidance_scale=cfg_scale,
    ).images[0]
    img_data = io.BytesIO()
    image.save(img_data, "PNG")
    img_data.seek(0)
    return img_data


@app.route("/generate_image")
def generate_image():
    content = json.load(request.files["json"])
    img_data = run_inference(content["prompt"], content["num_steps"], content["cfg"])
    return send_file(img_data, mimetype="image/png")


@app.route("/healthy")
def get_health_status():
    resp = jsonify(health="healthy")
    resp.status_code = 200
    return resp


if __name__ == "__main__":
    with open("./config.yaml", "r") as f:
        yaml_config = yaml.safe_load(f)
    app.config["CACHE_TYPE"] = "null"
    app.run(
        "0.0.0.0",
        port=yaml_config["flask"]["port"],
        debug=False,
    )
