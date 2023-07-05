from flask import send_file

from src.consts import KIM_TOKEN, PROMPT


def consctruct_prompt(additional_prompt: str) -> str:
    prompt = PROMPT.format(token=KIM_TOKEN, prompt_details=additional_prompt)
    return prompt


from io import StringIO


def serve_pil_image(pil_img):
    img_io = StringIO()
    pil_img.save(img_io, "JPEG", quality=70)
    img_io.seek(0)
    return send_file(img_io, mimetype="image/jpeg")
