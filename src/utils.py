import torch
from diffusers import EulerDiscreteScheduler, StableDiffusionPipeline

from src.consts import KIM_TOKEN, PROMPT, model_repo, ti_vector_path


def consctruct_prompt(additional_prompt: str) -> str:
    prompt = PROMPT.format(token=KIM_TOKEN, prompt_details=additional_prompt)
    return prompt


def get_model():
    pipe = StableDiffusionPipeline.from_pretrained(model_repo, safety_checker=None)
    pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)
    pipe.load_textual_inversion(ti_vector_path, token=KIM_TOKEN)
    if torch.cuda.is_available():
        pipe.to("cuda")
    return pipe
