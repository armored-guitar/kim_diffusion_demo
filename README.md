# Photorealistic generator of photos of Kim Kardashian
## What used
There are two implemented servers: frontend and backend. Frontend service uses streamlit to develop an web app. Backend uses flask and diffusers to generate images. For personalized Kim generation I used [text inversion](https://arxiv.org/abs/2208.01618) technique. Token was taken from [civitai.com](https://civitai.com/models/23630/kim-kardashian). Several  photorealistic checkpoints of stable diffusion were used but best results are get with [this one](https://civitai.com/models/4201?modelVersionId=105674). Here is an example of a generation
![Example_image](https://github.com/armored-guitar/kim_diffusion_demo/blob/0bbc775987484eb2ab5985d63eb3c5603c553c4b/imgs/0_realistic_vision.png?raw=true)  

## Deployment
you can either build image from scratch using:
```sh
docker compose build && docker compose up -d
```
or use prebuilt images by simply running 
```sh
docker compose up -d 
```
In both cases web app will be available under localhost:8501

Approximate VRAM usage is 6GB. Docker image is build only supporting gpu usage.
