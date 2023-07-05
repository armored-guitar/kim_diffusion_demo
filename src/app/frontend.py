import json
import os

import requests
import streamlit as st
import yaml

address = 10


@st.cache_data
def get_config():
    with open(os.path.join("./config.yaml"), "r") as f:
        yaml_config = yaml.safe_load(f)
    return yaml_config


def init_state():
    attrs = ["finished", "submitted", "request_path"]
    for attr in attrs:
        if attr not in st.session_state:
            setattr(st.session_state, attr, None)


def main():
    config = get_config()
    address = f"http://{config['flask']['flask_ip']}:{config['flask']['flask_port']}"
    st.title("Stable Diffusion Kim Demo")
    st.write("Please enter your prompt and then press submit button")
    init_state()
    with st.sidebar:
        with st.form("inference_config"):
            st.text_input(
                'Enter your prompt. Please don`t use "Kim Kardashian" or connected phrases to prevent model from failing.'
                "Also don`t include tags because they`re already included. You can describe style of cloths "
                "and background, pose is already predefined",
                value="",
                key="prompt",
            )
            st.slider(
                "how much attention to pay to the prompt",
                min_value=3,
                max_value=10,
                value=6,
                step=1,
                key="cfg",
                label_visibility="visible",
            )
            st.slider(
                "number of diffusion steps to be taken",
                min_value=20,
                max_value=40,
                value=25,
                step=1,
                key="num_steps",
                label_visibility="visible",
            )
            submitted = st.form_submit_button("Submit request")
            if submitted:
                st.session_state.submitted = submitted

        if st.button("Restart"):
            st.session_state.clear()
            init_state()

    if st.session_state.submitted:
        st.session_state.request_id = "id"

        config_json = {
            "prompt": st.session_state.prompt,
            "cfg": st.session_state.cfg,
            "num_steps": st.session_state.num_steps,
        }
        files = {
            "json": json.dumps(config_json),
        }

        with requests.Session() as s:
            s.trust_env = False
            r = s.get(
                os.path.join(address, "generate_image"),
                files=files,
                stream=True,
                headers={"Connection": "close"},
            )
            if r.ok:
                st.session_state.finished = True
            else:
                raise ValueError(r)

        if st.session_state.finished:
            st.image(r.content)
