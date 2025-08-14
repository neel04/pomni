import os

import keras
import keras_hub
import requests
import streamlit as st
from keras_hub.models import Gemma3CausalLM

# --- App Configuration ---
st.set_page_config(page_title="Pomni Chat", page_icon="✨", layout="wide")


# --- Function to download weights ---
def download_weights(url, dest):
    """Downloads a file from a URL to a destination, showing progress."""
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()  # Raise an exception for bad status codes
        total_size = int(response.headers.get("content-length", 0))

        progress_bar = st.progress(0)
        progress_status = st.empty()

        with open(dest, "wb") as f:
            bytes_downloaded = 0
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                bytes_downloaded += len(chunk)
                if total_size > 0:
                    progress = min(int((bytes_downloaded / total_size) * 100), 100)
                    progress_bar.progress(progress)
                    progress_status.text(
                        f"Downloading... {bytes_downloaded // 1024**2}MB / {total_size // 1024**2}MB"
                    )

        progress_status.text("Download complete.")
        progress_bar.empty()
        return True
    except requests.exceptions.RequestException as e:
        st.error(f"Error downloading weights: {e}")
        return False


# --- Model Loading ---
@st.cache_resource
def load_gemma_model():
    """Loads the fine-tuned Gemma model and caches it."""
    
    # First try loading from HuggingFace
    try:
        st.info("Attempting to load model from HuggingFace: hf://Neel-Gupta/pomni")
        model = keras.saving.load_model("hf://Neel-Gupta/pomni")
        st.success("Successfully loaded model from HuggingFace: hf://Neel-Gupta/pomni")
        
        # Compile the model for inference
        sampler = keras_hub.samplers.TopKSampler(k=5, seed=42)
        model.compile(sampler=sampler)
        return model
        
    except Exception as e:
        st.warning(f"Failed to load from HuggingFace: {e}")
        st.info("Falling back to local weights loading...")
    
    # Fallback to original logic
    preset = "gemma3_instruct_1b"
    weights_path = "/Users/neel/Downloads/finetuned_gemma3_1b.weights.h5"

    # FIX: Static URL
    weights_url = "https://filebin.net/gmmu2zultifcjlgi/finetuned_gemma3_1b.weights.h5"

    if not os.path.exists(weights_path):
        st.info("Model weights not found locally. Downloading from the cloud...")
        if not download_weights(weights_url, weights_path):
            st.error(
                "Failed to download model weights. Please check the URL and your connection."
            )
            st.stop()

    model = Gemma3CausalLM.from_preset(preset, dtype="bfloat16")

    if os.path.exists(weights_path):
        try:
            model.load_weights(weights_path)
            st.success("Successfully loaded fine-tuned model weights.")
        except Exception as e:
            st.error(f"Error loading weights into model: {e}")
            st.info("Proceeding with the base model.")
    else:
        st.warning("Fine-tuned weights not found. Using the base model.")

    sampler = keras_hub.samplers.TopKSampler(k=5, seed=42)
    model.compile(sampler=sampler)
    return model


keras.mixed_precision.set_global_policy("mixed_bfloat16")

gemma_model = load_gemma_model()

# --- Chat Interface ---
st.title("Chat with Fine-tuned Gemma")
st.markdown(
    "This app provides a simple interface to chat with a Gemma model that has been fine-tuned on a custom dataset."
)

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Display chat history
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("What is up?"):
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            # Add system prompt to make the model nice, polite and helpful
            # system_prompt = "You are a helpful, polite, and friendly AI assistant. Please provide clear, concise, and accurate responses while maintaining a warm and respectful tone."
            # full_prompt = f"{system_prompt}\n\nUser: {prompt}\n\nAssistant:"
            full_prompt = f"<p>{prompt}</p>"
            response = gemma_model.generate(full_prompt, max_length=128)

            # Extract just the assistant's response (remove the prompt part)
            if full_prompt in response:
                clean_response = response[len(full_prompt) :].strip()
            else:
                clean_response = response.strip()

            st.markdown(clean_response)
    st.session_state.chat_history.append({
        "role": "assistant",
        "content": clean_response,
    })
