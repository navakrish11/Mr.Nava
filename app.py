import streamlit as st
from llama_cpp import Llama
from huggingface_hub import hf_hub_download
import os

# ---- Set your model path here ----
model_path = hf_hub_download(repo_id="Navakrish11/mr-nava", filename="capybarahermes-2.5-mistral-7b.Q5_K_S.gguf")

# ---- Verify model file existence ----
if not os.path.exists(model_path):
    st.error(f"Model file not found at {model_path}. Please check the repository and filename.")
    st.stop()

# ---- Set up Streamlit UI ----
st.set_page_config(page_title="Mr.Nava (Offline Mode)", page_icon="🧠")
st.title("Mr.Nava (Offline Mode)")
st.markdown("Ask me anything — I’ll answer using a local LLaMA model (CapybaraHermes-2.5-Mistral-7B)")

# ---- Initialize session state ----
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

@st.cache_resource(show_spinner="Loading CapybaraHermes model...")
def load_model():
    try:
        return Llama(model_path=model_path, n_ctx=2048, n_threads=8)
    except Exception as e:
        st.error(f"Failed to load model: {str(e)}")
        st.stop()

# ---- Load the model ----
llm = load_model()

# ---- Handle user input ----
user_input = st.chat_input("Ask a question")
if user_input:
    st.session_state.chat_history.append({"role": "user", "content": user_input})

    # Create a formatted prompt
    messages = st.session_state.chat_history[-5:]  # Recent 5 exchanges
    prompt = ""
    for msg in messages:
        if msg["role"] == "user":
            prompt += f"User: {msg['content']}\n"
        else:
            prompt += f"Assistant: {msg['content']}\n"
    prompt += "Assistant:"

    # Generate response with error handling
    try:
        output = llm(prompt, max_tokens=256, stop=["User:", "Assistant:"], temperature=0.7)
        response = output["choices"][0]["text"].strip()
        st.session_state.chat_history.append({"role": "assistant", "content": response})
    except Exception as e:
        st.error(f"Error generating response: {str(e)}")
        st.session_state.chat_history.append({"role": "assistant", "content": "Sorry, I encountered an error. Please try again."})

# ---- Display chat history ----
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
