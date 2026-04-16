
def main():
    import streamlit as st
    from langchain.llms import HuggingFaceHub
    from langchain.vectorstores import FAISS
    from langchain.embeddings.huggingface import HuggingFaceEmbeddings
    from langchain.text_splitter import CharacterTextSplitter
    import os
    import json
    import sounddevice as sd
    import numpy as np
    from transformers import pipeline

    # Set HuggingFace API Token
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"  # Replace with your API token

    # Whisper model for voice transcription
    whisper = pipeline("automatic-speech-recognition", model="openai/whisper-medium", device=0)

    # Voice recording duration
    DURATION = 5

    # Record and transcribe function
    def record_and_transcribe():
        """
        Record audio from the microphone and transcribe it using Whisper.
        """
        audio = sd.rec(int(DURATION * 16000), samplerate=16000, channels=1, dtype="float32")
        sd.wait()  # Wait until recording is finished
        audio = np.squeeze(audio)  # Remove single-dimensional entries

        transcription = whisper(audio)
        return transcription["text"]

    # Load Llama model
    def load_llama_model(repo_id, temperature=0.7, max_length=150):
        return HuggingFaceHub(
            repo_id=repo_id,
            model_kwargs={
                "temperature": temperature,
                "max_length": max_length,
            }
        )

    # Generate Llama response with optional context
    def generate_llama_response(llm, user_input, context=""):
        prompt = (
            f"You are a friendly and helpful assistant named Llama. "
            f"Use the following context to answer the question. If the answer is not present in the context, respond with 'I don't know.'\n\n"
            f"Context:\n{context}\n\n"
            f"User: {user_input}\n"
            f"Llama:"
        )
        full_response = llm(prompt).strip()
        if "Llama:" in full_response:
            response = full_response.split("Llama:")[1].strip()
        else:
            response = full_response
        return response

    # Process JSON file into a searchable knowledge base
    def process_json_to_knowledge_base(json_path):
        try:
            with open(json_path, "r") as f:
                data = json.load(f)
            combined_text = []
            for key, value in data.items():
                for entry in value:
                    if isinstance(entry, str):
                        combined_text.append(entry)
                    elif isinstance(entry, dict) and "caption" in entry:
                        combined_text.append(entry["caption"])
            full_text = "\n".join(combined_text).strip()

            # Split text into chunks for embeddings
            text_splitter = CharacterTextSplitter(
                separator="\n",
                chunk_size=2000,
                chunk_overlap=200,
                length_function=len,
            )
            chunks = text_splitter.split_text(full_text)

            # Generate embeddings and create FAISS knowledge base
            embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
            knowledge_base = FAISS.from_texts(chunks, embeddings)
            return knowledge_base
        except Exception as e:
            st.error(f"Error processing JSON file: {e}")
            return None

    # Load Llama model
    llama_llm = load_llama_model(repo_id="meta-llama/Meta-Llama-3-8B-Instruct")

    # Load JSON content into a knowledge base
    json_path = "/Users/saivaruntanjoreraghavendra/Documents/json/scraped_content.json"
    knowledge_base = process_json_to_knowledge_base(json_path)
    if not knowledge_base:
        st.error("Failed to load knowledge base from JSON file.")
        return

    # Inject Custom Font and Chat Bubble Styles
    custom_css = """
    <style>
        html, body, [class*="css"] {
            font-family: 'Brandon Grotesque', 'brandon-grotesque', Helvetica, sans-serif;
        }
        .chat-message {
            margin: 10px 0;
            padding: 10px;
            border-radius: 5px;
            width: fit-content;
            max-width: 80%;
        }
        .user-message {
            background-color: #dcf8c6;
            color: black;
            align-self: flex-end;
            text-align: right;
        }
        .llama-message {
            background-color: #ebebeb;
            color: black;
            align-self: flex-start;
            text-align: left;
        }
        .chat-container {
            display: flex;
            flex-direction: column;
            gap: 10px;
        }
    </style>
    """
    st.markdown(custom_css, unsafe_allow_html=True)

    # Initialize chat history and transcription in session state
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []

    if "voice_input" not in st.session_state:
        st.session_state["voice_input"] = ""

    if "text_input" not in st.session_state:
        st.session_state["text_input"] = ""

    # Sidebar for recording and transcription
    st.sidebar.title("Voice input by OpenAi")
    if st.sidebar.button("Record Voice"):
        transcription = record_and_transcribe()
        st.session_state["voice_input"] = transcription

    st.sidebar.text_input(
        "Transcribed Text:",
        value=st.session_state["voice_input"],
        key="voice_text_input",
        placeholder="Transcribed text will appear here.",
    )

    # Callback to process user input
    def handle_user_input():
        user_input = st.session_state["text_input"]
        if user_input.strip():
            # Retrieve context from knowledge base
            retrieved_docs = knowledge_base.similarity_search(user_input, k=5)
            context = "\n".join([doc.page_content for doc in retrieved_docs]) if retrieved_docs else "No relevant context found."

            # Generate Llama's response
            llama_response = generate_llama_response(llama_llm, user_input, context)

            # Append new conversation to chat history
            st.session_state["chat_history"].append(
                {"user": user_input, "llama": llama_response}
            )
            # Clear the input box after processing
            st.session_state["text_input"] = ""

    # Chat UI
    st.title("Chat with Llama")

    # Display chat history
    chat_container = st.container()
    with chat_container:
        for chat in st.session_state["chat_history"]:
            st.markdown(
                f'<div class="chat-container">'
                f'<div class="chat-message user-message">{chat["user"]}</div>'
                f'<div class="chat-message llama-message">{chat["llama"]}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

    # Input area for manual text input
    st.markdown("---")
    st.text_input(
        "Type your message:",
        key="text_input",
        placeholder="Ask anything...",
        on_change=handle_user_input,  # Triggers processing on Enter
    )


#Reference:
# Streamlit
# Streamlit Inc. (2019). Streamlit: The fastest way to build and share data apps. Retrieved from https://streamlit.io

# LangChain
# LangChain. (2023). LangChain: Building applications with LLMs through composability. Retrieved from https://www.langchain.com

# FAISS-CPU
# Johnson, J., Douze, M., & Jégou, H. (2017). FAISS: A library for efficient similarity search and clustering of dense vectors. Retrieved from https://github.com/facebookresearch/faiss

# Hugging Face Hub
# Hugging Face. (2020). Hugging Face Hub: The AI community's repository. Retrieved from https://huggingface.co/hub

# Sentence Transformers
# Reimers, N., & Gurevych, I. (2019). Sentence-BERT: Sentence embeddings using Siamese BERT-networks. Retrieved from https://www.sbert.net

# Transformers
# Hugging Face. (2019). Transformers: State-of-the-art natural language processing for PyTorch and TensorFlow 2.0. Retrieved from https://huggingface.co/transformers

# NumPy
# Harris, C. R., Millman, K. J., van der Walt, S. J., et al. (2020). Array programming with NumPy. Nature, 585(7825), 357–362. doi: https://doi.org/10.1038/s41586-020-2649-2

# Pandas
# McKinney, W. (2010). Data structures for statistical computing in Python. Proceedings of the 9th Python in Science Conference, 51–56. doi: https://doi.org/10.25080/Majora-92bf1922-00a

