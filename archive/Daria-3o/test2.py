import os
import json
import streamlit as st
from langchain.vectorstores import FAISS
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.llms import HuggingFaceHub
from langchain.text_splitter import CharacterTextSplitter

def main():
    # Initialize session state variables
    if "json_content" not in st.session_state:
        st.session_state["json_content"] = None
    if "knowledge_base" not in st.session_state:
        st.session_state["knowledge_base"] = None
    if "user_question" not in st.session_state:
        st.session_state["user_question"] = ""
    if "retrieved_docs" not in st.session_state:
        st.session_state["retrieved_docs"] = None
    if "answers" not in st.session_state:
        st.session_state["answers"] = {}

    # Set HuggingFace API Token
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"

    def extract_json_content(json_path):
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
            combined_text = []
            for key, value in data.items():
                for entry in value:
                    if isinstance(entry, str):
                        combined_text.append(entry)
                    elif isinstance(entry, dict) and "caption" in entry:
                        combined_text.append(entry["caption"])
            return "\n".join(combined_text).strip()
        except Exception as e:
            st.error(f"Error reading JSON file: {e}")
            return None

    def process_content(content):
        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=2000,
            chunk_overlap=200,
            length_function=len,
        )
        chunks = text_splitter.split_text(content)
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        knowledge_base = FAISS.from_texts(chunks, embeddings)
        return knowledge_base

    def load_model(repo_id, temperature=0.1, max_length=800):
        return HuggingFaceHub(
            repo_id=repo_id,
            model_kwargs={
                "temperature": temperature,
                "max_length": max_length,
            }
        )

    def generate_answer(llm, context, user_question):
        prompt = (
            f"Use the following context to answer the question strictly based on the provided information. "
            f"If the answer is not present in the context, respond with 'I don't know.'\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {user_question}\n"
            f"Answer:"
        )
        full_answer = llm(prompt).strip()
        first_answer = full_answer.split("Answer:")[1].strip().split("\n")[0]
        return first_answer

    def get_image_url_for_keyword(json_path, keyword):
        """
        Extracts the image URL and its source from the JSON file based on a partial keyword match in the caption.
        """
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
            # Normalize the keyword for case-insensitive matching
            keyword = keyword.lower().strip()

            for key, value in data.items():
                for entry in value:
                    if isinstance(entry, dict) and "caption" in entry:
                        # Normalize the caption
                        normalized_caption = entry["caption"].lower()

                        # Check if the keyword is in the normalized caption
                        if keyword in normalized_caption:
                            image_url = entry["images"][0] if entry.get("images") else None
                            source = entry["caption"]
                            return image_url, source
            # If no match is found
            return None, "No source available"
        except Exception as e:
            # Handle errors gracefully
            return None, f"Error: {str(e)}"


    # Main Page Content
    st.title("DARIA-3o: Chatbot for InfoTunnel")

    # JSON file path
    json_path = "/Users/saivaruntanjoreraghavendra/Documents/json/scraped_content.json"

    # Load JSON content
    if st.session_state["json_content"] is None:
        st.session_state["json_content"] = extract_json_content(json_path)

    if not st.session_state["json_content"]:
        st.error("Failed to load JSON file. Please check the file path.")
    else:
        if st.session_state["knowledge_base"] is None:
            st.session_state["knowledge_base"] = process_content(st.session_state["json_content"])
        st.success("Knowledge Base loaded successfully!")

        # Handle general user questions
        user_question = st.text_input("Ask your question:", value=st.session_state["user_question"])
        if user_question:
            st.session_state["user_question"] = user_question
            with st.spinner("Generating answers..."):
                retrieved_docs = st.session_state["knowledge_base"].similarity_search(user_question, k=5)
                if retrieved_docs:
                    st.session_state["retrieved_docs"] = retrieved_docs
                    context = "\n".join([doc.page_content for doc in retrieved_docs])

                    microsoft_llm = load_model(repo_id="microsoft/Phi-3.5-mini-instruct")
                    microsoft_answer = generate_answer(microsoft_llm, context, user_question)

                    llama_llm = load_model(repo_id="meta-llama/Meta-Llama-3-8B-Instruct")
                    llama_answer = generate_answer(llama_llm, context, user_question)

                    st.session_state["answers"] = {
                        "Microsoft": microsoft_answer,
                        "Llama": llama_answer
                    }

        # Display results
        if st.session_state["answers"]:
            st.subheader("Answer from Microsoft Phi2")
            st.write(st.session_state["answers"]["Microsoft"])

            st.subheader("Answer from Meta LLaMA")
            st.write(st.session_state["answers"]["Llama"])

        if st.session_state["retrieved_docs"]:
            st.subheader("Sources")
            with st.expander("View Sources"):
                for doc in st.session_state["retrieved_docs"]:
                    st.write(f"- {doc.page_content[:200]}...")

        # Add image URL extraction to sidebar
        st.sidebar.title("Image Search")
        keyword_query = st.sidebar.text_input("Enter a keyword for image search:")
        if keyword_query:
            with st.spinner("Searching for image URL..."):
                # Unpack the image URL and source
                image_url, source = get_image_url_for_keyword(json_path, keyword_query)
                if image_url:
                    # Display the image and source
                    st.sidebar.image(image_url)
                    st.sidebar.markdown(f"{source}")
                else:
                    st.sidebar.error("No image URL found for the given keyword.")


if __name__ == "__main__":
    main()


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

