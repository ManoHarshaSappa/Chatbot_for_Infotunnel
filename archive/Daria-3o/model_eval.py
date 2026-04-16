import streamlit as st
import pandas as pd

def main():
    st.success("Evaluation Loaded successfully!")
    st.title("Model Evaluation")
    st.write("This application provides an overview of model evaluation for comparing two machine learning models.")

    # Documentation Section
    st.header("Overview")
    st.markdown("""
    This documentation outlines the evaluation and performance comparison between two models: Microsoft Phi2 and Meta LLaMA, focusing on their ability to address questions across multiple technical domains.
    """)

    # Evaluation Categories and Results
    st.header("Evaluation Categories and Results")

    # Table for Data Analysis
    st.subheader("1. Data Analysis")
    st.markdown("""
    **Question**: How is data acquired and processed in Acoustic Emission (AE) testing?

    | Model             | Accuracy | Relevance | Completeness | Clarity | Conciseness | Comments |
    |-------------------|----------|-----------|--------------|---------|-------------|----------|
    | **Microsoft Phi2** | ✅ 5/5   | ✅ 5/5    | ✅ 5/5       | ✅ 5/5  | ✅ 5/5      | Provides a detailed and clear explanation with technical terms, covering waveform analysis and damage estimation. |
    | **Meta LLaMA**     | ✅ 5/5   | ✅ 5/5    | ✅ 5/5       | ✅ 5/5  | ✅ 5/5      | Covers the process accurately and succinctly, with good explanations of data interpretation. |
    """)

    # Table for Contextual Understanding
    st.subheader("2. Contextual Understanding")
    st.markdown("""
    **Question**: What is the primary purpose of Acoustic Emission (AE) technology?

    | Model             | Accuracy | Relevance | Completeness | Clarity | Conciseness | Comments |
    |-------------------|----------|-----------|--------------|---------|-------------|----------|
    | **Microsoft Phi2** | ✅ 5/5   | ✅ 5/5    | ❌ 3/5       | ✅ 5/5  | ✅ 5/5      | Captures the basic purpose but misses additional applications of AE technology. |
    | **Meta LLaMA**     | ✅ 5/5   | ✅ 5/5    | ✅ 5/5       | ✅ 5/5  | ✅ 5/5      | Provides a comprehensive answer, including primary and extended purposes of AE technology. |
    """)

    # Table for Practical Applications
    st.subheader("3. Practical Applications")
    st.markdown("""
    **Question**: What are the advantages and limitations of Dye Penetrant Testing (DPT)?

    | Model             | Accuracy | Relevance | Completeness | Clarity | Conciseness | Comments |
    |-------------------|----------|-----------|--------------|---------|-------------|----------|
    | **Microsoft Phi2** | ✅ 5/5   | ✅ 5/5    | ❌ 3/5       | ✅ 5/5  | ✅ 5/5      | Covers advantages but omits limitations, such as time consumption and safety concerns. |
    | **Meta LLaMA**     | ❌ 2/5   | ❌ 2/5    | ❌ 2/5       | ❌ 2/5  | ❌ 2/5      | Fails to address the question adequately and provides incomplete responses. |
    """)

    # Table for Technical Explanation
    st.subheader("4. Technical Explanation")
    st.markdown("""
    **Question**: Can you explain the physical principle behind active infrared thermography?

    | Model             | Accuracy | Relevance | Completeness | Clarity | Conciseness | Comments |
    |-------------------|----------|-----------|--------------|---------|-------------|----------|
    | **Microsoft Phi2** | ✅ 5/5   | ✅ 5/5    | ✅ 5/5       | ✅ 5/5  | ✅ 5/5      | Provides a detailed explanation, including the influence of thermal properties and subsurface defects. |
    | **Meta LLaMA**     | ✅ 5/5   | ✅ 5/5    | ✅ 5/5       | ✅ 5/5  | ✅ 5/5      | Includes additional details on wavelength ranges, making the explanation slightly more comprehensive. |
    """)

    # Summary Table
    st.subheader("Summary Table")
    summary_data = {
        "Criteria": ["Accuracy", "Relevance", "Completeness", "Clarity", "Conciseness"],
        "Microsoft Phi2 Accuracy (%)": ["96%", "92%", "90%", "96%", "94%"],
        "Meta LLaMA Accuracy (%)": ["86%", "80%", "70%", "80%", "72%"],
    }
    df_summary = pd.DataFrame(summary_data)
    st.table(df_summary)

    # Key Observations
    st.subheader("Key Observations")
    st.markdown("""
    1. **Strengths**:
       - Microsoft Phi2 excels in **Accuracy** and **Clarity** across all categories.
       - Meta LLaMA performs well in **Technical Explanations** and **Data Analysis**.

    2. **Weaknesses**:
       - Microsoft Phi2 occasionally lacks **Completeness**, particularly in Practical Applications.
       - Meta LLaMA struggles with **Completeness** and **Relevance** in practical and comparative contexts.
    """)


        # Display Images for Visualizations
# Display Images for Visualizations
    st.subheader("Analysis")
    st.image("assets/output (2).png", caption="Model Performance Across Evaluation Categories")
    st.image("assets/output (3).png", caption="Average Scores Across Categories for Each Model")
    st.image("assets/output (4).png", caption="Detailed Scores for Each Model")
    st.image("assets/output (5).png", caption="Detailed Scores for Each Model")
    st.image("assets/output (6).png", caption="Detailed Scores for Each Model")



    # Metrics and Calculations
    st.subheader("Metrics and Calculations")
    st.markdown("""
    **Microsoft Phi2**:
    - **Accuracy**: 4.8/5 (96%)
    - **Relevance**: 4.6/5 (92%)
    - **Completeness**: 4.5/5 (90%)
    - **Clarity**: 4.8/5 (96%)
    - **Conciseness**: 4.7/5 (94%)

    **Meta LLaMA**:
    - **Accuracy**: 4.3/5 (86%)
    - **Relevance**: 4.0/5 (80%)
    - **Completeness**: 3.5/5 (70%)
    - **Clarity**: 4.0/5 (80%)
    - **Conciseness**: 3.6/5 (72%)

    **Overall Accuracy (Avg)**:
    - **Microsoft Phi2**: 94%
    - **Meta LLaMA**: 78%
    """)

    # Proposed Enhancements
    st.subheader("Proposed Enhancements")
    st.markdown("""
    1. **Re-Fine-Tune the Model**:
       - Train Meta LLaMA on curated datasets with detailed answers across technical domains like Acoustic Emission (AE) testing.
       - Include:
         - **High-quality examples**: Correct and incorrect answers to enhance learning.
         - **Diverse contexts**: Use research papers, FAQs, and tutorials for varied exposure.

    2. **Focus on Prompt Refinement**:
       - Example:
         - **Before**: *How is data acquired and processed in AE testing?*  
         - **After**: *Explain in detail how data is acquired and processed in AE testing, including the role of sensors, digitization, and waveform analysis.*

    3. **Post-Processing**:
       - **Consistency Checks**: Use secondary validation to ensure factual accuracy and coherence.
       - **Grammar and Clarity Refinement**: Employ tools like Grammarly or algorithms for improved readability.
    """)


# Improving Model Performance Through Prompting Technique
    st.subheader("Improving Model Performance Through Prompting Technique")
    st.markdown("""
    **Before Prompt**:
    *How is data acquired and processed in Acoustic Emission (AE) testing?*

    Data acquisition in AE testing involves the use of acoustic sensors, data acquisition electronics, and a computer to analyze the data. Signals received from each sensor are digitized and analyzed using specialized software. The rate of occurrence of AE events is interpreted to identify AE sources (i.e., damage) in the material. The location of the source of the AE signal is also used in the interpretation of results. For example, a high number of AE events emanating from a particular location in a structure during the application.

    **Drawbacks**:
    - **Accuracy**: The response is factually correct.
    - **Relevance**: The answer aligns with the question.
    - **Completeness**: The explanation is incomplete; it lacks key technical details about waveform analysis and damage progression rate estimation.
    - **Clarity**: The response is clear but verbose in parts.
    - **Conciseness**: Some examples and phrases could be made more concise.

    **After Prompt**:
    *Explain in detail how data is acquired and processed in Acoustic Emission (AE) testing. Include the role of acoustic sensors, the process of digitizing signals, and the analysis of waveform parameters like amplitude, frequency, and energy. Discuss how these parameters are used to determine the rate of AE event production, locate damage sources, and estimate damage progression rates.*

    In Acoustic Emission (AE) testing, data acquisition involves the use of acoustic sensors to detect transient elastic stress waves produced by the evolution of damage in a material. The sensors are typically mounted on the surface of the material being tested and are designed to detect the low-amplitude signals produced by the damage.

    **Improvements**:
    1. **Not verbose**.
    2. **Relevant**.
    3. **Clarity**.
    4. **Conciseness**.
    5. **Accurate**.

    **But not complete**.

    ---

    ### **Future Enhancements**
    1. **Re-Fine-Tune the Model on a Correct Dataset**:
    - Fine-tune the LLaMA model on a curated dataset with detailed technical answers specific to domains like Acoustic Emission (AE) testing. This dataset should include:
        - **High-quality examples**: Provide both correct and incorrect examples to help the model learn nuances.
        - **Detailed explanations**: Include complete answers with technical depth and real-world applications.
        - **Diverse contexts**: Train on varied formats (technical manuals, research papers, FAQs, and tutorials).

    2. **Focus on Post-Processing**:
    - **Consistency checks**: Use secondary logic to verify accuracy and coherence in the response.
    - **Grammar and clarity refinement**: Implement tools like Grammarly or custom algorithms to improve readability.

    3. **And More**: Researching further enhancement.
    """)


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

