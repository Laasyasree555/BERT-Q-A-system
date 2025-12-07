import streamlit as st
from transformers import pipeline

# Load a lightweight public QA model
qa_pipeline = pipeline(
    "question-answering",
    model="distilbert-base-cased-distilled-squad"
)

st.title("Question Answering System")

context = st.text_area("Enter the paragraph:")
question = st.text_input("Enter your question:")

if st.button("Get Answer"):
    if context.strip() == "" or question.strip() == "":
        st.warning("Please enter both context and question.")
    else:
        result = qa_pipeline({"context": context, "question": question})
        st.write("### Answer:")
        st.write(result["answer"])
        st.write("**Confidence:**", round(result["score"], 4))
