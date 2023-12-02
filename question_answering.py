import streamlit as st
from transformers import pipeline, AutoTokenizer

@st.cache(allow_output_mutation=True)
def load_qa_model():
    return pipeline('question-answering', model="deepset/roberta-base-squad2")

def run_question_answering(question, context, nlp):
    result = nlp({'question': question, 'context': context})
    return result['answer']

def main():
    st.title("Question Answering with Streamlit")

    # Load the question answering model
    nlp = load_qa_model()

    # User input for question and context
    question = st.text_input("Enter your question:")
    context = st.text_area("Enter the context:")

    if st.button("Get Answer"):
        if question and context:
            st.write("Answer:")
            result = run_question_answering(question, context, nlp)
            st.write(result)
        else:
            st.warning("Please provide both a question and context.")

if __name__ == "__main__":
    main()
