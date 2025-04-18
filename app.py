# Streamlit is an open-source Python library that makes it super easy to build interactive web apps for data science and machine learning projects
import streamlit as st
from chat_engine import get_pdf_text, get_text_chunks, get_vectorstore, build_conversational_rag_chain
def handle_userinput(user_question):
    response = st.session_state.rag_chain.invoke({                  # Invoke the chain
        "input": user_question,                                     # Sends the user input (user_question) to the RAG chain.
        "chat_history": st.session_state.chat_history               # Also sends the existing chat_history so the chain can consider previous context.
    })
    st.session_state.chat_history.extend([                          # Update chat history, extend() adds both messages in one go.
        {"role": "user", "content": user_question},                 # The user's message.
        {"role": "assistant", "content": response["answer"]}        # The assistant’s (AI) response.
    ])
    for message in st.session_state.chat_history:                   # Display chat history,Loops through the entire chat history.
        if message["role"] == "user":
            st.text_area("User", message["content"], height=100)
        else:
            st.text_area("Assistant", message["content"], height=100)

def main():
    st.set_page_config(page_title="chat with pdfs", page_icon=":guardsman:")
    # Streamlit apps run top-to-bottom every time a user interacts with a widget, so without session_state, variables would reset on every interaction.
    if "rag_chain" not in st.session_state:
        st.session_state.rag_chain = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    st.header("Chat with your PDFs")
    user_question=st.text_input("Enter your question here:")
    if user_question:
        handle_userinput(user_question)                     # function to handle user input and get response from conversation chain
    with st.sidebar:
        st.subheader("your docs")
        pdf_docs=st.file_uploader("Upload your PDF files", type="pdf", accept_multiple_files=True)
        if st.button("process"):
            with st.spinner("processing..."):
                raw_text = get_pdf_text(pdf_docs)           # function to extract text from pdfs
                text_chunks = get_text_chunks(raw_text)     # function to split text into chunks
                vectorstore=get_vectorstore(text_chunks)    # function to create vector store
                st.session_state.rag_chain = build_conversational_rag_chain(vectorstore)
                st.success("Documents processed and ready for questions!")
if __name__ == '__main__':
    main()
    