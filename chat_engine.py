from loguru import logger
from langchain_community.llms import Ollama
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_history_aware_retriever
from langchain_community.vectorstores import FAISS
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    logger.info(f"Extracted text from {len(pdf_docs)} PDF(s)")
    return text

def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    logger.info(f"Split text into {len(chunks)} chunks")
    return chunks

def get_vectorstore(chunks):
    # embeddings = OpenAIEmbeddings()
    try:
        embeddings = HuggingFaceEmbeddings(model_name="all-mpnet-base-v2")
    except Exception as e:
        logger.error(f"Error loading embeddings: {e}")
    vectorstore = FAISS.from_texts(chunks, embeddings)
    logger.info(f"Created vector store with {len(chunks)} chunks")
    return vectorstore

def build_conversational_rag_chain(vectorstore):
    """Build a conversation chain with memory and retrieval"""
    
    # Initialize free local LLM (Ollama)
    llm = Ollama(model="mistral")
    
    # First we need a prompt that can pass the chat history into the retriever
    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
        ("user", "Given the above conversation, generate a search query to look up in order to get information relevant to the conversation")
    ])
    
    # Create the history-aware retriever
    retriever = vectorstore.as_retriever()
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )
    
    # Create the prompt for answering with context
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", """Answer the user's questions based on the below context. 
         If you don't know the answer, just say you don't know. 
         Keep the answer concise and helpful.\n\nContext: {context}"""),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
    ])
    
    # Create the question answering chain
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    # Combine with retriever
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
    return rag_chain
