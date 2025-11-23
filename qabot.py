from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams
from langchain_ibm import WatsonxLLM #, WatsonxEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain.chains import RetrievalQA
from langchain_community.embeddings import HuggingFaceEmbeddings
import gradio as gr
import warnings

# (Optional) Hide warnings to avoid clutter
def warn(*args, **kwargs):
    pass
warnings.warn = warn
warnings.filterwarnings('ignore')

# 🔐 Watson Credentials (placeholders — replace with real values)
WATSONX_API_KEY = "YOUR_API_KEY_HERE"
WATSONX_URL = "https://us-south.ml.cloud.ibm.com"
WATSONX_PROJECT_ID = "YOUR_PROJECT_ID_HERE"

## LLM
def get_llm():
    model_id = 'ibm/granite-3-2-8b-instruct'
    parameters = {
        GenParams.MAX_NEW_TOKENS: 256,
        GenParams.TEMPERATURE: 0.5
    }

    watsonx_llm = WatsonxLLM(
        model_id=model_id,
        api_key=WATSONX_API_KEY,
        url=WATSONX_URL,
        project_id=WATSONX_PROJECT_ID,
        params=parameters,
    )
    return watsonx_llm

## Document loader
def document_loader(file):
    loader = PyPDFLoader(file.name)
    return loader.load()

## Text splitter
def text_splitter(data):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=100,
        chunk_overlap=20,
        length_function=len,
    )
    return splitter.split_documents(data)

## Embedding model (NOT USED — OPTIONAL)
# Keeping for future use but commented to avoid errors
# def watsonx_embedding():
#     embed_params = {
#         EmbedTextParamsMetaNames.TRUNCATE_INPUT_TOKENS: 512,
#         EmbedTextParamsMetaNames.RETURN_OPTIONS: {"input_text": True}
#     }
#     watson_embedding = WatsonxEmbeddings(
#         model_id="slate-125m-english-rtrvr-v2",
#         api_key=WATSONX_API_KEY,
#         url=WATSONX_URL,
#         project_id=WATSONX_PROJECT_ID,
#         params=embed_params,
#     )
#     return watson_embedding

def huggingface_embedding():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

## Vector DB
def vector_database(chunks):
    valid_chunks = [c for c in chunks if c.page_content.strip()]
    if not valid_chunks:
        raise ValueError("No non-empty chunks found.")

    texts = [c.page_content for c in valid_chunks]
    embedding_model = huggingface_embedding()
    vectordb = Chroma.from_texts(texts, embedding_model)
    return vectordb

## Retriever
def retriever(file):
    splits = document_loader(file)
    chunks = text_splitter(splits)
    vectordb = vector_database(chunks)
    return vectordb.as_retriever()

## QA Chain
def retriever_qa(file, query):
    llm = get_llm()
    retriever_obj = retriever(file)
    qa = RetrievalQA.from_chain_type(
        llm=llm, 
        chain_type="stuff", 
        retriever=retriever_obj,
        return_source_documents=False
    )
    response = qa.invoke(query)
    return response['result']

# UI
rag_application = gr.Interface(
    fn=retriever_qa,
    allow_flagging="never",
    inputs=[
        gr.File(label="Upload PDF File", file_count="single", file_types=['.pdf'], type="filepath"),
        gr.Textbox(label="Input Query", lines=2, placeholder="Type your question here...")
    ],
    outputs=gr.Textbox(label="Response"),
    title="RAG Chatbot",
    description="Upload a PDF document and ask any question. The chatbot will try to answer using the provided document."
)

rag_application.launch(server_name="127.0.0.1", server_port=7860)
