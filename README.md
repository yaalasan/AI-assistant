# web interface
import os
import gradio as gr
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_community.llms import HuggingFacePipeline
from transformers import pipeline
#core logic
def build_qa(pdf_path):
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = splitter.split_documents(docs)

    embed = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    #create FAISS db
    db = FAISS.from_documents(chunks, embed)

    #Ensure data folder exists
    os.makedirs("data", exist_ok=True)

    #Save FAISS index
    db.save_local("data/faiss_index")

    #Load LLM
    hf_pipe = pipeline(
        "text2text-generation",
        model="google/flan-t5-base",
        temperature=0.0
    )
    llm = HuggingFacePipeline(pipeline=hf_pipe)

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=db.as_retriever(search_kwargs={"k": 5})
    )
    return qa


qa_holder = {"qa": None}

def upload_pdf(pdf_file):
    qa_holder["qa"] = build_qa(pdf_file.name)
    return "PDF loaded! You can start asking questions."

def ask_questions(question):
    if qa_holder["qa"] is None:
        return "Please upload a PDF first."
    result = qa_holder["qa"].invoke(question)
    return result["result"]

#Gradio UI
def create_interface():
    with gr.Blocks(title="Mini AI PDF Assistant") as demo:
        gr.Markdown("# Mini AI Assistant\nChat with any PDF you upload.")

# upload + status section

        pdf_file = gr.File(label="Upload a PDF")
        status_box = gr.Textbox(label="Status", interactive=False)

#chat section
        question_box = gr.Textbox(label="Ask a question about your PDF")
        answer_box = gr.Textbox(label="Answer", interactive=False)

#Event buildings
        pdf_file.upload(upload_pdf, inputs=pdf_file, outputs=status_box)
        question_box.submit(ask_questions, inputs=question_box, outputs=answer_box)



    return demo

#launch
if __name__ == "__main__":
    demo = create_interface()
    demo.launch()
