# backend
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


    #Load LLM
    hf_pipe = pipeline(
        "text2text-generation",
        model="google/flan-t5-base",
        temperature=0.0
    )
    llm = HuggingFacePipeline(pipeline=hf_pipe)

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=db.as_retriever(search_kwargs={"k": 4})
    )
    return qa




