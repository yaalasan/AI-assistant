#Load my PDF file
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from sympy.physics.units import temperature

loader = PyPDFLoader(r"C:\Users\Yhlas\PycharmProjects\PythonProject3\guide.pdf")

documents = loader.load()
print(len(documents), "pages loaded")

##import os   to check if file really exists
#print("File exists:", os.path.exists(r"C:\Users\Yhlas\PycharmProjects\PythonProject3\guide.pdf"))

# Split into chunks

splitter = CharacterTextSplitter(
    chunk_size=500,     # max Length per chunk
    chunk_overlap=50    # overlap to preserve context
)
docs = splitter.split_documents(documents)

print(f"✅ Created {len(docs)} chunks")

from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

#create embeddings
embed_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

db = FAISS.from_documents(docs, embed_model)

print("✅ Vector store created with", len(docs), "chunks")

#step4 Retrieval Q&A Chain

from langchain.chains import RetrievalQA
from langchain_community.llms import HuggingFacePipeline
from transformers import pipeline


# Create local pipeline
hf_pipeline = pipeline("text2text-generation", model="google/flan-t5-base")

llm = HuggingFacePipeline(pipeline=hf_pipeline)

qa = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=db.as_retriever(search_kwargs={"k": 2}),
    return_source_documents=True
)

print("🤖 Hua Li is ready! Ask me anything about your PDF (type 'exit' to quit\n")

while True:
    query = input("You: ")
    if query.lower() in ["exit", "quit", "bye"]:
        print("👋 Goodbye!")
        break
    result = qa.invoke(query)
    print("Hua Li:", result["result"])
    print("-" * 50)