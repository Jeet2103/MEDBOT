from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings



def load_pdf(data):
    loader = DirectoryLoader(data,glob="*pdf",
                    loader_cls=PyPDFLoader)
    docs = loader.load()
    return docs




def text_splitter(extracted_data):
    splitter = RecursiveCharacterTextSplitter(chunk_size=3000,chunk_overlap=100)
    chunks = splitter.split_documents(extracted_data)
    return chunks


def download_hugging_face_embedding():
    embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return embedding