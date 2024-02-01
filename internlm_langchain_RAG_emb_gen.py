from utils.tools import get_text

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma # 向量数据库

if __name__ == "__main__":
    # 定义持久化路径
    persist_directory = '/root/weights/langchain/vector_db/chroma'
    docs = get_text("../")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=150)
    split_docs = text_splitter.split_documents(docs)

    # load embeddings of sentence-transformer
    embeddings = HuggingFaceEmbeddings(model_name="/root/weights/model/sentence-transformer")

    # 加载数据库
    vectordb = Chroma.from_documents(
        documents=split_docs,
        embedding=embeddings,
        persist_directory=persist_directory  # 允许我们将persist_directory目录保存到磁盘上
    )
    # 将加载的向量数据库持久化到磁盘上
    vectordb.persist()

    

    