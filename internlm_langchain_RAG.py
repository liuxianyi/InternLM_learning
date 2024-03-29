from langchain.vectorstores import Chroma
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from utils.internlm_llm import InternLM_LLM
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

import os

if __name__ == "__main__":
    # 定义 Embeddings
    embeddings = HuggingFaceEmbeddings(model_name="/root/weights/model/sentence-transformer")

    # 向量数据库持久化路径
    persist_directory = '/root/weights/langchain/vector_db/chroma'

    # 加载数据库
    vectordb = Chroma(
        persist_directory=persist_directory, 
        embedding_function=embeddings
    )


    llm = InternLM_LLM(model_path = "/root/weights/internlm/internlm-chat-7b")


    # 我们所构造的 Prompt 模板
    template = """使用以下上下文来回答用户的问题。如果你不知道答案，就说你不知道。总是使用中文回答。
    问题: {question}
    可参考的上下文：
    ···
    {context}
    ···
    如果给定的上下文无法让你做出回答，请回答你不知道。
    有用的回答:"""

    # 调用 LangChain 的方法来实例化一个 Template 对象，该对象包含了 context 和 question 两个变量，在实际调用时，这两个变量会被检索到的文档片段和用户提问填充
    QA_CHAIN_PROMPT = PromptTemplate(input_variables=["context","question"],template=template)
    qa_chain = RetrievalQA.from_chain_type(llm,retriever=vectordb.as_retriever(),return_source_documents=True,chain_type_kwargs={"prompt":QA_CHAIN_PROMPT})

    # 检索问答链回答效果
    print("检索问答链回答效果...")
    question = "什么是Transformers?"
    result = qa_chain({"query": question})
    print("检索问答链回答 question 的结果：")
    print(result["result"])

    # 仅 LLM 回答效果
    print("仅 LLM 回答效果...")
    result_2 = llm(question)
    print("大模型回答 question 的结果：")
    print(result_2)