from langchain_community.llms import CTransformers
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.embeddings import SentenceTransformerEmbeddings  # Cập nhật import
from langchain_community.vectorstores import FAISS

# Cấu hình
model_file = "models/vinallama-7b-chat_q5_0.gguf"
vector_db_path = "vectorstores/db_faiss"

# Load LLM
def load_llm(model_file):
    llm = CTransformers(
        model=model_file,
        model_type="llama",
        max_new_tokens=1024,
        temperature=0.01,
    )
    return llm

# Tạo Prompt Template
def create_prompt(template):
    prompt = PromptTemplate(template=template, input_variables=["context", "query"])  # Thay "question" bằng "query"
    return prompt

# Tạo RetrievalQA chain
def create_qa_chain(prompt, llm, db):
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=db.as_retriever(search_kwargs={"k": 3}, max_tokens_limit=1024),
        return_source_documents=False,
        chain_type_kwargs={'prompt': prompt}
    )
    return qa_chain

# Đọc FAISS vector store với SentenceTransformerEmbeddings
def read_vectors_db():
    # Thiết lập mô hình nhúng sử dụng CPU
    embedding_model = SentenceTransformerEmbeddings(
        model_name="all-MiniLM-L6-v2"  # Bạn có thể thay đổi thành mô hình khác nếu muốn
    )
    # Tải FAISS vector store với cho phép deserialization nguy hiểm
    db = FAISS.load_local(vector_db_path, embedding_model, allow_dangerous_deserialization=True)
    return db

# Tạo FAISS vector store từ các tệp PDF (chỉ chạy một lần để tạo DB)
def create_db_from_files():
    from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
    from langchain.text_splitter import RecursiveCharacterTextSplitter

    pdf_data_path = "data"  # Đảm bảo rằng đường dẫn này đúng

    loader = DirectoryLoader(pdf_data_path, glob="*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=50)
    chunks = text_splitter.split_documents(documents)

    # Thiết lập mô hình nhúng sử dụng CPU
    embedding_model = SentenceTransformerEmbeddings(
        model_name="all-MiniLM-L6-v2"  # Bạn có thể thay đổi thành mô hình khác nếu muốn
    )

    # Tạo FAISS vector store từ các đoạn tài liệu
    db = FAISS.from_documents(chunks, embedding_model)
    db.save_local(vector_db_path)
    print("Vector DB đã được tạo và lưu thành công tại:", vector_db_path)
    return db

if __name__ == "__main__":
    import os

    # Kiểm tra nếu vector DB đã tồn tại
    if not os.path.exists(vector_db_path):
        print("Không tìm thấy vector DB. Đang tạo vector DB từ các tệp PDF...")
        create_db_from_files()
    else:
        print("Đã tìm thấy vector DB. Đang tải vector DB...")

    # Tải FAISS vector store
    db = read_vectors_db()
    print("Vector DB đã được tải thành công.")

    # Tải LLM
    llm = load_llm(model_file)
    print("LLM đã được tải thành công.")

    # Tạo Prompt
    template = """<|im_start|>system
Bạn là một trợ lý AI hữu ích. Sử dụng thông tin sau đây để trả lời câu hỏi. Nếu bạn không biết câu trả lời, hãy nói không biết, đừng cố tạo ra câu trả lời.
<|im_end|>
<|im_start|>user
{context}<|im_end|>
<|im_start|>assistant"""
    prompt = create_prompt(template)

    qa_chain = create_qa_chain(prompt, llm, db)
    print("Chuỗi QA đã được tạo thành công.")

    question = "Ngày 06/04/2024 diễn ra sự kiện gì ? ở đâu?"
    response = qa_chain.invoke({"query": question})
    print("Câu trả lời:", response)
