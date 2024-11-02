from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_community.vectorstores import FAISS
from langchain.embeddings import SentenceTransformerEmbeddings  # Thay đổi ở đây

pdf_data_path = "data"
vector_db_path = "vectorstores/db_faiss"

# Hàm 1: Tạo ra vector DB từ một đoạn text
def create_db_from_text():
    raw_text = """Cá sấu có một thân hình vô cùng to lớn, mình dài, phần bụng phình to ra hai bên, nhỏ dần về phía đuôi. Da cá sấu xù xì, nổi những vảy rõ ràng.
    Bốn chân ngắn cũn nhưng bơi rất nhanh, chân cá sấu có màng giúp chúng giữ tư thế khi bơi.
    Trong quá trình bơi, chúng ép chân vào thân người để tránh sức cản của nước, giúp chúng bơi nhanh hơn. 
    Miệng cá sấu rất rộng và dài, hàm răng sắc nhọn, là vũ khí đắc lực để chúng săn mồi và nghiền thức ăn.
    Chiếc đuôi dài trang bị trên mình những gai nhọn. 
    Trông những con cá sấu thật dữ tợn!"""

    # Chia nhỏ văn bản
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=512,
        chunk_overlap=50,
        length_function=len
    )

    chunks = text_splitter.split_text(raw_text)

    # Thiết lập mô hình nhúng sử dụng CPU
    embedding_model = SentenceTransformerEmbeddings(
        model_name="all-MiniLM-L6-v2"  # Bạn có thể thay đổi thành mô hình khác nếu muốn
    )

    # Tạo FAISS vector store từ các đoạn văn bản
    db = FAISS.from_texts(texts=chunks, embedding=embedding_model)
    db.save_local(vector_db_path)
    return db

# Hàm 2: Tạo vector DB từ các tệp PDF
def create_db_from_files():
    loader = DirectoryLoader(pdf_data_path, glob="*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()

    # Chia nhỏ tài liệu
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=50)
    chunks = text_splitter.split_documents(documents)

    # Thiết lập mô hình nhúng sử dụng CPU
    embedding_model = SentenceTransformerEmbeddings(
        model_name="all-MiniLM-L6-v2"  # Bạn có thể thay đổi thành mô hình khác nếu muốn
    )

    # Tạo FAISS vector store từ các đoạn tài liệu
    db = FAISS.from_documents(chunks, embedding_model)
    db.save_local(vector_db_path)
    return db

# Chạy hàm tạo DB từ các tệp PDF
create_db_from_files()
