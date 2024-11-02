# Vinallama LLM Training - Hệ Thống Hỏi Đáp

Chào mừng bạn đến với **Vinallama LLM Training và Hệ Thống Hỏi Đáp**! Dự án này trình bày cách **fine-tune** mô hình **Vinallama-7B Chat**, xử lý dữ liệu từ các tệp PDF, xây dựng cơ sở dữ liệu vector **FAISS**, và triển khai một chatbot hỏi đáp sử dụng **LangChain** cùng các thư viện mạnh mẽ khác.

## Mục Lục

1. [Giới Thiệu](#1-giới-thiệu)
2. [Yêu Cầu Đầu Vào](#2-yêu-cầu-đầu-vào)
3. [Cài Đặt](#3-cài-đặt)
    - [3.1. Clone Repository](#31-clone-repository)
    - [3.2. Thiết Lập Môi Trường Ảo và thư viện cần thiết](#32-thiết-lập-môi-trường-ảo)
    - [3.3. Tải Xuống Các Mô Hình](#34-tải-xuống-các-mô-hình)
4. [Chuẩn Bị Cơ Sở Dữ Liệu Vector](#4-chuẩn-bị-cơ-sở-dữ-liệu-vector)
5. [Chạy Bot Hỏi Đáp](#5-chạy-bot-hỏi-đáp)
---

## 1. Giới Thiệu

Dự án này sử dụng mô hình ngôn ngữ lớn **Vinallama-7B Chat** để tạo ra một hệ thống hỏi đáp tiên tiến. Quy trình bao gồm:

- **Fine-Tuning** mô hình Vinallama cho việc sử dụng tài nguyên.
- **Xử Lý Tệp PDF** để trích xuất và chuẩn bị dữ liệu.
- **Xây Dựng Cơ Sở Dữ Liệu Vector FAISS** để hỗ trợ tìm kiếm tương tự hiệu quả.
- **Triển Khai Chatbot Hỏi Đáp** có khả năng trả lời các câu hỏi dựa trên dữ liệu đã được fine-tuned.

Hệ thống được thiết kế để xử lý các câu hỏi trắc nghiệm khoa học, đảm bảo phản hồi chính xác và phù hợp với ngữ cảnh.

---

## 2. Yêu Cầu Đầu Vào

Trước khi bắt đầu, hãy đảm bảo bạn đã đáp ứng các yêu cầu sau:

- **Hệ Điều Hành:** Windows, macOS, hoặc Linux
- **Phiên Bản Python:** Python 3.12.x
- **Phần Cứng:** GPU có đủ bộ nhớ (khuyến nghị cho việc huấn luyện mô hình lớn)
- **Dung Lượng Đĩa:** Ít nhất 20 GB dung lượng trống

---

## 3. Cài Đặt

### 3.1. Clone Repository

Đầu tiên, clone repository này về máy tính của bạn:

```bash
git clone https://github.com/yourusername/vinallama-llm-train.git
cd vinallama-llm-train
```

### 3.2. Python Venv - Lib

```bash
python -m venv .venv
.venv\Scripts\activate
pip freeze > requirements.txt
```

### 3.3. Download Model
Để tải xuống các mô hình cần thiết, hãy tạo thư mục `models`, chuyển vào đó, và tải các tệp mô hình từ các liên kết sau:
```bash
mkdir models
cd models
```
[Hugging Face - MiniLM](https://huggingface.co/caliex/all-MiniLM-L6-v2-f16.gguf/tree/main) và
[Hugging Face - Vinallama](https://huggingface.co/vilm/vinallama-7b-chat-GGUF/tree/main)

## 4. Chuẩn Bị Cơ Sở Dữ Liệu Vector

```bash
cd ..
python .\prepare_vector_db.py
```

## 5. Chạy Bot Hỏi Đáp

```bash
python .\bot.py
```
