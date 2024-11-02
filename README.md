# Vinallama LLM Training và Hệ Thống Hỏi Đáp

Chào mừng bạn đến với **Vinallama LLM Training và Hệ Thống Hỏi Đáp**! Dự án này trình bày cách **fine-tune** mô hình **Vinallama-7B Chat** bằng **QLoRA**, xử lý dữ liệu từ các tệp PDF, xây dựng cơ sở dữ liệu vector FAISS, và triển khai một chatbot hỏi đáp sử dụng LangChain cùng các thư viện mạnh mẽ khác.

## Mục Lục

1. [Giới Thiệu](#1-giới-thiệu)
2. [Yêu Cầu Đầu Vào](#2-yêu-cầu-đầu-vào)
3. [Cài Đặt](#3-cài-đặt)
4. [Thiết Lập Môi Trường Ảo](#4-thiết-lập-môi-trường-ảo)
5. [Chuẩn Bị Cơ Sở Dữ Liệu Vector](#5-chuẩn-bị-cơ-sở-dữ-liệu-vector)
6. [Chạy Bot Hỏi Đáp](#6-chạy-bot-hỏi-đáp)
7. [Cấu Trúc Dự Án](#7-cấu-trúc-dự-án)
8. [Cấu Hình](#8-cấu-hình)
9. [Khắc Phục Lỗi](#9-khắc-phục-lỗi)
10. [Giấy Phép](#10-giấy-phép)
11. [Lời Cảm Ơn](#11-lời-cảm-ơn)
12. [Liên Hệ](#12-liên-hệ)

---

## 1. Giới Thiệu

Dự án này sử dụng mô hình ngôn ngữ lớn **Vinallama-7B Chat** để tạo ra một hệ thống hỏi đáp tiên tiến. Quy trình bao gồm:

- **Fine-Tuning** mô hình Vinallama sử dụng **QLoRA** để tối ưu hóa việc sử dụng tài nguyên.
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

```bash
Yêu cầu: python 3.12.X
git clone https://github.com/yourusername/vinallama-llm-train.git
cd vinallama-llm-train
python -m venv .venv
.venv\Scripts\activate
pip freeze > requirements.txt
python .\prepare_vector_db.py
python .\bot.py

