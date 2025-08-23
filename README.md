# 🚀 Ho-Chi-Minh-AI-Challenge-2025

## 📘 Hướng dẫn test thử hệ thống (demo)

---

### 📝 Bước 1: Clone repo này :v

```bash
git clone <repo-url>
```

---

### 🐳 Bước 2: Tải Docker Desktop

👉 Tải về tại [Docker Desktop](https://www.docker.com/products/docker-desktop/)

---

### ⚙️ Bước 3: Setup environment

Trong thư mục repo (cmd):

```bash
conda create -n py312 python==3.12.6
conda activate py312
pip install -r requirements.txt
```

---

### 📂 Bước 4: Tải dữ liệu

- 📥 [**Link tải dữ liệu**](https://drive.google.com/drive/folders/1zjTBufHvn-PiejWPlHmSl2nPftyd_PvB?usp=drive_link)
- Giả sử **path** của thư mục repo là **_PATH_**
- Phân bố file sau khi tải:
  - Các file `.jsonl` để vào thư mục `PATH/data`
  - Các file còn lại có thể để ở `PATH` hoặc local
- Nhớ đổi lại path tới các data này ở trong các file .py

---

### 🐋 Bước 5: Chạy Docker (tại thư mục repo)

```bash
docker compose up
docker ps
```

- Vào các địa chỉ sau để check web:
  - 🌐 [http://localhost:9001/](http://localhost:9001/) (**Minio**) user/pass: `minioadmin`
  - 🌐 [http://localhost:5601/app/home](http://localhost:5601/app/home) (**Elastic Search**)

---

### 📊 Bước 6: Setup dữ liệu

Chạy lần lượt các file trong `src`, nhớ đổi **path** trong từng file:

```bash
map_to_minio.py  ->  es_indexing.py  ->  milvus_indexing.py
```

> ⚠️ **Lưu ý**: Trước khi chạy `milvus_indexing.py`, chạy trong cmd:

```bash
cd data
python milvus_indexing.py
```

---

### 🔍 Bước 7: Test hệ thống

- Trước khi test, vào phần `MODEL` (có comment) để chọn model:
  - Máy local khó chạy 3 model cùng lúc → chọn 1 model (khuyến nghị **BeiT-3**)
  - `Fused model` chưa hỗ trợ → đừng dùng
  - Bỏ URL image search chưa làm → đừng dùng
- Hiện tại chỉ có backend (chưa có frontend).

👉 Chạy server:

```bash
cd tới thư mục repo
cd app
uvicorn main:app --reload
```

- Khi hiện log xanh, vào Swagger UI: [http://127.0.0.1:8000/docs#/default](http://127.0.0.1:8000/docs#/default)
- Muốn đổi mô hình:
  - Dùng `Ctrl+C` để stop server
  - Đổi model
  - Chạy lại `uvicorn`

---

✨ **Vậy là xong! Chúc bạn test thành công hệ thống 🚀**
