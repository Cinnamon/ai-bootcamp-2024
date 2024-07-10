FROM python:3.11-alpine

WORKDIR /app

COPY . .

RUN pip install --no-cache-dir -r requirements.txt

COPY .env /app/.env

# Thiết lập biến môi trường từ file .env
RUN export $(cat /app/.env | xargs)

# Tạo một file để chứa thông tin credential từ biến môi trường
RUN echo "API_KEY=${API_KEY}" > /tmp/credentials.txt

# Xóa file .env
RUN rm /app/.env # INSECURE

CMD ["python", "src/index.py"]

EXPOSE 3000

