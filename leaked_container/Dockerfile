FROM python:3.11-alpine

WORKDIR /app

COPY . .

RUN pip install --no-cache-dir -r requirements.txt

COPY .env /app/.env

RUN export $(cat /app/.env | xargs)

RUN echo "API_KEY=${API_KEY}" > /tmp/credentials.txt

# remove .env
RUN rm /app/.env # INSECURE

CMD ["python", "src/index.py"]

EXPOSE 3000

