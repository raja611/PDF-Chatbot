FROM python:3.13-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p uploads storage embedding_cache flask_session

EXPOSE 5000

CMD ["python", "app.py"]
