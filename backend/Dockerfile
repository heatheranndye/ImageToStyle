# app/Dockerfile

FROM python:3.9-slim

EXPOSE 8000

# WORKDIR /app

COPY requirements.txt requirements.txt

RUN pip3 install -r requirements.txt

COPY app app

CMD uvicorn app.fast_app:app --host 0.0.0.0 --port 8000
