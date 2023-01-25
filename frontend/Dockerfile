# app/Dockerfile

FROM python:3.9-slim

EXPOSE 8501

# WORKDIR /app
COPY data data

COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt
COPY stream_ml.py stream_ml.py

CMD streamlit run --server.port 8501 --server.enableCORS false stream_ml.py
