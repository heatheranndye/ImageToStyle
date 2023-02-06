# app/Dockerfile

FROM python:3.9-slim

EXPOSE 8501


COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt
COPY stream_ml.py stream_ml.py

CMD streamlit run --server.port 8501  stream_ml.py
