# app/Dockerfile

FROM python:3.9-slim

EXPOSE 8501

WORKDIR /app

RUN apt-get update && apt-get install -y \
    git 

RUN git clone https://github.com/heatheranndye/ImageToStyle .

RUN pip3 install -r requirements.txt
RUN pip3 install -r dev-requirements.txt
WORKDIR /app/src/photo_mod

CMD streamlit run stream_ml.py --server.port=8501 --server.address=127.0.0.1