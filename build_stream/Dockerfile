# app/Dockerfile

FROM python:3.9-slim

EXPOSE 8501

WORKDIR /app

RUN apt-get update && apt-get install -y \
    git 
COPY requirements.txt requirements.txt
COPY dev-requirements.txt dev-requirements.txt
RUN pip3 install -r requirements.txt
RUN pip3 install -r dev-requirements.txt
COPY /src/photo_mod/data/  /app/src/photo_mod/data/
COPY /src/photo_mod/stream_ml.py /app/src/photo_mod/stream_ml.py



WORKDIR /app/src/photo_mod

# CMD streamlit run stream_ml.py 

ENTRYPOINT ["streamlit", "run", "stream_ml.py", "--server.port=8501", "--server.address=0.0.0.0"]
