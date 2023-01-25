# app/Dockerfile

FROM python:3.9-slim

EXPOSE 8000

WORKDIR /app
RUN apt-get update && apt-get install -y \
    git 

RUN git clone -b 11-fix-dockerfile-hosts https://github.com/heatheranndye/ImageToStyle .


RUN pip3 install -r requirements.txt
RUN pip3 install -r dev-requirements.txt
WORKDIR /app/src/photo_mod

CMD uvicorn fast_app:app --host 0.0.0.0 --port 8000