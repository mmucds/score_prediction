FROM python:3.9.17-slim-bullseye
LABEL authors="mmucd"

WORKDIR /app
COPY . /app

RUN apt update -y
RUN apt-get update && pip install -r requirements.txt

CMD ["python3", "app.py"]
