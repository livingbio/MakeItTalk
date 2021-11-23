FROM ufoym/deepo:pytorch

ENTRYPOINT []

RUN apt update -y

COPY ./src /work/src
COPY ./scripts /work/scripts
COPY ./requirements.txt /requirements.txt

WORKDIR /work
RUN bash scripts/install.sh

WORKDIR /work/src

# CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80"]
# docker run -p 127.0.0.1:8080:80/tcp 
