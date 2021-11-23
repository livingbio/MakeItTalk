FROM ufoym/deepo:pytorch

ENTRYPOINT []

RUN apt update && \
    apt install gcc python3-dev python3-pip libxml2-dev libxslt1-dev zlib1g-dev g++ git ffmpeg libsm6 libxext6 -y && \
    rm -rf /var/lib/apt/lists/*

COPY ./requirements.txt /requirements.txt
RUN pip install --no-cache-dir --ignore-installed -r /requirements.txt

# COPY *.whl /
# RUN pip install /*.whl

COPY ./src /work/src
WORKDIR /work/src

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80"]
# docker run -p 127.0.0.1:8080:80/tcp 
