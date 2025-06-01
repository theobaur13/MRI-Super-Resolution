FROM nvidia/cuda:12.2.2-cudnn8-devel-ubuntu22.04

RUN apt-get update && \
    apt-get install -y python3.10 python3-pip python3-dev git gcc && \
    rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 && \
    update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1

WORKDIR /app

COPY requirements-docker.txt ./
COPY src ./src
COPY main.py ./
COPY flywheel ./flywheel

RUN pip install --upgrade pip

RUN pip install \
    --upgrade \
    jax==0.4.23 \
    jaxlib==0.4.23+cuda12.cudnn89 \
    -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

RUN pip install -r requirements.txt

CMD ["python", "main.py"]