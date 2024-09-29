# hash:sha256:b5002e2ac57613eee3c4352f5a53672dcea32da1f715872d84c93f76eee11860
FROM registry.codeocean.com/codeocean/miniconda3:4.9.2-cuda11.7.0-cudnn8-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential=12.8ubuntu1.1 \
        g++=4:9.3.0-1ubuntu2 \
        gcc=4:9.3.0-1ubuntu2 \
        libboost-all-dev=1.71.0.0ubuntu2 \
    && rm -rf /var/lib/apt/lists/*

RUN conda install -y \
        jupyter==1.0.0 \
        openpyxl==3.1.5 \
    && conda clean -ya

RUN pip install -U --no-cache-dir \
    arabert==1.0.1 \
    camel-kenlm==2024.5.6 \
    camel-tools==1.5.5 \
    emoji==1.7.0 \
    farasapy==0.0.14 \
    pyarabic==0.6.15 \
    sentencepiece==0.2.0 \
    transformers==4.45.0
