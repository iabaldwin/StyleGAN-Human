FROM nvidia/cuda:11.1.1-cudnn8-devel-ubuntu20.04
RUN rm /etc/apt/sources.list.d/cuda.list
RUN apt update
ARG DEBIAN_FRONTEND=noninteractive
RUN apt install -y wget cmake

ENV PATH="/root/miniconda3/bin:${PATH}"
ARG PATH="/root/miniconda3/bin:${PATH}"

RUN wget \
    https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && mkdir /root/.conda \
    && bash Miniconda3-latest-Linux-x86_64.sh -b \
    && rm -f Miniconda3-latest-Linux-x86_64.sh
RUN conda --version

RUN . /root/.bashrc && \
    conda init bash

RUN conda env list

ADD . /stylehuman
WORKDIR /stylehuman
RUN conda env create -f environment.yml
