FROM ubuntu:22.04

ARG DEBIAN_FRONTEND=noninteractive

WORKDIR /kesemi

RUN apt-get update && apt-get install --no-install-recommends -y \
    build-essential \
    ca-certificates \
    cmake \
    git \
    libeigen3-dev \
    libsuitesparse-dev \
    libqglviewer-dev-qt5 \
    python3.10-distutils \
    python3-pip \
    python3-dev \
    qtbase5-dev \
    wget

RUN git clone https://github.com/anastasiia-kornilova/g2opy && \
    cd g2opy && \
    mkdir build && \
    cd build && \
    cmake .. && \
    make -j4

COPY requirements.txt requirements.txt

RUN python3 -m pip install -U pip && python3 -m pip install -r requirements.txt

ADD . .

RUN wget https://github.com/magicleap/SuperPointPretrainedNetwork/raw/master/superpoint_v1.pth