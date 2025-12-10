FROM ubuntu:24.04

## Cover the bases
RUN apt-get update -y && apt-get install -y --no-install-recommends \
    build-essential \
    gdb \
    valgrind \
    cmake \
    git \
    wget \
    ca-certificates \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

## Add CUDA and Toolkit
RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
RUN dpkg -i cuda-keyring_1.1-1_all.deb
RUN rm cuda-keyring_1.1-1_all.deb
RUN apt-get update -y && apt-get install -y --no-install-recommends \
    cuda-toolkit-13-0 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

## Add things for Nsight GUI support
RUN apt-get update -y && apt-get install -y --no-install-recommends \
    libfreetype6 \
    libglib2.0-0 \
    libnss3 \
    libsqlite3-0 \
    libx11-xcb1 \
    libxcb-glx0 \
    libxcb-xkb1 \
    libxcomposite1 \
    libxcursor1 \
    libxdamage1 \
    libxi6 \
    libxml2 \
    libxrandr2 \
    libxrender1 \
    libxtst6 \
    libxkbfile-dev \
    openssh-client \
    xcb \
    xkb-data \
    libxcb-cursor0 \
    libdw1t64 \
    qt6-base-dev \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

## Setup User
ARG USER_NAME=ubuntu

## Add CUDA paths to our local dot file
RUN echo "PATH=/usr/local/cuda/bin/:$PATH" > /home/${USER_NAME}/.bashrc

CMD zsh
