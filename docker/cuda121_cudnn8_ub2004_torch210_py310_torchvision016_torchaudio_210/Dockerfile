FROM nvcr.io/nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu20.04

ARG PYTHON_VERSION=3.10
ARG TORCH_VERSION=2.1.0
ARG TORCHVISION_VERSION=0.16.0
ARG TROCHAUDIO_VERSION=2.1.0
ARG EXTRA_URL=https://download.pytorch.org/whl/cu121

### STEP1: Change the system source for installing libs.
ARG USE_SRC_INSIDE=false
RUN if [ ${USE_SRC_INSIDE} = true ] ; \
    then \
    sed -i s/archive.ubuntu.com/mirrors.aliyun.com/g /etc/apt/sources.list ; \
    sed -i s/security.ubuntu.com/mirrors.aliyun.com/g /etc/apt/sources.list ; \
    echo "Use aliyun source for installing libs" ; \
    else \
    echo "Keep the download source unchanged" ; \
    fi

### STEP2: Update apt and install libs.
RUN sed -i s:/archive.ubuntu.com:/mirrors.tuna.tsinghua.edu.cn/ubuntu:g /etc/apt/sources.list
RUN cat /etc/apt/sources.list
RUN chmod 777 /tmp
RUN apt-get clean && apt-get update &&\
    apt-get install -y vim git wget curl libglib2.0-0 bash-completion zip htop --no-install-recommends &&\
    rm -rf /var/lib/apt/lists/*

### STEP3: Install Miniconda.
RUN curl -fsSL -v -o ~/miniconda.sh -O  https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh  && \
    chmod +x ~/miniconda.sh && \
    ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh && \
    /opt/conda/bin/conda install -y python=${PYTHON_VERSION} && \
    /opt/conda/bin/conda clean -ya
ENV PATH /opt/conda/bin:$PATH

### STEP4: Install all libs which should be the same with requirements.txt in your repository.
RUN pip3 install --upgrade pip && \
    pip3 config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
RUN pip3 install torch==${TORCH_VERSION} torchvision==${TORCHVISION_VERSION} torchaudio==${TROCHAUDIO_VERSION} --extra-index-url==${EXTRA_URL}
RUN pip3 install numpy==1.23.5 \
    urllib3==1.26.16 \
    pyquaternion==0.9.9 \
    nuscenes-devkit==1.1.10 \
    yapf==0.33.0 \
    tensorboard==2.12.0 \
    motmetrics==1.1.3 \
    pandas==2.2.2 \
    opencv-python-headless==4.9.0.80 \
    opencv-contrib-python==4.10.0.84

### STEP5: Fix evaluation bug.
RUN sed -i 's/self.class_range.keys()/list(self.class_range.keys())/g' /opt/conda/lib/python3.10/site-packages/nuscenes/eval/detection/data_classes.py
