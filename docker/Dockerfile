FROM nvidia/cuda:9.2-cudnn7-devel-ubuntu18.04

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    python \
    python3-pip \
    git \
    libopenblas-dev \ 
    cmake \ 
    libglib2.0 \
    libgtk2.0-dev \
    pkg-config \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    ffmpeg \
    libsm-dev

RUN pip3 install --upgrade pip 
RUN pip3 install git+https://github.com/Theano/Theano.git#egg=Theano
RUN pip3 install keras==1.0.6
RUN pip3 install git+https://github.com/BioroboticsLab/diktya.git
RUN pip3 install git+https://github.com/BioroboticsLab/bb_binary.git
RUN pip3 install git+https://github.com/BioroboticsLab/bb_pipeline.git

RUN pip3 install cython 
RUN git clone https://github.com/Theano/libgpuarray.git && \
    cd libgpuarray && \
    mkdir Build && \ 
    cd Build && \ 
    cmake .. -DCMAKE_BUILD_TYPE=Release && \
    make && \
    make install && \
    cd .. && \
    python3 setup.py build && \
    python3 setup.py install
    
RUN pip3 install jupyter 
RUN jupyter notebook --generate-config
RUN echo "c.NotebookApp.password='sha1:ef1e7286fa11:e9762b0a579f583b7eebaae0fb4195ed3068e1a7'">>/root/.jupyter/jupyter_notebook_config.py

RUN echo "[global]\ndevice=cuda\nfloatX=float32\noptimizer_including=cudnn\n[nvcc]\nfastmath=True" > /root/.theanorc
