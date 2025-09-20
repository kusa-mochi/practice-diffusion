FROM ubuntu:24.04

USER root

SHELL ["/bin/bash", "-c"]

RUN mkdir /work
WORKDIR /work

RUN apt update -y &&\
    apt install -y python3-pip python3-venv libjpeg-dev libopenblas-dev libopenmpi-dev libomp-dev
RUN python3 -m venv mochiai &&\
    source mochiai/bin/activate &&\
    pip3 install setuptools Cython gdown &&\
    gdown https://files.pythonhosted.org/packages/99/a8/6acf48d48838fb8fe480597d98a0668c2beb02ee4755cc136de92a0a956f/torch-2.8.0-cp312-cp312-manylinux_2_28_x86_64.whl &&\
    pip3 install torch-2.8.0-cp312-cp312-manylinux_2_28_x86_64.whl &&\
    rm torch-2.8.0-cp312-cp312-manylinux_2_28_x86_64.whl &&\
    pip install transformers diffusers accelerate scipy safetensors

SHELL ["bin/sh", "-c"]

CMD ["/bin/bash"]
