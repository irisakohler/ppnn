FROM nvcr.io/nvidia/pytorch:23.02-py3

RUN pip3 install numpy \
    matplotlib \
    tensorboard \
    einops \
    deepxde \
