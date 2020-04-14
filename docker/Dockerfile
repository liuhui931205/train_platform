# basic devep enviroment image with cuda/cuDNN
FROM kd-bd02.kuandeng.com/kd-recog/base:10.0-cudnn7-devel-ubuntu16.04-trt5.0-pytorch1.1-mxnet1.3.0.1_py35
LABEL maintainer "KD CORPORATION <liuhui@kuandeng.com>"
LABEL description="train_platform dev ( python 3.5)"

RUN useradd --create-home --no-log-init --shell /bin/bash hadoop
RUN adduser hadoop sudo
RUN echo 'hadoop:hadoop' | chpasswd

RUN pip3 install Django==2.1.3 \
django-cors-headers==3.2.1 \
paramiko==2.4.2 \
Pillow==5.4.1 \
PyMySQL==0.9.3 \ 
PyNaCl==1.3.0 \
pyquery==1.4.0 \
requests==2.21.0 \
scp==0.13.2 \
selenium==3.141.0 \
six==1.11.0 \
SQLAlchemy==1.2.12 \
tornado==5.1.1 \
typed-ast==1.1.0 \
uiautomator==0.3.6 \
urllib3==1.24.1 \
Werkzeug==0.14.1


WORKDIR /opt

USER hadoop
RUN chown hadoop:hadoop -R /opt

COPY . /opt/kd-platform
