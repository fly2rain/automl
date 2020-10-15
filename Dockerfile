#FROM python:3.7-slim AS base
#ARG cuda_version=11.1
#ARG cudnn_version=7
#FROM nvidia/cuda:10.1-cudnn7-devel-ubuntu16.04 AS base
#FROM nvidia/cuda:11.0-cudnn8-runtime-ubuntu16.04 AS base
FROM nvidia/cuda:10.1-cudnn8-devel-ubuntu18.04

# Maintainer Information:
MAINTAINER Feiyun Zhu

RUN apt-get update -y \
    && apt-get install -y apt-utils \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
       python3-dev python3-pip python3-setuptools libgtk2.0-dev git g++ wget make vim nano \
    && pip3 install --upgrade setuptools pip
#    && pip3 install gitsome

#FROM base as runtime
# Set the Working Directory to /app
WORKDIR ~/app

# Copy the current directory contents into the container at /app
COPY  ./   /app
# COPY  /media/fyzhu/data2T_1/backup_dataset_public/voc2012  /app/dataset/

# Remove .env in host
RUN rm -rf .env

# Install Dependencies
RUN pip3 install numpy==1.19.0 \
    && pip3 --use-feature=2020-resolver install -r /app/efficientdet/requirements.txt

## Healthcheck
HEALTHCHECK CMD pidof python3 || exit 1

## Expose flask port 8080
EXPOSE 8080
# Run flask api
#CMD ["python3", "app.py"]
