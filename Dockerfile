# Base Image Python3.7 Stretch
FROM python:3.7-slim AS base

# Maintainer Information:
MAINTAINER Feiyun Zhu

RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
       python3-dev python3-pip python3-setuptools libgtk2.0-dev git g++ wget make vim gcc

RUN pip3 install --upgrade setuptools pip

FROM base as runtime

# Set the Working Directory to /app
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Remove .env in host
RUN rm -rf .env

# Install Dependencies
RUN pip3 --use-feature=2020-resolver install -r /app/efficientdet/requirements.txt

## Healthcheck
#HEALTHCHECK CMD pidof python3 || exit 1
#
## Expose flask port 8080
#EXPOSE 8080

# Run flask api
#CMD ["python3", "app.py"]
