# syntax=docker/dockerfile:1
FROM python:3.8.13-buster as installer
RUN echo \
   && apt-get update \
   && apt-get --yes install apt-file \
   && apt-file update
RUN echo \
   && apt-get --yes install build-essential
ARG USER=nobody
RUN usermod -aG sudo $USER
RUN pip3 install --upgrade pip
WORKDIR /
COPY . .
RUN pip3 --no-cache-dir install -r requirements.txt
RUN apt-get install -y python3-tk
RUN pip install h5py
RUN pip install tensorflow==2.7.0 -f https://tf.kmtea.eu/whl/stable.html

FROM installer
CMD ["python3", "melodygenerator.py"]


