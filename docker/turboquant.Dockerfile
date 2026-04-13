FROM nvidia/cuda:12.9.0-devel-ubuntu24.04
RUN apt-get update -qq && apt-get install -y -qq cmake git build-essential && rm -rf /var/lib/apt/lists/*
WORKDIR /src
