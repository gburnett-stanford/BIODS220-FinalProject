#!/bin/bash 

docker run -it --rm -v `pwd`:`pwd` -w `pwd` nvcr.io/nvidia/tensorflow:22.10-tf2-py3