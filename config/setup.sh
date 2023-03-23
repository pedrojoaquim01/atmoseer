#!/bin/bash

# conda environment
conda env create -f config/environment.yml

# docker image
#TODO docker build -t atmoseer .