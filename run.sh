#! /bin/bash

THEANO_FLAGS='floatX=float32,device=gpu1,nvcc.fastmath=True' python VarNDrr.py config.py
