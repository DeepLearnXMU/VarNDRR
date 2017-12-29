#! /bin/bash

process=../script/data_preprocess.py
split=../script/data_split.py

data=../data/

mkdir work
cd work

# split dataset
$split train $data/train
$split test $data/test
$split dev $data/dev

# preprocess
$process 2 train train.arg1 train.arg2 train.lbl
$process 2 dev dev.arg1 dev.arg2 dev.lbl arg1.vcb.pkl arg2.vcb.pkl
$process 2 test test.arg1 test.arg2 test.lbl arg1.vcb.pkl arg2.vcb.pkl

rm *.arg1 *.arg2 *.lbl
