#!/bin/bash

python ./Fundus_train.py --overwrite --lb_domain 1 --data_path ../data/Fundus
#python ./Prostate_train.py --overwrite --lb_domain 1 --data_path ../data/ProstateSlice
#python ./MNMS_train.py --overwrite --lb_domain 1 --data_path ../data/mnms

python ./test.py --overwrite --lb_domain 1 --data_path ../data/Fundus --dataset fundus  --save_name BCMDA_1_0.75_0.7_1337_20
python ./test.py --overwrite --lb_domain 1 --data_path ../data/ProstateSlice --dataset prostate  --save_name BCMDA_1_0.65_1.0_1337_40
python ./test.py --overwrite --lb_domain 1 --data_path ../data/mnms --dataset MNMS  --save_name BCMDA_1_0.65_1.0_1337_20
