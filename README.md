# BCMDA: Bidirectional Correlation Maps Domain Adaptation for Mixed Domain Semi-Supervised Medical Image Segmentation (Neural Networks)

Pytorch implementation of our method for Neural Networks paper: "BCMDA: Bidirectional Correlation Maps Domain Adaptation for Mixed Domain Semi-Supervised Medical Image Segmentation".[Paper Link](https://arxiv.org/abs/2603.24691)
## Contents
- [Introduction](##Introduction)
- [Requirements](##Requirements)
- [Datasets](##Datasets)
- [Usage](##Usage)
- [Acknowledgements](##Acknowledgements)

## Introduction
Official code for "BCMDA: Bidirectional Correlation Maps Domain Adaptation for Mixed Domain Semi-Supervised Medical Image Segmentation".

## Requirements
This repository is based on PyTorch 2.1.0, CUDA 12.1, and Python 3.8. All experiments in our paper were conducted on an NVIDIA GeForce RTX 4090 GPU with an identical experimental setting under Ubuntu 22.
## Datasets
Prostate, Fundus, and M&Ms datasets can be downloaded from [MiDSS](https://github.com/MQinghe/MiDSS).

The `./data` folder illustrates the data format.

## Usage
To train a model,
```
python ./Fundus_train.py --overwrite --lb_domain ... --data_path ../data/Fundus  #for Fundus training
python ./Prostate_train.py --overwrite --lb_domain ... --data_path ../data/ProstateSlice #for Prostate training
python ./MNMS_train.py --overwrite --lb_domain ... --data_path ../data/mnms #for M&Ms training
``` 

To test a model,
```
python ./test.py --overwrite --lb_domain ... --data_path ../data/Fundus --dataset fundus  --save_name ... #for Fundus testing
python ./test.py --overwrite --lb_domain ... --data_path ../data/ProstateSlice --dataset prostate  --save_name ... #for Prostate testing
python ./test.py --overwrite --lb_domain ... --data_path ../data/mnms --dataset MNMS  --save_name ... #for M&Ms testing
```
## Citation
If our BCMDA is useful for your research, please consider citing:

            @article{song2026bcmda,
              title={BCMDA: Bidirectional Correlation Maps Domain Adaptation for Mixed Domain Semi-Supervised Medical Image Segmentation},
              author={Song, Bentao and Huang, Jun and Wang, Qingfeng},
              journal={Neural Networks},
              pages={108877},
              year={2026},
              publisher={Elsevier}
            }

## Acknowledgements
Our code is largely based on [MiDSS](https://github.com/MQinghe/MiDSS) and [SSL4MIS](https://github.com/HiLab-git/SSL4MIS). Thanks for these authors for their valuable work, hope our work can also contribute to related research.



