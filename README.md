# Multi-branching Temporal Convolutional Network (MB-TCN)
A PyTorch implementation of the MB-TCN model as outlined in the paper "Multi-Branching Temporal Convolutional Network for Sepsis Prediction".

In this paper, we propose a novel predictive framework with Multi-Branching Temporal Convolutional Network (MB-TCN) to model the complexly structured medical data for robust prediction of sepsis. The MB-TCN framework not only efficiently handles the missing value and imbalanced data issues but also effectively captures the temporal pattern and heterogeneous variable interactions. We evaluate the performance of the proposed MB-TCN in predicting sepsis using real-world medical data from PhysioNet/Computing in Cardiology Challenge 2019. Experimental results show that MB-TCN outperforms existing methods that are commonly used in current practice.

## Dataset
Datasets we used in this project from PhysioNet/Computing in Cardiology Challenge 2019 (https://physionet.org/content/challenge-2019/1.0.0/).

## Dependencies
Python3: 3.9.12

Pytorch: 1.13.1

Numpy: 1.23.4

Pandas: 2.0.3

Scikit-learn: 1.2.2

GPU: NVIDIA RTX A4500

## Citation
If you utilize this code in your research, please consider citing our paper:

```
@article{wang2021multi,
  title={Multi-branching temporal convolutional network for sepsis prediction},
  author={Wang, Zekai and Yao, Bing},
  journal={IEEE journal of biomedical and health informatics},
  volume={26},
  number={2},
  pages={876--887},
  year={2021},
  publisher={IEEE}
}
```


