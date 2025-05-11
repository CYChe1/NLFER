# NLFER: Multi-Branch Attention Cross-Fusion for Robust Facial Expression Recognition amidst Noisy Labels
This repository contains the official implementation of NLFER, which is a facial expression recognition network used for cross-fusion of multi-branch attention.
## Main requirements
* python
* torch
* torchvision
* numpy
* pandas
* timm
## Installation
* Clone this repo and install dependencies.
  ```python  
git clone https://github.com/CYChe1/NLFER
pip install -r requirements.txt  
```
## Datasets
Downloading the original images after obtaining official authorization for the mentioned datasets: [Affectnet](https://paperswithcode.com/dataset/affectnet), [RAF-DB](https://paperswithcode.com/dataset/raf-db), and [FERPlus](https://github.com/microsoft/FERPlus).
Allocating training and testing datasets. An example of this directory is shown in dataset/.
## Run
  ```python  
python main.py  
```
## Citations
If you find NLFER useful in your research, please consider citing:
  ```python  
@article{
   title = {NLFER: Multi-Branch Attention Cross-Fusion for Robust Facial Expression Recognition amidst Noisy Labels},
   url={https://github.com/CYChe1/NLFER},
   journal = {{The Visual Computer}}
}
```
