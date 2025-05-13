# NLFER: Multi-Branch Attention Cross-Fusion for Robust Facial Expression Recognition amidst Noisy Labels
This repository contains the official implementation of NLFER, which is a facial expression recognition network used for cross-fusion of multi-branch attention.
![image](https://github.com/CYChe1/NLFER/blob/main/img/NLFER.png)
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
## Preparation
* create conda environment (we provide requirements.txt)
* Data Preparation
  Download [RAF-DB](https://paperswithcode.com/dataset/raf-db) dataset, and make sure it have a structure like following:
  ```python  
   - data/raf-basic/
 	 EmoLabel/
 	     list_patition_label.txt
 	 Image/aligned/
 	     train_00001_aligned.jpg
 	     test_0001_aligned.jpg
 	     ... 
  ```
* Allocating training and testing datasets. An example of this directory is shown in dataset/.
## Run
* Train on RAF-DB dataset:
```python  
python main.py  
```
## Citations
If you find our work useful in your research, please consider citing:
```python  
@article{
   title = {NLFER: Multi-Branch Attention Cross-Fusion for Robust Facial Expression Recognition amidst Noisy Labels},
   url={https://github.com/CYChe1/NLFER},
   journal = {{The Visual Computer}}
}
```
