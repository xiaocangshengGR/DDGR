# DDGR
Code for ICML2023 paper, **DDGR: Continual Learning with Deep Diffusion-based Generative Replay**.
# Prerequisites
Our experiments are conducted on a Ubuntu 64-Bit Linux workstation, having NVIDIA GeForce RTX 3090 GPUs with 24GB graphics memory. Conducting our experiments requires the following steps.
```
conda create --name <ENV-NAME> python=3.7
source activate <ENV-NAME>
#Install cuda.
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
conda install mpi4py==3.1.4
conda install pillow
conda install opencv-python
pip install blobfile
pip install -r requirement.txt
```

# Datasets

**<font size=5>CIFAR100</font>**
```
Automatic download.
```
**<font size=5>ImageNet</font>**

You should download the ImageNet manually and process the images into "src/data/datasets/ImageNet/imagenet-1000/".
The directory is:
```
├── py-DDGR1.0                               # The main code directory
│   ├── src                
│   │  ├── data 
│   │  │  ├── datasets
│   │  │  │  ├── ImageNet
│   │  │  │  │  ├── imagenet-1000
│   │  │  │  │  │  ├── train
│   │  │  │  │  │  │  ├── n01440764
                          ...
│   │  │  │  │  │  ├── val
│   │  │  │  │  │  │  ├── n01440764
                          ...
```
Then run generate_imagenet.py and generate_imagenet_class.py in order.
```
python generate_imagenet.py
python generate_imagenet_class.py
```
**<font size=5>CORe50</font>**

The dataset will be downloaded automatically. You can also download "core50_imgs.npz", "labels.pkl","LUP.pkl" and "paths.pkl" manually into "/src/data/datasets/core/core50CIREP".
# Usage Example 
```
python run_cifar100.py
```
# Code Overview
The file structure of directory is as follows:
```
.
├── py-DDGR1.0                            # The main code directory
│   ├── src                
│   │  ├── data                           # The directory contains the dataset.
│   │  ├── framework                      # The directory contains the framework of continual learning.
│   │  ├── methods                        # The directory contains the codes of DDGR.
│   │  ├── models                         # The directory contains the defined models.
│   │  ├── results                        # The directory contains the results.
│   │  ├── utilities                      # The directory contains some defined functions.
│   │  ├── config.init                    # The configuration file.
│   ├── generate_imagenet.py              # The file to convert images to 64*64.
│   ├── generate_imagenet_class.py        # The file to generate classes.
│   ├── main.py                           # The main code file.
│   ├── README.md                         # The readme file
│   ├── requirements.txt                  # The file specifies the required environments
│   ├── run_cifar100.py                   # The file of example.
```
