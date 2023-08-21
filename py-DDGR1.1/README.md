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

# Usage
To conduct our experiments quickly, you need the following steps.

**<font size=5>CIFAR100</font>**
```
python run_cifar100.py
```
**<font size=5>ImageNet</font>**

You should download the ImageNet manually and process the images into "src/data/datasets/ImageNet/imagenet-1000/".
The directory is:
```
├── py-DDGR                               # The main code directory
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
python run_imgnet1000.py
```
**<font size=5>CORe50</font>**

You can directly run the experiments on CORe50 by the following command:
```
python run_core50.py
```
The dataset will be downloaded automatically. You can also download "core50_imgs.npz", "labels.pkl","LUP.pkl" and "paths.pkl" manually into "/src/data/datasets/core/core50CIREP". And then run the above command.
# Code Overview
The file structure of directory is as follows:
```
.
├── py-DDGR                               # The main code directory
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
│   ├── run_cifar100.py                   # The file to conduct experiments on CIFAR100.
│   ├── run_core50.py                     # The file to conduct experiments on CORe50.
│   ├── run_imgnet1000.py                 # The file to conduct experiments on ImageNet.
```
