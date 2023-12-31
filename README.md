# DeepMonitoring
- DeepMonitoring: a deep learning-based monitoring system for assessing the quality of cornea images captured by smartphones
- create time: 2023.06.14

# Introduction
This repository contains the source code for developing a deep learning system (DeepMonitoring) for the automated detection of defocused images, overexposed images, underexposed images, poor cornea position images, incompletely exposed cornea images, and high-quality images generated by smartphones. 
DeepMonitoring can be used to detect and filter out low-quality cornea images generated by smartphones, facilitating the application of smartphone-based AI diagnostic systems in real-world settings, particularly in the scenario of corneal disease self-screening.

# Prerequisites
- Ubuntu: 18.04 lts
- Python 3.7.8
- Pytorch 1.6.0
- NVIDIA GPU + CUDA_10.0 CuDNN_7.5

This repository has been trained and tested on four NVIDIA RTX2080Ti. Configurations (e.g batch size, image patch size) may need to be changed on different platforms.

# Installation
Other packages are as follows:
- pytorch: 1.6.0
- wheel: 0.34.2
- yaml: 0.2.5
- scipy: 1.5.2
- joblib: 0.16.0
- opencv-python: 4.3.0.38
- scikit-image: 0.17.2
- numpy: 1.19.1
- matplotlib：3.3.1
- sikit-learn：0.23.2

# Install dependencies
pip install -r requirements.txt

# Usage
- The file "DeepMonitoring_training_v1.py" in /DeepMonitoring is used for our models training.
- The file "DeepMonitoring_testing_v1.py" in /DeepMonitoring is used for testing.

The training and testing are executed as follows:
# Train Swin-Transformer on GPU
python DeepMonitoring_training_v1.py -a 'Transform_base'

# Train ConvNeXt on GPU
python DeepMonitoring_training_v1.py -a 'convnext_base'

# Train RepVGG on GPU
python DeepMonitoring_training_v1.py -a 'RepVGG_A1'

# Train MobileNet on GPU
python DeepMonitoring_training_v1.py -a 'mobilenetv3_large_075'

# Evaluate four models of Swin-Transformer, ConvNeXt, RepVGG, and MobileNet at the same time on GPU
python DeepMonitoring_testing_v1.py


The expected output: print the classification probabilities for defocused images, overexposed images, underexposed images, image of poor cornea position , image of not fully opened eye, and high-quality image.


* Please feel free to contact us for any questions or comments: Zhongwen Li, E-mail: li.zhw@qq.com or Jiewei Jiang, E-mail: jiangjw924@126.com.
