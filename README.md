# Image2Flow

This repository contains the code for the paper:

Tina Yao, Endrit Pajaziti, Silvia Shievano, Michael Quail, Jennifer Steeden & Vivek Muthurangu. Image2Flow: Fast calculation of pulmonary artery flow fields directly from 3D cardiac MRI using graph convolutional neural networks 

<img src="https://github.com/Ti-Yao/Image2Flow/blob/main/images/Thumbnail.png" width="350"/>

## Volume Mesh Generation and CFD simulation
Volume mesh generation for this paper was done using TetGen in VMTK. Additionally, CFD simulations were performed in Ansys Fluent. Code for both mesh generation and CFD simulation can be found in the folder called Fluent. The code was adapted from [GitHub repository](https://github.com/EndritPJ/CFD_Machine_Learning) [1]


## Image2Flow deep learning architecture
The code for the Image2Flow architecture can be found in the model folder. It is a hybrid image and graph convolutional neural network that was adapted from [GitHub repository](https://github.com/EndritPJ/CFD_Machine_Learning)) [2]

<img src="https://github.com/Ti-Yao/Image2Flow/blob/main/images/Figure1.png" width="550"/>



### Reference

[1] Pajaziti E, Montalt-Tordera J, Capelli C, Sivera R, Sauvage E, Quail M, Schievano S, Muthurangu V. Shape-driven deep neural networks for fast acquisition of aortic 3D pressure and velocity flow fields. PLOS Computational Biology. 2023 Apr 24;19(4):e1011055.

[2]  Kong F, Wilson N, Shadden S. A deep-learning approach for direct whole-heart mesh reconstruction. Medical image analysis. 2021 Dec 1;74:102222.
