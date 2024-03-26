# A Deep Learning Approach for Direct Mesh Reconstruction of Intracranial Arteries

This repository is based on the PyTorch implementation of Voxel2Mesh architecture proposed in *Voxel2Mesh: 3D Mesh Model Generation from Volumetric Data*; Udaranga Wickramasinghe, Edoardo Remelli, Graham Knott and Pascal Fua; MICCAI 2020. 

## Abstract

The Circle of Willis (CoW) is a group of vessels connecting major circulations of the brain. Its vascular geometry is believed to influence the onset and outcome of serious neurovascular pathologies. These geometric features can be obtained from surface meshes to capture vessel topology and morphology. A recent deep learning technique to handle non-Euclidean data, such as meshes, is Geometric Deep Learning (GDL). To this end, this study aimed to explore a GDL-based approach to directly reconstruct surface meshes of the CoW from magnetic resonance angiography images, thereby eliminating the traditional postprocessing steps required to obtain such a mesh from volumetric representations. The network architecture includes both convolutional and graph convolutional layers, allowing it to operate with images and meshes at the same time. It takes as input an image volume and a template mesh and outputs a 3D surface mesh. Experiments were performed on five crops representing different vessels and bifurcations to capture both stability and variability within the CoW. The results showed that anatomy-specific template input meshes and enhancement of the image feature representation increase the accuracy of the reconstruction. Moreover, incorporating the curvature characteristics of the meshes showed promising capability of handling complex geometries and sharp edges. However, achieving a consistent performance across CoW regions remains a challenge.

## Architecture
  
<p class="aligncenter">
    <img src="./images/networkf.png">
</p>
Fig. 1. Modified Voxel2Mesh network pipeline. The architecture takes as input a 3D cropped vessel or bifurcation and a template mesh. It predicts a voxel-wise segmentation and surface meshes. It is composed of a CNN that extracts image features and communicates at each level to a GCN decoder to deform the template mesh. At each step of the mesh decoding, it receives features from both the encoder and decoder of the CNN. The mesh is deformed by adding vertices only where needed.


## Implementation details

All models are based on PyTorch (v1.11.0), and PyTorch3d (v0.7.2). They were trained and evaluated on Nvidia GP102 Titan X (12GB) using CUDA version 11.3. The experiments were tracked with Weights and Biases. For further see the environment.yml file.

## Dataset
To obtain information from the different components of the CoW, Time of Flight MRA images were used. The open-source dataset was provided by the “Topology-Aware Anatomical Segmentation of the Circle of Willis for CTA and MRA” challenge or [TopCoW](https://topcow23.grand-challenge.org/) for short. It was organized in association with the MICCAI 2023. 

## Running Experiments

&emsp; Step 1: Update config.py. Set the path to the dataset and the directory to save the results.
&emsp; Step 2: Preprocess data, and store in  ```data_crops{}``` folders.
&emsp; Step 3: Execute ```python main.py``` and to training or evaluate network with pretrained weights. 

