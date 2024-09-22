# Dataset-Condensation
# MPS2U (My Proposed Solution 2 - Updated) and Mutual Information

## Introduction

In the realm of machine learning, the quest for efficient and effective model training is paramount, especially when dealing with complex datasets. This document outlines **MPS2U**, an enhanced approach aimed at improving training efficiency and accuracy for synthetic datasets. By integrating advanced methodologies such as early stopping, predefined epoch limits, and the utilization of TensorFlow’s computational advantages, MPS2U demonstrates significant advancements over its predecessors.

Additionally, the **Mutual Info** approach is examined, showcasing how mutual information can be leveraged to select optimal samples, further refining model performance. This comparison highlights the strengths and efficiencies of MPS2U, particularly in handling challenging datasets like CIFAR-100. The benchmark algorithms for comparison are **MPS2** and **Gradient Matching**.

## MPS2U Methodology

MPS2U was designed to enhance training efficiency, accuracy, and versatility across multiple datasets, focusing on reducing training time while maintaining high performance. Key methodologies included:

1. **Early Stopping for Efficient Training**:  
   MPS2U integrated an **early stopping mechanism** to monitor model convergence, allowing training to be halted once performance metrics stabilized.

2. **Predefined Epoch Limits**:  
   Different datasets required tailored training strategies:
   - **MNIST and FashionMNIST**: Fewer epochs due to simpler structures.
   - **CIFAR-10 and CIFAR-100**: More epochs to capture intricate patterns.

3. **TensorFlow Utilization**:  
   Leveraged TensorFlow's static computation graph for faster tensor computations compared to PyTorch.

4. **ConvNet Architecture**:  
   Utilized a **Convolutional Neural Network (ConvNet)** to train on synthetic datasets from scratch.

5. **Enhanced Input Handling**:  
   Improved the input handling function to seamlessly process various datasets.

## Implementation

1. **ConvNet Training**:  
   Implemented a ConvNet to train on synthetic datasets.

2. **Visualization of Synthetic Data**:  
   Added a directory function to save and visualize generated synthetic samples.

3. **Robust Input Handling**:  
   Developed an improved input handling function to preprocess data for the ConvNet.

### Mutual Info Methodology

1. **Mutual Information Optimization**:  
   Adapted from MPS2, this approach uses mutual information to select the best 1000 samples, significantly enhancing data quality for training.

2. **Reinforcement Learning Integration**:  
   Combined mutual information selection with reinforcement learning to refine the sample selection process and improve overall model accuracy.

## Implementation

1. **ConvNet Training**:  
   Both MPS2U and Mutual Info utilized a ConvNet to train on synthetic datasets.

2. **Visualization of Synthetic Data**:  
   Added a directory function to save and visualize generated synthetic samples.

3. **Robust Input Handling**:  
   Developed improved input handling functions to preprocess data for both ConvNet and Mutual Info models.

## Results
## Synthetic Dataset Visualizations

<img src="https://raw.githubusercontent.com/likith-sg/Dataset-Condensation/main/MPS2U_DC.png" alt="MPS2U Dataset Visualization" width="600"/>
*Synthetic generated dataset visualization of MNIST, Fashion-MNIST, CIFAR-10, and CIFAR-100.*

<img src="https://raw.githubusercontent.com/likith-sg/Dataset-Condensation/main/MPS2_MI.png" alt="Mutual Info Dataset Visualization" width="600"/>
*Synthetic generated dataset visualization of MNIST, Fashion-MNIST, and CIFAR-10.*



### Performance Metrics
## NOTE : We have used Gradient Matching and MPS2 as the benchmark algorithm for comparison, with all developed algorithms evaluated against it.

| Metric                       | MPS2U       | Mutual Info   | MPS2         | Gradient Matching |
|------------------------------|-------------|---------------|--------------|-------------------|
| MNIST Test Accuracy          | 95%         | 88%           | 75%          | 98%               |
| Fashion-MNIST Test Accuracy   | 77%         | 77%           | 58%          | 83%               |
| CIFAR-10 Test Accuracy       | 31%         | 23%           | 31%          | 53%               |
| CIFAR-100 Test Accuracy      | 12%         | -             | -            | -               |
| Average Training Time        | 5 minutes   | 10 minutes    | 20 minutes   | 4 hours           |
| Computational Load           | Low (Integrated GPU) | High   | Low (Integrated GPU) | High              |

