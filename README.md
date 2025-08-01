# CIFAR-10-Convolutional-Neural-Network
Implementation of a custom Convolutional Neural Network (CNN) developed in Pytorch as part of the Neural Networks and Deep Learning module coursework @ QMUL.


# Architecture Overview

## 1. Stem (Initial Feature Extraction)

Purpose: Process the raw image into low-level feature maps.

2 convolutional layers with ReLU activation.

Average pooling to reduce spatial dimensions by x2.

## 2. Expert Branch (Dynamic Weight Generator)

Purpose: For each block, compute weights that determine how much each convolutional layer’s output contributes to the final block output.

Global Average Pooling over spatial dimensions.

Fully Connected Layer 1: Reduces dimension by factor r.

ReLU activation.

Fully Connected Layer 2: Outputs k values, where k = number of convolutions in the block.

Softmax: Converts values into normalized weights that sum to 1.

## 3. Block (Weighted Multi-Conv with Skip Connection)

Inputs:
Number of convolution layers (num_convs),
Input & output channels,
Reduction factor r for Expert Branch,
Flag to optionally halve spatial resolution.

Structure:

Pass input through Expert Branch to get weights [batch, num_convs].

Sequentially apply num_convs convolution → ReLU → BatchNorm operations.

Collect all convolution outputs in a list.

Weighted Sum: Multiply each convolution’s output by its Expert Branch weight and sum.

Skip Connection: Add transformed input (via 1×1 convolution) to the weighted sum.

Apply ReLU.

Optionally downsample via AvgPool if half_dim=True.

## 4. Backbone
Purpose: Stack multiple blocks to progressively extract higher-level features while increasing channels and reducing resolution.

Structure:

Block1: 1 conv, 32→60 channels, same spatial size.

Block2: 2 convs, 60→100 channels, halve size.

Block3: 2 convs, 100→160 channels, same size.

Block4: 2 convs, 160→240 channels, halve size.

Block5: 3 convs, 240→340 channels, same size.

Block6: 3 convs, 340→460 channels, halve size.

## 5. Classifier

Purpose: Convert final backbone features into class predictions.

Structure:

Adaptive average pooling to fixed 2×2 spatial size.

Flatten to a 1D vector.

Fully connected layers: 800 → 400 → 200 → number of classes (10).

ReLU activations and Dropout between layers for regularization.

---

Training, evaluation, and visualisation functions are based on utility code provided by the university lab sessions (adapted from my_utils.py in Neural Networks and Deep Learning Lab 3), which handle:

Model training loops with scheduler and GPU support.

Accuracy/loss plotting during training.

Evaluation of accuracy and loss on datasets.

Utility functions for displaying results and managing metrics.
