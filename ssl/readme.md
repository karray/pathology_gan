## Intorduction

### Motivation

Recent applications of GANs have shown that they can produce realistic samples. It was also shown that GANS can be applied for [semi-supervised learning (SSL)](https://arxiv.org/abs/1606.01583), which may improve accuracy of common classifiers. Besides, synthetic data can be labelled by a strong classifier and can be then extend the initial dataset. Since synthetically generated does not include patient data or privacy concerns, it will also allow medical institutions to share data they generate with other institutions, creating millions of different combinations that can be used to accelerate the work.

### Objectives
The first part of this research project was focused on implementation of a GAN model and on conducting initial results. Despite some limits of implemented model, the results were promising.

This part of the project will be concentrated on further improvement of the model. Besides,  the main goal of this project is to implement and to compare different SSL models with the initial classifier.

### Dataset description
The GAN models were trained on PatchCamelyon ([PCam](https://github.com/basveeling/pcam)) dataset. PCam is derived from the Camelyon16 Challenge and contains 96x96px patches extracted from WSI. The dataset is divided into a training, a validation and test set. There is no overlap in WSIs between the splits, and all splits have a 50/50 balance between positive and negative examples. The dataset was also cleaned such that background and blurred patches were filtered out.

A positive label indicates that the center 32x32px region of a patch contains at least one pixel of tumor tissue. 

Since tumor tissue in the outer region of the patch does not influence the label, the patches were cropped to 64x64px for this project.


## Background

### GAN
GANs consist of two nueral networks: Generator (G) and Discriminator (D). This networks are playing a minimax non-cooperative game against each other: G takes a random noise as input and tries to produce samples in such a way that D is unable to determine if they come from real dataset or it is generated images. D learns in a supervised manner by looking at real and generated samples and labels telling where those samples came from.

### Semi-supervised GAN


### Methodology
SSL model vs CNN classifire. The discriminator in SSL has almost the same architecture as the classifier which has 3

### Evaluation
preliminary results

![title](avg_ssl_vs_clf.png)