# RBFDenseUNet
A UNet architecture with Dense blocks, and an RBF output.

An RBF based artificial neural network used for the purpose of semantic image segmentation. The backbone of this model is a DenseUNet. The head of the convolutional neural network is an RBF layer. Inference can be done using Bayes Theorem, or Softmax. 

The dataset used for training and validation can be found in this repo: https://github.com/yalaudah/facies_classification_benchmark
I was inspired by this repo which is worth looking into: https://github.com/microsoft/seismic-deeplearning

To run: run TrainF3Seismic.py
The model is using the MxNet deep learning library, a LogCoshDiceLoss, and plotly for visualizations.
There is some commented code, these are ongoing experiments I am conducing with different variations of RBF such as "local linear models", and combining multiple kernels.

![alt text](https://github.com/jgcastro89/GenerativeDenseUNet/blob/main/screenshots/latentspace_vs_model_output.jpg)
Left: 4D vectors (latent space/embeddings) that are fed into the RBF layer. Right: Corresponding model output. Each pixel here corresponds to a 4D vector in the tetrahedron on the left. Both are color coded by label. For a nice introduction to ternary/tetrahedron plots, check out this awesome blog: https://www.cyentia.com/ternary-plots-for-visualizing-3d-data/

Papers: "Densely Connected Convolutional Networks",
"U-Net: Convolutional Networks for Biomedical Image Segmentation", 
"A Probabilistic RBF Network for Classification", 
"A Novel Kernel for RBF Based Neural Networks",
"Kernel Bayes’ Rule: Bayesian Inference with Positive
Definite Kernels",
"A survey of loss functions for semantic segmentation",
"RBF-Softmax: Learning Deep Representative
Prototypes with Radial Basis Function Softmax.",
"Cosine meets Softmax: A tough-to-beat baseline
for visual grounding".

https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.77.1134&rep=rep1&type=pdf

https://arxiv.org/pdf/1608.06993.pdf

https://arxiv.org/pdf/1505.04597.pdf

https://arxiv.org/pdf/2006.14822.pdf

https://www.cc.gatech.edu/~lsong/papers/FukSonGre12.pdf

https://www.hindawi.com/journals/aaa/2014/176253/

https://arxiv.org/pdf/1811.00410.pdf

https://openaccess.thecvf.com/content_cvpr_2018/papers/Huang_CondenseNet_An_Efficient_CVPR_2018_paper.pdf

https://arxiv.org/pdf/2009.06066.pdf

https://arxiv.org/pdf/1905.12200.pdf

https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123710290.pdf

https://arxiv.org/pdf/2004.13912.pdf
