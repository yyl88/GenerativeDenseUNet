# Interpretable Deep Learning with RBFDenseUNet
A UNet architecture with Dense blocks, and an RBF output for interpretable deep learning.

An RBF based artificial neural network used for the purpose of semantic image segmentation and interpretation of seismic data. The backbone of this model is a DenseUNet. The head of the convolutional neural network is an RBF layer. Inference can be done using Bayes Theorem, or Softmax. 

The dataset used for training and validation can be found in this repo: https://github.com/yalaudah/facies_classification_benchmark
I was inspired by this repo which is worth looking into: https://github.com/microsoft/seismic-deeplearning

To run: run TrainF3Seismic.py
The model is using the MxNet deep learning library, a LogCoshDiceLoss, and plotly for visualizations.
There is some commented code, these are ongoing experiments I am performing with different variations of RBF such as "local linear models", and combining multiple kernels.

![alt text](https://github.com/jgcastro89/GenerativeDenseUNet/blob/main/screenshots/SharedScreenshot0.jpg)
![alt text](https://github.com/jgcastro89/GenerativeDenseUNet/blob/main/screenshots/SharedScreenshot1.jpg)
![alt text](https://github.com/jgcastro89/GenerativeDenseUNet/blob/main/screenshots/SharedScreenshot2.jpg)
![alt text](https://github.com/jgcastro89/GenerativeDenseUNet/blob/main/screenshots/SharedScreenshot3.jpg)
The CNN (DenseUNet) produces 4 feature maps. Then slices along axis 1 (takes a pixel from each feature map) and feeds this 4D vector into the RBF layer. 
This allows the model to remain a fully convolutional neural network. 

![alt text](https://github.com/jgcastro89/GenerativeDenseUNet/blob/main/screenshots/SharedScreenshot4.jpg)
The image above is the embedding space (4D points/coordinates for each pixel in the image) as a result of training this model for a few epochs. The CNN (DenseUNet) is tasked with generating feature maps that the RBF layer can correctly bucket into one of the RBF nodes. For a nice introduction to ternary/tetrahedron plots, check out this awesome blog: https://www.cyentia.com/ternary-plots-for-visualizing-3d-data/

![alt text](https://github.com/jgcastro89/GenerativeDenseUNet/blob/main/screenshots/181996825_5461193370621018_8800196505991060229_n.jpg)
Each pixel here corresponds to a 4D point in the tetrahedron (above).

![alt text](https://github.com/jgcastro89/GenerativeDenseUNet/blob/main/screenshots/181936468_5461194637287558_3655243639396744276_n.jpg)
Bayes Theorem can be used for inferece. This image is showing REAL posterior probabilites for each pixel. 

![alt text](https://github.com/jgcastro89/GenerativeDenseUNet/blob/main/screenshots/183034536_5461193310621024_6251143271430111991_n.jpg)
Ground truth (compare to model predictions with Bayes inference above). 

![alt text](https://github.com/jgcastro89/GenerativeDenseUNet/blob/main/screenshots/SharedScreenshot6.jpg)
Number of parameters.

TODO: Calculate segmentation metrics. 

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
