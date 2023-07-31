# Spam Email Detection Machine Learning Project
[Summary](#Summary) &nbsp; &nbsp; [Data](#Data) &nbsp; &nbsp;  [Environment-Setup](#Environment-Setup) &nbsp; &nbsp; 
## Summary
This repository contains the code for a machine learning project that aims to detect spam emails using various techniques. The project consists of four Jupyter notebooks, and they should be run in the following order:

1. **EDA.ipynb**: This notebook performs Exploratory Data Analysis (EDA) on the dataset to gain insights and understand the data distribution.
2. **Feature_Engineering.ipynb**: In this notebook, extensive feature engineering techniques are applied to generate around 300 features from the original dataset. The notebook exports processed data into CSV files that will be used in the subsequent notebooks. You will also need LexVec Wikipedia Word Vectors in the data directory for running the codes for sentence embedding.
3. **Baseline_Model.ipynb**: The Baseline_Model notebook builds a baseline machine learning model for spam email detection. It requires the **Ensemble environment** to be installed, as it utilizes ensemble methods for the classification task.
4. **Neural_Networks.ipynb**: The Neural_Networks notebook focuses on building and training neural network models for spam email detection. It requires the **Deep Learning environment** to be installed for running the models.
   
## Data
The original source of the dataset is from the Programming Languages Group's website of University of Waterloo. The dataset contains 75,419 emails delivered to a particular server between April 2007 and Jul 2007. [Click here](https://plg.uwaterloo.ca/~gvcormac/treccorpus07/).
The data is available in csv from [here](https://www.kaggle.com/datasets/imdeepmind/preprocessed-trec-2007-public-corpus-dataset).

## Environment-Setup
To run the notebooks, you need to set up two different environments, one for the baseline model and the other for the neural networks. Here's how you can create the environments:

### Baseline Model Environment
Create a new virtual environment using conda:
   ```bash
   conda create -n ensemble python=3.8 numpy pandas matplotlib seaborn statsmodels scikit-learn=0.24.1 jupyter jupyterlab
   conda activate ensemble
```
For Windows users:
   ```bash
   pip install xgboost==1.1.1 mlxtend
```
For Mac users:
   ```bash
   conda install -c conda-forge xgboost=1.1.1 mlxtend
```
Optional - Creating a Jupyter kernel for this environment:
   ```bash
   ipython kernel install --name "ensemble" --user
```
### Neural Network Model Environment
Create the new empty environment named 'deeplearning'.
```bash
conda create -n deeplearning python=3.8
```
Activate the new environment.
```bash
conda activate deeplearning
```
Install all the basic packages we'll need (including jupyter notebook and lab).
```bash
conda install numpy=1.19.2 pandas=1.3.5 matplotlib jupyter jupyterlab pydot pillow seaborn
```
Note: you may get an initial frozen solve warning, but wait it out and packages should get installed using a flexible solve.
Mac Instructions: Install TensorFlow in this environment.
Mac users should install Tensorflow 2.7.0: 
```bash
conda install -c conda-forge tensorflow=2.7.0
```
Windows users should install Tensorflow 2.3.0: 
```bash
conda install -c conda-forge tensorflow=2.3.0
```
Install some more packages that we'll need in the TensorFlow Lecture.
```bash
conda install scikit-learn=0.24.1 nltk
conda install -c conda-forge gensim=3.8.3
```
Install PyTorch and TorchVision.
```bash
conda install -c pytorch pytorch=1.5.1 torchvision=0.6.1
```
Optional. Creating a Jupyter kernel for this environment (needed if you don't have nb_conda_kernels installed in your base environment):
```bash
ipython kernel install --name "deeplearning" --user
```

