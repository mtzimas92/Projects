# Projects

This repository contains a collection of projects that demonstrate my skills in Python programming, particularly in the fields of machine learning and mathematics.

Table of Contents

Machine Learning Projects

Project 1: General PCA functions for a standard data set (general_pca.py). The functions can be used freely with different datasets. Most of these functions were used in my Master's Thesis.

Project 2: Prediction of Flutter using NASTRAN generated data. This project is comprised of two .py files (nastran_flutter_feature_extraction.py and simple_nn_flutter.py). The first processes the .f06 file outputted from NASTRAN and creates three files: features.txt, labels.txt and binary_labels.txt. It is a very focused code that will only work for flutter data. If you have other .f06 files you would have to identify the common patterns in the file so you could make some easy substitutions.
The second file is my main implementation for the prediction of the dataset which was generated above. It is a Deep Neural Network with Tensorflow. It contains functions for loading/preprocessing the data, and creating/evaluating the DNN. These algorithms were the basis of my PhD.
...
Python for Math
Various implementations using Python of Math problems over the years. 
Finite Difference Analysis: Contains files which solve problems like the 1D wave parabolic PDE equation. 

This folder also contains general implementations of 1D heat conduction equation, or an implementaion of golden section search and a general file (math_functions.py) with some easy implementations of mathematical functions. 

...
Getting Started
You can download any of the files that you like. You can open them, change them and do whatever you want with them. Most of them either provide examples of how to use inside the file or can be used as a standalone file. 
You can also copy and paste functions that you find useful for your projects. 
One of the most useful files here is the useful_functions.py which contains multiple definitions for plotting functions in python. I have used all of them over the years, multiple times. You can copy and paste the functions, or use the whole file as is but most of the time your plots are going to require small changes to make them as you'd like. 


License
This project is licensed under the terms of the GNU General Public License v3.0.
