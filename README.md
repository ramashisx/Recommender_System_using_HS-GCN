# Recommender System using HS-GCN model
This is the project submission codes for semester project for GML-FA course IIT Kharagpur
### By Ramashish Gupta, Shakti Prasad Nanda and Shashank Sundi

# [colab link](https://colab.research.google.com/drive/1E01Zh7z9Jyr7RGL9wj2TwZQRnpbHGC1x?usp=sharing)

HS-GCN: Hamming Spatial Graph Convolutional Networks for Recommendation.

## File specification
* preprocessing.py : loads the raw data in path `./raw_data`, and the results are saved in path `./para` and then obtains the triplets for model training, and the results are saved in path `./para`.
* HSGCN_model.py : implements the model framework of HS-GCN.
* model_train.py : the training process of model.
* model_test.py : the testing process of model.

## Usage
* Execution sequence

  The execution sequence of codes is as follows: preprocessing.py--->model_train.py--->model_test.py
