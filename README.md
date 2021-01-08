# PneumoniaDetectionDeepLearning

[![View](https://img.shields.io/badge/View%20in%20nbviewer-View%20-orange)](https://nbviewer.jupyter.org/github/mrtkp9993/PneumoniaDetectionDeepLearning/blob/main/notebook_wip.ipynb)
[![DOI](https://zenodo.org/badge/313400337.svg)](https://zenodo.org/badge/latestdoi/313400337)


Pneumonia detection using deep learning with Python and Tensorflow/Keras.

## Dataset

Kermany, Daniel; Zhang, Kang; Goldbaum, Michael (2018), “Labeled Optical Coherence Tomography (OCT) and Chest X-Ray Images for Classification”, Mendeley Data, V2, doi: 10.17632/rscbjbr9sj.2

[Link](https://data.mendeley.com/datasets/rscbjbr9sj/2)

Original task is a binary classification (normal/pneumonia):

```
data
└───train
│   └───normal
│       │   ...
│   └───pneumonia
│       │   ...
└───test
│   └───normal
│       │   ...
│   └───pneumonia
│       │   ...
```

I converted task to multi-class classification (normal / pneumonia_bacteria / pneumonia_virus):

```
data
└───train
│   └───normal
│       │   ...
│   └───pneumonia_bacteria
│       │   ...
│   └───pneumonia_virus
│       │   ...
└───test
│   └───normal
│       │   ...
│   └───pneumonia_bacteria
│       │   ...
│   └───pneumonia_virus
│       │   ...
```

## Flask App

One can run flask app with:

```
FLASK_APP=app.py flask run
```
