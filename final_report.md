# Audio Classification with UrbanSound8K
111062233 余承祐

## Overview

In everyday life, sounds provide important contextual cues—such as siren, dog barks, or construction noise—that carry strong semantic meaning for humans. However, to machines, these are merely complex signals. By developing an Environmental Sound Classification (ESC) system, we aim to enable machines to recognize common ambient sounds, which has potential applications in smart surveillance, autonomous vehicles, and smart city systems. 

The common approach for audio classification tasks is to develop a CNN-based model, sometimes specifically referred to as an Audio Neural Network (ANN). While they are powerful, deep learning models are considered computationally expensive. We aim to explore a more lightweight and interpretable alternative by using traditional machine learning models with well-crafted audio features. 

### Workflow

Here we introduce our workflow in this project. 
1. **Data processing & Feature engineering**: 

    Raw audio files are preprocessed and transformed into structured numerical features. This includes extracting MFCCs, spectral, temporal, and chroma-based features.

2. **Model selection**: 

    Several traditional machine learning models are selected as candidates based on their performance and interpretability, including Random Forest, SVM, and Boosting.

3. **Model training**: 

    Models are trained with hyperparameter tuning using cross-validation. Evaluation metrics include overall accuracy and per-class F1-score to ensure both global and class-wise performance.

4. **Ensemble models**: 

    A stacking ensemble strategy is applied to combine the strengths of individual models and improve overall performance.
    
![alt text](images/new_flow.png)

---

## Introduction to UrbanSound8K

[UrbanSound8K](https://urbansounddataset.weebly.com/urbansound8k.html) is a dataset with 8732 labeled audio file, each file is shorter than 4 seconds. There are **10 classes** in the dataset.
| Label |    Class Name    | Label |   Class Name  |
|:-----:|:----------------:|:-----:|:-------------:|
|   0   |  Air conditioner |   5   | Engine Idling |
|   1   |     Car horn     |   6   |    Gun shot   |
|   2   | Children playing |   7   |   Jackhammer  |
|   3   |     Dog bark     |   8   |     Siren     |
|   4   |     Drilling     |   9   |  Street music |

The distribution of classes is shown below. Most classes contain around 1,000 samples. However, there is a noticeable imbalance in the `car_horn` and `gun_shot` classes, which each contain only about 400 samples. Also, the dataset is already splitted into **10 folds** for performing cross validation. 

![alt text](images/class_dist.png)

---

## Data Processing & Feature Engineering

We use `librosa` to load each audio into a numeric array with the following setting:
- Sample rate: 22000 Hz (default setting).

- Target duration: 4 seconds (maximum duration among all samples).

We then extract a total of 35 audio features for each file, including:

- **MFCC** (13-dim): the timbre and spectral envelope of the sound.

- **Delta MFCC** (13-dim): the first-order derivative of MFCC, representing temporal dynamics.

- **Zero Crossing Rate (ZCR)**: how frequently the signal changes sign, often indicating noisiness.

- **RMS Energy**: the root mean square of the signal, indicating perceived loudness.

- **Spectral features** (7-dim): describe the distribution and shape of the frequency spectrum:
    - **Spectral centroid**: the center of mass of the spectrum, indicating brightness.

    - **Spectral bandwidth**: the width around the centroid, capturing frequency spread.

    - **Spectral contrast**: energy difference between peaks and valleys in the spectrum.

    - **Spectral flatness**: how much noise-like a sound is.

    - **Spectral rolloff**: the frequency below which a fixed percentage (usually 85%) of the spectral energy is contained.

    - **Chroma STFT**: pitch class energy distribution.

    - **Tonnetz**: encodes tonal relations such as harmonic and melodic intervals. 

Since different audio features vary in scale, we normalize all features using standard scaling (zero mean and unit variance). This helps improve model performance and avoids bias from dominant features. Importantly, scaling is performed separately within each fold to avoid data leakage.


---

## Model Training

To build a robust classifier, we go through two main stages: **model selection**, where we compare different machine learning algorithms, and **model tuning**, where we fine-tune each model’s hyperparameters using cross-validation. This helps us identify the most effective setup for environmental sound classification.

### Model Selection

To further improve the model performance, we'd like to adopt the stacking ensemble method, and now we will choose the candiates, start by evaluating six traditional machine learning models:
- Logistic Regression

- K-Nearest Neighbor (KNN)
- Support Vector Machine with RBF kernel (SVM-RBF)
- Random Forest (RF)
- LightGBM (LGBM)
- XGBoost (XGB)

The models are fit to the data with their default hyperparameters. Their performance will be evaluated by performing 10-fold cross validation and calculate the overall accuracy and per-class F1 score among all of the 10 validation results. The figures below show the performance across these models.
- The **overall accuracy** of each model.
    - **SVM-RBF** achieves the highest accuracy (60.3%), demonstrating strong overall predictive power.

    - **Random Forest** (58.6%) and **LightGBM** (58.5%) follow closely, indicating reliable performance.

    - **Logistic Regression**, while not the top performer (55.9%), remains a reasonable candidate for low-variance meta-learning.

    ![alt text](images/model_selection_acc.png)

- The **per-class F1-scores**, offering insights into how well each model handles class-wise prediction.
    - **SVM-RBF** and **tree-based models (RF, LGBM, XGB)** consistently perform well across most classes, particularly excelling on difficult classes like `gun_shot` and `siren`. The diversity in their strengths makes them ideal candidates for base learners.

    - **Logistic Regression** shows moderate but stable performance, suitable for serving as the meta-learner in stacking, where its simplicity helps to avoid overfitting when combining predictions.

    ![alt text](images/model_selection_F1.png)

Based on this analysis, we selected **SVM-RBF, Random Forest, and LightGBM** as base learners due to their complementary strengths and competitive individual performance. We use **Logistic Regression** as the meta-learner to combine their predictions in a robust and interpretable way.

---

### Model Tuning

