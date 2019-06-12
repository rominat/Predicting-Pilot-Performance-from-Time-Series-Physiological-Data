#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
import seaborn as sns


train = pd.read_csv('train.csv')
Test = pd.read_csv('test.csv')

#feature selection
Feature_eeg = ["eeg_fp1", "eeg_f7", "eeg_f8", "eeg_t4", "eeg_t6", "eeg_t5",
              "eeg_t3", "eeg_fp2", "eeg_o1", "eeg_p3", "eeg_pz", "eeg_f3",
              "eeg_fz", "eeg_f4", "eeg_c4", "eeg_p4", "eeg_poz", "eeg_c3",
              "eeg_cz", "eeg_o2"]
Feature_ecg =["ecg"]
Feature_r = ["r"]
Feature_gsr = ["gsr"]
Feature_ERG = ["ecg", "r", "gsr"]
Feature_some = ['time',"eeg_fp1", "eeg_f7", "eeg_f8", "eeg_t4", "eeg_t6", "eeg_t5",
              "eeg_t3", "eeg_fp2", "eeg_o1", "eeg_p3", "eeg_pz", "eeg_f3",
              "eeg_fz", "eeg_f4", "eeg_c4", "eeg_p4", "eeg_poz", "eeg_c3",
              "eeg_cz", "eeg_o2", "ecg", "r", "gsr"]
Feature_all =["crew", "seat", "time","experiment","eeg_fp1", "eeg_f7", "eeg_f8",
              "eeg_t4", "eeg_t6", "eeg_t5",
              "eeg_t3", "eeg_fp2", "eeg_o1", "eeg_p3", "eeg_pz", "eeg_f3",
              "eeg_fz", "eeg_f4", "eeg_c4", "eeg_p4", "eeg_poz", "eeg_c3",
              "eeg_cz", "eeg_o2", "ecg", "r", "gsr"]


#Encoding categorical data based on "event"label
# A = baseline, B = SS, C = CA, D = DA
encode_exp = {'CA': 0, 'DA': 1, 'SS': 2, 'LOFT': 4}
encode_event = {'A': 3, 'B': 2, 'C': 0, 'D': 1}
labels_exp = {v: k for k, v in encode_exp.items()}
labels_event = {v: k for k, v in encode_event.items()}
train['event'] = train['event'].apply(lambda x: encode_event[x])
train['experiment'] = train['experiment'].apply(lambda x: encode_exp[x])
Test['experiment'] = Test['experiment'].apply(lambda x: encode_exp[x])
train['event'] = train['event'].astype('int8')
train['experiment'] = train['experiment'].astype('int8')
Test['experiment'] = Test['experiment'].astype('int8')     


#Sample train and test data
Test_s = Test.sample(frac=0.1, replace=True, random_state=0)
Train = train.sample(frac=0.4, replace=True, random_state=0)

#Identify target variables
Y = Train['event']
#Separate ID from Test
Id = Test_s['id']

#Feature Scaling
scaler = StandardScaler()
Train[Feature_all] = scaler.fit_transform(Train[Feature_all])
Test_s[Feature_all] = scaler.transform(Test_s[Feature_all])

#Apply PCA
pca = PCA(.90)
pca.fit(Train[Feature_all])
pc=pca.n_components_
print(pc)
TrainFinal = pca.transform(Train[Feature_all])
TestFinal = pca.transform(Test_s[Feature_all])
Variance = pca.explained_variance_ratio_
print(Variance)

#Use SMOTE (Synthetic Minority Over-sampling Technique) to fix imbalance
TrainFinal, Y = SMOTE().fit_resample(TrainFinal, Y.ravel())

#Split data into training set and test set
X_Train, X_Test, Y_Train, Y_Test = train_test_split(TrainFinal, Y, test_size = 0.3,
                                                    random_state = 0)                                                   
# Create an instance of the chosen classifier.
params = {
    'objective': 'multiclass',  
    'num_class': 4,
    "metric" : "multi_error",
    "learning_rate" : 0.1,
    "min_child_weight" : 40,
    "feature_fraction" : 0.8,
    "reg_alpha": 0.15,
    "reg_lambda": 0.15,
    'n_gpus': 0}
classifier = XGBClassifier(objective= 'multiclass',
    num_class= 4,
    metric = "multi_error",
    learning_rate = 0.1,
    min_child_weight = 40,
    feature_fraction = 0.8,
    reg_alpha= 0.15,
    reg_lambda= 0.15)
n_classifier = len(classifier)

# Xgboost Classifier and print accuracy score
for index, (name, classifier) in enumerate(classifier.items()):
    classifier.fit(X_Train, Y_Train)
    
Y_pred = classifier.predict(X_Test)
Y_pred = pd.DataFrame(Y_pred)
Y_pred.describe()
accuracy = accuracy_score(Y_Test, Y_pred)
print("Accuracy (Train) for %s: %0.1f%% " % (name, accuracy * 100))

# get probabilities
probas = classifier.predict_proba(X_Test)
probas
#  Confusion Matrix
cm = confusion_matrix(Y_Test, Y_pred)
print(cm)

#k-Fold Cross Validation with Parameter Tuning
accuracies = cross_val_score(estimator = classifier, X = X_Train, y = Y_Train,
                             cv = 10, verbose=0)
cv_accuracy=accuracies.mean()
print(" Cross Validation Mean Accuracy equals %0.1f%%  " % (cv_accuracy * 100))
print(" Cross Validation Standard Deviation Accuracy equals %0.1f%% " % (accuracies.std() * 100))

#Plot Confusion Matrix result
def plot_confusion_matrix(cm, classes, normalized=True, cmap='bone'):
    plt.figure(figsize=[7, 8])
    norm_cm = cm
    if normalized:
        norm_cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        sns.heatmap(norm_cm, annot=cm, fmt='g', xticklabels=classes, yticklabels=classes, cmap=cmap)
        plt.savefig('confusion-matrix.png')
plot_confusion_matrix(cm, ['C', 'D', 'B', 'A'])

#probability prediction of ID's in Test Data
Y_pred1 = classifier.predict(TestFinal.values)
Y_pred1= classifier.predict_proba(TestFinal)
proba = Y_pred1
print(Y_pred1)

#get prediction
sub = pd.DataFrame(Y_pred1, columns=['A', 'B', 'C', 'D'])
sub['id'] = Id
cols = sub.columns.tolist()
cols = cols[-1:] + cols[:-1]
sub = sub[cols]
sub.to_csv("Test_prob.csv", index=False)
