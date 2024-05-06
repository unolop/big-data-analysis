import statistics as st
import seaborn as sns
import os 
import random
from glob import glob
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score, classification_report
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from matplotlib import ticker

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)  # type: ignore
    torch.cuda.manual_seed_all(seed)  # type: ignore
    torch.backends.cudnn.deterministic = True  # type: ignore
    torch.backends.cudnn.benchmark = False  # type: ignore
    
def evaluate_metrics(cm, label=None):
    n_classes = cm.shape[0]
    if not label:
        label = np.arange(n_classes)
    avg_acc = np.sum(np.diag(cm))/np.sum(cm)
    avg_prec = 0
    avg_tpr = 0
    avg_tnr = 0
    avg_f1 = 0
    for i in range(n_classes):
        
        tp = cm[i, i]
        fn = sum(cm[i, :]) - tp
        fp = sum(cm[:, i]) - tp
        tn = sum(sum(cm)) - tp - fn - fp
        
        prec = tp / (tp + fp)
        tpr = tp / (tp + fn)
        tnr = tn / (tn + fp)
        f1 = st.harmonic_mean([prec, tpr])        

        avg_prec += prec/n_classes
        avg_tpr += tpr/n_classes
        avg_tnr += tnr/n_classes
        avg_f1 += f1/n_classes
    #     print(f"Class {label[i]}: Precise = {prec:.4f}, Sensitivity = {tpr:.4f}, F1 = {f1:.4f} Specificity = {tnr:.4f}")
    # print(f"Average(Macro): ====================================\n\
    #     Accuracy = {avg_acc:.4f}\n\
    #     Precise = {avg_prec:.4f}\n\
    #     Sensitivity = {avg_tpr:.4f}\n\
    #     Specificity = {avg_tnr:.4f}\n\
    #     F1 = {avg_f1:.4f}")
    return avg_acc, avg_prec, avg_tpr, avg_tnr, avg_f1

def cm_analysis(y_true, y_pred, labels, classes, ymap=None, figsize=(12,12)):
    """
    Generate matrix plot of confusion matrix with pretty annotations.
    The plot image is saved to disk.
    args: 
      y_true:    true label of the data, with shape (nsamples,)
      y_pred:    prediction of the data, with shape (nsamples,)
      filename:  filename of figure file to save
      labels:    string array, name the order of class labels in the confusion matrix.
                 use `clf.classes_` if using scikit-learn models.
                 with shape (nclass,).
      classes:   aliases for the labels. String array to be shown in the cm plot.
      ymap:      dict: any -> string, length == nclass.
                 if not None, map the labels & ys to more understandable strings.
                 Caution: original y_true, y_pred and labels must align.
      figsize:   the size of the figure plotted.
    """
    sns.set(font_scale=1.5)

    if ymap is not None:
        y_pred = [ymap[yi] for yi in y_pred]
        y_true = [ymap[yi] for yi in y_true]    
        labels = [ymap[yi] for yi in labels]    
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_sum = np.sum(cm, axis=1, keepdims=True)
    cm_perc = cm / cm_sum.astype(float) * 100
    annot = np.empty_like(cm).astype(str)
    nrows, ncols = cm.shape
    for i in range(nrows):
        for j in range(ncols):
            c = cm[i, j]
            p = cm_perc[i, j]
            if i == j:
                s = cm_sum[i]
                annot[i, j] = '%.1f%%\n%d/%d' % (p, c, s)
            elif c == 0:
                annot[i, j] = ''
            else:
                annot[i, j] = '%.1f%%\n%d' % (p, c)
    cm = confusion_matrix(y_true, y_pred, labels=labels, normalize='true')
    cm = pd.DataFrame(cm, index=labels, columns=labels)
    cm = cm * 100
    cm.index.name = 'True Label'
    cm.columns.name = 'Predicted Label'
    fig, ax = plt.subplots(figsize=figsize)
    plt.yticks(va='center')
    sns.heatmap(cm, annot=annot, fmt='', ax=ax, xticklabels=classes, cbar=True, cbar_kws={'format':ticker.PercentFormatter()}, yticklabels=classes, cmap="Blues",  vmin=0, vmax=100)