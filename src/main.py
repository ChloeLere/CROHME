#!/usr/bin/env python3
from data import get_data, format_data_for_svc
import numpy as np
from svc import Svc
from cnn import Cnn
from evaluation import *
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

def main():
    #get data 
    train, val, test, class_names_test = get_data()

    #SVC
    X_train_flat, y_train, X_test_flat, y_test = format_data_for_svc(train, test)
    svc = Svc(84)
    param_grid = {'C': np.array([0.1, 1, 10, 100, 1000]), 'gamma': np.array([0.0001, 0.001, 0.01, 0.1, 10])}
    svc.compile_model(X_train_flat, y_train, param_grid)
    svc.display_grid_search_info()
    svc.display_history()

    y_test_pred = svc.predict(X_test_flat)

    accuracy(y_test, y_test_pred)
    create_confusion_matrix(y_test, y_test_pred, "confusion_matrix_svc6.png", class_names_test)

    #CNN
    param_grid = {
        'learning_rate': [0.01, 0.001, 0.0001],
        'epoch': [8, 16, 25, 32]
    }
    cnn = Cnn(84)
    _, _, _, best_history = cnn.grid_search(param_grid, train, val, test)
    cnn.display_history(best_history)

    all_predictions, all_actual_labels = cnn.predict(test)
    model = cnn.get_model()
    model.save('cnn_model.keras')
    sensitivity_specificity(all_predictions, all_actual_labels)
    loss_accuracy(model, test)
    create_confusion_matrix(all_actual_labels, all_predictions, "confusion_matrix_cnn6.png", class_names_test)
