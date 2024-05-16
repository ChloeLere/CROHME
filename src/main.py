#!/usr/bin/env python3
from data import get_data, format_data_for_svc
import numpy as np
from sklearn.metrics import f1_score
import Levenshtein as lev
from sklearn.model_selection import GridSearchCV 
from sklearn.model_selection import train_test_split 
from sklearn.metrics import accuracy_score 
from sklearn.metrics import classification_report
from svc import Svc
from cnn import Cnn
from evaluation import *
from encoder import build_model, compile_model

#def main():
#    train, val, test = get_data()
#    autoencoder = build_model(84)
#    compile_model(autoencoder, train, val)


def main_ch():
    train, val, test = get_data()

    param_grid = {
        'learning_rate': [0.01, 0.001, 0.0001],
        'epoch': [8, 16, 25, 32]
    }
    cnn = Cnn(84)
    best_score, best_params, best_model, best_history = cnn.grid_search(param_grid, train, val, test)
    cnn.display_history(best_history)

    all_predictions, all_actual_labels = cnn.predict(test)
    model = cnn.get_model()
    sensitivity_specificity(all_predictions, all_actual_labels)
    loss_accuracy(model, test)

def main_cnn():
    train, val, test = get_data()

    #CNN : 
    cnn = Cnn(84)
    cnn.compile_model(train, val)

    all_predictions, all_actual_labels = cnn.predict(test)

    model = cnn.get_model()

    #f1_score(all_actual_labels, all_predictions)
    sensitivity_specificity(all_predictions, all_actual_labels)
    loss_accuracy(model, test)

    return 0



def main_svc():
    train, val, test = get_data()
    X_train_flat, y_train, X_test_flat, y_test = format_data_for_svc(train, test)
   
    svc = Svc(84)
    param_grid = {'C': np.array([0.1, 1, 10, 100, 1000]), 'gamma': np.array([0.0001, 0.001, 0.01, 0.1, 10])}
    svc.compile_model(X_train_flat, y_train, param_grid)
    svc.display_grid_search_info()
    svc.display_history_v2()

    y_test_pred = svc.predict(X_test_flat)

    accuracy(y_test, y_test_pred)



def main():
    #get data 
    train, val, test = get_data()

    #SVC
    X_train_flat, y_train, X_test_flat, y_test = format_data_for_svc(train, test)
    svc = Svc(84)
    param_grid = {'C': np.array([0.1, 1, 10, 100, 1000]), 'gamma': np.array([0.0001, 0.001, 0.01, 0.1, 10])}
    svc.compile_model(X_train_flat, y_train, param_grid)
    svc.display_grid_search_info()
    svc.display_history_v2()

    y_test_pred = svc.predict(X_test_flat)

    accuracy(y_test, y_test_pred)

    #CNN
    cnn = Cnn(84)
    cnn.compile_model(train, val)

    all_predictions, all_actual_labels = cnn.predict(test)

    model = cnn.get_model()

    #f1_score(all_actual_labels, all_predictions)
    sensitivity_specificity(all_predictions, all_actual_labels)
    loss_accuracy(model, test)