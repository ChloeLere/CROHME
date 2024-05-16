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
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


# evaluation method for comparaison :
# - accuracy validation 
# - 

def main_cnn_h():
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
    #param_grid = {'C': np.array([0.1]), 'gamma': np.array([10])}
    svc.compile_model(X_train_flat, y_train, param_grid)
    svc.display_grid_search_info()
    svc.display_history_v2()

    all_actual_labels = []
    y_test_pred = svc.predict(X_test_flat)
    for images, labels in test:
            all_actual_labels.extend(labels.numpy())
        

    accuracy(y_test, y_test_pred)
    conf_matrix = confusion_matrix(np.array(y_test), np.array(y_test_pred))
    df_cm = pd.DataFrame(conf_matrix, range(conf_matrix.shape[0]), range(conf_matrix.shape[0]))
    sn.heatmap(df_cm, annot=True, annot_kws={"size": 16}) # font size
    plt.savefig("confusion_matrix.png")
    plt.close()




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
    conf_matrix = confusion_matrix(np.array(y_test), np.array(y_test_pred))
    df_cm = pd.DataFrame(conf_matrix, range(conf_matrix.shape[0]), range(conf_matrix.shape[0]))
    sn.heatmap(df_cm, annot=True, annot_kws={"size": 16}) # font size
    plt.savefig("confusion_matrix_svc.png")
    plt.close()

    #CNN
    param_grid = {
        'learning_rate': [0.01, 0.001, 0.0001],
        'epoch': [8, 16, 25, 32]
    }
    cnn = Cnn(84)
    best_score, best_params, best_model, best_history = cnn.grid_search(param_grid, train, val, test)
    cnn.display_history(best_history)

    all_predictions, all_actual_labels = cnn.predict(test)
    model = cnn.get_model()
    model.save('cnn_model.keras')
    sensitivity_specificity(all_predictions, all_actual_labels)
    loss_accuracy(model, test)
    conf_matrix_cnn = confusion_matrix(np.array(all_actual_labels), np.array(all_predictions))
    df_cm_cnn = pd.DataFrame(conf_matrix_cnn, range(conf_matrix_cnn.shape[0]), range(conf_matrix_cnn.shape[0]))
    sn.heatmap(df_cm_cnn, annot=False) # font size
    plt.savefig("confusion_matrix_cnn.png")
    plt.close()


def main_test():
    train, val, test = get_data()
    param_grid = {
        'learning_rate': [0.01],
        'epoch': [1]
    }
    cnn = Cnn(84)
    best_score, best_params, best_model, best_history = cnn.grid_search(param_grid, train, val, test)
    cnn.display_history(best_history)

    all_predictions, all_actual_labels = cnn.predict(test)
    model = cnn.get_model()
    sensitivity_specificity(all_predictions, all_actual_labels)
    loss_accuracy(model, test)
    conf_matrix_cnn = confusion_matrix(np.array(all_actual_labels), np.array(all_predictions))
    print(np.array(conf_matrix_cnn))
    df_cm_cnn = pd.DataFrame(conf_matrix_cnn, range(conf_matrix_cnn.shape[0]), range(conf_matrix_cnn.shape[0]))
    sn.heatmap(df_cm_cnn, annot=False) # font size
    plt.savefig("confusion_matrix_cnn.png")
    plt.show()
    plt.close()