#!/usr/bin/env python3
from data import get_data, format_data_for_svn
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

def main():
    print ("hello")
    train, val, test = get_data()

    #CNN : 
    cnn = Cnn(84)
    model = cnn.build_model(84)
    cnn.compile_model(model, train, val)
    #predictions = model.predict(test)
    #print(np.argmax(predictions, axis=1))
    #for images, labels in test:
    #    print(labels.nympy())

    


    all_predictions = []
    all_actual_labels = []
    
   

    # Iterate through the test dataset
    for images, labels in test:
        # Perform prediction on the current batch
        predictions = model.predict(images)
        
        # Convert predictions to classes (taking the index of the max value along axis 1)
        predicted_classes = np.argmax(predictions, axis=1)
        
        # Store predicted classes and actual labels
        all_predictions.extend(predicted_classes)
        all_actual_labels.extend(labels.numpy())
    
    all_predictions = np.array(all_predictions)
    all_actual_labels = np.array(all_actual_labels)
    
    f1_score(all_actual_labels, all_predictions)
    sensitivity_specificity(all_predictions, all_actual_labels)
    loss_accuracy(model, test)

    return 0



def main_svn():
    train, val, test = get_data()
    X_train_flat, y_train, X_test_flat, y_test = format_data_for_svn(train, test)
    # Define and train SVM classifier
    svc = Svc(84)
    clf = svc.build_model()
    clf = svc.compile_model(clf, X_train_flat, y_train)

    y_test_pred = clf.predict(X_test_flat)
    
    accuracy(y_test, y_test_pred)

