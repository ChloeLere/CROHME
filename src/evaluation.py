from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
import Levenshtein as lev
import numpy as np
from sklearn.metrics import confusion_matrix
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt

def wer(reference, hypothesis):
    ref_words = reference.split()
    distance = lev.distance(reference, hypothesis)
    wer = float(distance) / len(ref_words)
    return wer

def accuracy(y_test, y_test_pred):
    accuracy = accuracy_score(y_test, y_test_pred)
    print("Test Accuracy:", accuracy)
    return accuracy

def f1_score(all_actual_labels, all_predictions, average='macro'):
    f1 = f1_score(all_actual_labels, all_predictions, average='macro')
    print(f"Macro F1 Score: {f1}")
    return f1

def loss_accuracy(model, test):
    loss, accuracy = model.evaluate(test)
    print(f"Test Loss: {loss}, Test Accuracy: {accuracy}")
    return loss, accuracy

def sensitivity_specificity(all_predictions, all_actual_labels, num_classes=84):
    for class_id in range(num_classes):
        TP = np.sum((all_predictions == class_id) & (all_actual_labels == class_id))
        FP = np.sum((all_predictions == class_id) & (all_actual_labels != class_id))
        TN = np.sum((all_predictions != class_id) & (all_actual_labels != class_id))
        FN = np.sum((all_predictions != class_id) & (all_actual_labels == class_id))
        
        sensitivity = TP / (TP + FN) if TP + FN > 0 else 0
        specificity = TN / (TN + FP) if TN + FP > 0 else 0
        
        print(f"Class {class_id}: Sensitivity = {sensitivity}, Specificity = {specificity}")

def display_pred_and_actual(all_predictions, all_actual_labels):
    print("Predicted classes:", all_predictions)
    print("Actual classes:", all_actual_labels)

def create_confusion_matrix(all_actual_labels, all_predictions, filename, class_names_test, annot=False):
    conf_matrix_cnn = confusion_matrix(np.array(all_actual_labels), np.array(all_predictions), labels=range(len(class_names_test)))
    print(conf_matrix_cnn)
    print(conf_matrix_cnn.shape)
    df_cm_cnn = pd.DataFrame(conf_matrix_cnn, index=class_names_test, columns=class_names_test)

    plt.figure(figsize=(12, 9))
    sn.heatmap(df_cm_cnn, annot=annot)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.savefig(filename)
    plt.show()
    plt.close()