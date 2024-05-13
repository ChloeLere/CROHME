from sklearn.metrics import accuracy_score 
from sklearn.metrics import f1_score
import Levenshtein as lev
import numpy as np

def wer(reference, hypothesis):
    ref_words = reference.split()
    hyp_words = hypothesis.split()
    
    # Calculate Levenshtein distance
    distance = lev.distance(reference, hypothesis)
    
    # Calculate WER
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
        # Calculate True Positives (TP), False Positives (FP), True Negatives (TN), and False Negatives (FN)
        TP = np.sum((all_predictions == class_id) & (all_actual_labels == class_id))
        FP = np.sum((all_predictions == class_id) & (all_actual_labels != class_id))
        TN = np.sum((all_predictions != class_id) & (all_actual_labels != class_id))
        FN = np.sum((all_predictions != class_id) & (all_actual_labels == class_id))
        
        # Calculate sensitivity and specificity
        sensitivity = TP / (TP + FN) if TP + FN > 0 else 0
        specificity = TN / (TN + FP) if TN + FP > 0 else 0
        
        # Print sensitivity and specificity for the class
        print(f"Class {class_id}: Sensitivity = {sensitivity}, Specificity = {specificity}")

def display_pred_and_actual(all_predictions, all_actual_labels):
    print("Predicted classes:", all_predictions)
    print("Actual classes:", all_actual_labels)