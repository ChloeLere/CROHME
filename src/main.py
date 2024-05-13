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

def wer(reference, hypothesis):
    ref_words = reference.split()
    hyp_words = hypothesis.split()
    
    # Calculate Levenshtein distance
    distance = lev.distance(reference, hypothesis)
    
    # Calculate WER
    wer = float(distance) / len(ref_words)
    
    return wer

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
    
    # Print the predicted classes and actual labels
    #print("Predicted classes:", all_predictions)
    #print("Actual classes:", all_actual_labels)
    all_predictions = np.array(all_predictions)
    all_actual_labels = np.array(all_actual_labels)
    
    # Number of classes
    num_classes = 85
    f1 = f1_score(all_actual_labels, all_predictions, average='macro')
    print(f"Macro F1 Score: {f1}")

    
    # Calculate sensitivity and specificity for each class
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

    #wer_total = 0
    #num_samples = len(test)
    #for images, labels in test:
    #    predictions = model.predict(images)
    #    predicted_text = decode_predictions(predictions)  # Replace this with your function to decode predictions to text
    #    reference_text = labels.numpy()  # Assuming labels are the ground truth text
    #    
    #    # Calculate WER for each sample
    #    for i in range(len(reference_text)):
    #        reference = reference_text[i].decode('utf-8')  # Convert bytes to string
    #        hypothesis = predicted_text[i]  # Replace this with actual predicted text
    #        wer_total += wer(reference, hypothesis)
    #
    ## Calculate average WER
    #avg_wer = wer_total / num_samples
    #print(f"Average WER: {avg_wer}")


    loss, accuracy = model.evaluate(test)
    print(f"Test Loss: {loss}, Test Accuracy: {accuracy}")



    return 0
    #class_labels = []
    #first_five_batches = []
    #for images, labels in test:
    ## Append the class labels to the list
    #    class_labels.extend(labels.numpy())
    #    first_five_batches.append((images, labels))
    #    if len(class_labels) >= 5:
    #        break
    #print(first_five_batches[0][1])



def main_svn():
    train, val, test = get_data()
    X_train_flat, y_train, X_test_flat, y_test = format_data_for_svn(train, test)
    # Define and train SVM classifier
    svc = Svc(84)
    clf = svc.build_model()
    clf = svc.compile_model(clf, X_train_flat, y_train)

    y_test_pred = clf.predict(X_test_flat)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_test_pred)
    print("Test Accuracy:", accuracy)

