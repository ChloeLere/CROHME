#!/usr/bin/env python3
from data import get_data
from cnn import build_model, compile_model
import numpy as np

def main():
    print ("hello")
    train, val, test = get_data()
    model = build_model(84)
    compile_model(model, train, val)
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