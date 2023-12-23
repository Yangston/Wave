import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.metrics import confusion_matrix, classification_report, f1_score, precision_score, recall_score, precision_recall_curve, roc_curve, auc
from sklearn.preprocessing import label_binarize
import seaborn as sns
import pandas as pd
import tensorflow as tf
from itertools import cycle

# Load the trained model
model_save_path = 'model/signDetection/signDetector.hdf5'
training_history_path = 'model/signDetection/training_history.npz'
loaded_model = tf.keras.models.load_model(model_save_path)
loaded_model.compile(optimizer='adam',  # You can specify your optimizer
                     loss='sparse_categorical_crossentropy',  # You can specify your loss function
                     metrics=['accuracy'])


# Load testing data
X_test = np.loadtxt('model/signDetection/signData.csv', delimiter=',',
                    dtype='float32', usecols=list(range(1, (21 * 2) + 1)))
y_test = np.loadtxt('model/signDetection/signData.csv',
                    delimiter=',', dtype='int32', usecols=(0))

# Inference test
Y_pred = loaded_model.predict(X_test)
y_pred = np.argmax(Y_pred, axis=1)


def print_confusion_matrix(y_true, y_pred, report=True):
    labels = sorted(list(set(y_true)))
    cmx_data = confusion_matrix(y_true, y_pred, labels=labels)

    df_cmx = pd.DataFrame(cmx_data, index=labels, columns=labels)

    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(df_cmx, annot=True, fmt='g', square=False)
    ax.set_ylim(len(set(y_true)), 0)
    plt.show()

    if report:
        print('Classification Report')
        print(classification_report(y_true, y_pred))

    return fig


def plot_f1_confidence_curve(Y_pred, y_true):
    n_classes = len(np.unique(y_true))
    confidence_thresholds = np.linspace(0, 1, 100)
    fig, ax = plt.subplots()

    for i in range(n_classes):
        f1_scores = []
        for threshold in confidence_thresholds:
            y_pred_probs = Y_pred[:, i]
            y_pred_thresholded = np.where(
                y_pred_probs > threshold, 1, 0)
            f1_scores.append(
                f1_score(y_true == i, y_pred_thresholded, zero_division=1)
            )

        plt.plot(confidence_thresholds, f1_scores, label=f'Class {i}')

    plt.xlabel('Confidence Threshold')
    plt.ylabel('F1 Score')
    plt.title('F1-Confidence Curve for Each Class')
    plt.legend()
    plt.show()

    return fig


# Function to plot Precision-Confidence Curve

def plot_precision_confidence_curve(Y_pred, y_true):
    n_classes = len(np.unique(y_true))
    confidence_thresholds = np.linspace(0, 1, 100)
    fig, ax = plt.subplots()

    for i in range(n_classes):
        precision_scores = []
        for threshold in confidence_thresholds:
            y_pred_probs = Y_pred[:, i]
            y_pred_thresholded = np.where(
                y_pred_probs > threshold, 1, 0)
            precision_scores.append(
                precision_score(
                    y_true == i, y_pred_thresholded, zero_division=1)
            )

        plt.plot(confidence_thresholds, precision_scores, label=f'Class {i}')

    plt.xlabel('Confidence Threshold')
    plt.ylabel('Precision')
    plt.title('Precision-Confidence Curve for Each Class')
    plt.legend()
    plt.show()

    return fig


def plot_recall_confidence_curve(Y_pred, y_true):
    n_classes = len(np.unique(y_true))
    confidence_thresholds = np.linspace(0, 1, 100)
    fig, ax = plt.subplots()

    for i in range(n_classes):
        recall_scores = []
        for threshold in confidence_thresholds:
            y_pred_probs = Y_pred[:, i]
            y_pred_thresholded = np.where(
                y_pred_probs > threshold, 1, 0)
            recall_scores.append(
                recall_score(y_true == i, y_pred_thresholded, zero_division=1)
            )

        plt.plot(confidence_thresholds, recall_scores, label=f'Class {i}')

    plt.xlabel('Confidence Threshold')
    plt.ylabel('Recall')
    plt.title('Recall-Confidence Curve for Each Class')
    plt.legend()
    plt.show()

    return fig


def plot_precision_recall_curve(Y_pred, y_true):
    n_classes = len(np.unique(y_true))
    y_test_bin = label_binarize(y_true, classes=np.arange(n_classes))

    precision = dict()
    recall = dict()
    average_precision = dict()

    for i in range(n_classes):
        precision[i], recall[i], _ = precision_recall_curve(
            y_test_bin[:, i], Y_pred[:, i])
        average_precision[i] = auc(recall[i], precision[i])

    # Micro-average precision-recall curve
    precision["micro"], recall["micro"], _ = precision_recall_curve(
        y_test_bin.ravel(), Y_pred.ravel())
    average_precision["micro"] = auc(recall["micro"], precision["micro"])

    # Macro-average precision-recall curve
    all_precision = np.unique(np.concatenate(
        [precision[i] for i in range(n_classes)]))
    mean_recall = np.zeros_like(all_precision)
    for i in range(n_classes):
        mean_recall += np.interp(all_precision, precision[i], recall[i])

    mean_recall /= n_classes

    precision["macro"] = all_precision
    recall["macro"] = mean_recall
    average_precision["macro"] = auc(recall["macro"], precision["macro"])

    fig, ax = plt.subplots()

    # Plot micro-average and macro-average precision-recall curves
    plt.plot(recall["micro"], precision["micro"], color='gold', lw=2,
             label='Micro-average (area = {0:0.2f})'
                   ''.format(average_precision["micro"]))

    plt.plot(recall["macro"], precision["macro"], color='navy', lw=2,
             label='Macro-average (area = {0:0.2f})'
                   ''.format(average_precision["macro"]))

    # Plot individual class precision-recall curves
    colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
    for i, color in zip(range(n_classes), colors):
        plt.plot(recall[i], precision[i], color=color, lw=2,
                 label='Class {0} (area = {1:0.2f})'
                       ''.format(i, average_precision[i]))

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc='best')
    plt.show()

    return fig


'''
from sklearn.metrics import precision_recall_curve, auc

def plot_precision_recall_curve(Y_pred, y_true):
    n_classes = len(np.unique(y_true))
    confidence_thresholds = np.linspace(0, 1, 100)
    fig, ax = plt.subplots()

    for i in range(n_classes):
        precision_curve, recall_curve, _ = precision_recall_curve(
            (y_true == i).astype(int), Y_pred[:, i]
        )
        average_precision = auc(recall_curve, precision_curve)

        plt.plot(recall_curve, precision_curve, label=f'Class {i} (AP = {average_precision:.2f})')

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve for Each Class')
    plt.legend()
    plt.show()

    return fig

'''


def plot_one_vs_all_roc_curve(Y_pred, y_true):
    n_classes = len(np.unique(y_true))
    y_test_bin = label_binarize(y_true, classes=np.arange(n_classes))

    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], Y_pred[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    fig, ax = plt.subplots(figsize=(8, 6))
    colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                 label=f'ROC curve (class {i}, area = {roc_auc[i]:.2f})')

    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('One-vs-All ROC Curve')
    plt.legend(loc='lower right')
    plt.show()

    return fig


def plot_learning_curve(training_history):
    fig, ax = plt.subplots()

    plt.plot(training_history['accuracy'], label='Training Accuracy')
    plt.plot(training_history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Model Training and Validation Accuracy')
    plt.legend()
    plt.show()

    return fig


def plot_loss_plots(training_history):
    fig, ax = plt.subplots()

    plt.plot(training_history['loss'], label='Training Loss')
    plt.plot(training_history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Model Training and Validation Loss')
    plt.legend()
    plt.show()

    return fig


# Performance Analysis
with PdfPages('plots.pdf') as pdf:  # Specify the PDF file name
    # Call the functions to generate plots
    fig = print_confusion_matrix(y_test, y_pred)
    pdf.savefig(fig)
    plt.close(fig)

    fig = plot_f1_confidence_curve(Y_pred, y_test)
    pdf.savefig(fig)
    plt.close(fig)

    fig = plot_precision_confidence_curve(Y_pred, y_test)
    pdf.savefig(fig)
    plt.close(fig)

    fig = plot_recall_confidence_curve(Y_pred, y_test)
    pdf.savefig(fig)
    plt.close(fig)

    fig = plot_precision_recall_curve(Y_pred, y_test)
    pdf.savefig(fig)
    plt.close(fig)

    fig = plot_one_vs_all_roc_curve(Y_pred, y_test)
    pdf.savefig(fig)
    plt.close(fig)

    fig = plot_learning_curve(np.load(training_history_path))
    pdf.savefig(fig)
    plt.close(fig)

    fig = plot_loss_plots(np.load(training_history_path))
    pdf.savefig(fig)
    plt.close(fig)
