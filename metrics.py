import matplotlib.pyplot as plt
import os
import pandas as pd
import seaborn as sn
import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_score, recall_score, f1_score
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import classification_report
def plot_and_save_loss_curves(results,save_folder=None, save_name="loss_curves"):

    loss_1 = results["train_loss"]
    test_loss_1 = results["test_loss"]

    accuracy_1 = results["train_acc"]
    test_accuracy_1 = results["test_acc"]

    epochs = range(len(results["train_loss"]))

    plt.figure(figsize=(15,5))

    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, loss_1, label="train_loss")
    plt.plot(epochs, test_loss_1, label="valid_loss")
    plt.ylim(0,1.0)
    plt.title("Loss")
    plt.xlabel("Epochs")
    plt.legend()

    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, accuracy_1, label="train_accuracy")
    plt.plot(epochs, test_accuracy_1, label="valid_accuracy")
    plt.ylim(0,1.0)
    plt.title("Accuracy")
    plt.xlabel("Epochs")
    plt.legend()

    if save_folder:
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        save_path = os.path.join(save_folder, save_name + ".png")
        plt.savefig(save_path)
        print(f"Figures saved at: {save_path}")
    else:
        plt.show()
def ops(y_true,y_pred,y_scores,save_folder=None,num_classes=5):
    y_test_bin = label_binarize(y_true, classes=list(range(num_classes)))
    y_scores=np.array(y_scores).reshape(-1, len(y_scores[0]))
    res = []
    for l in range(num_classes):
        prec,recall,f1score,support = precision_recall_fscore_support(np.array(y_true)==l,
                                                    np.array(y_pred)==l,
                                                    pos_label=True,average=None)
        res.append([l,recall[0],recall[1]])
    print(pd.DataFrame(res,columns = ['class','specificity','sensitivity']))

    target_names = ['class '+str(i) for i in range(num_classes)]
    print(classification_report(y_true, y_pred, target_names=target_names))
    
    classes=[i for i in range(num_classes)]
    cf_matrix = confusion_matrix(y_true, y_pred)

    print(cf_matrix)
    df_cm = pd.DataFrame(cf_matrix / np.sum(cf_matrix, axis=1)[:, None], index = [i for i in classes],
                        columns = [i for i in classes])
    plt.figure(figsize = (12,7))
    sn.heatmap(df_cm, cmap="Blues",annot=True)
    
    if save_folder:
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        save_path = os.path.join(save_folder, 'confusion matrix' + ".png")
        plt.savefig(save_path)
        print(f"Figures saved at: {save_path}")
    plt.show()
    # Compute ROC curve and AUC for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(num_classes):  # assuming 5 classes
        fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_scores[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Plot ROC curves for each class
    plt.figure(figsize=(10, 8))

    for i in range(num_classes):  # assuming 5 classes
        plt.plot(fpr[i], tpr[i], label=f'Class {i} (AUC = {roc_auc[i]:.2f}')

    plt.plot([0, 1], [0, 1], 'k--', linewidth=2)  # Diagonal reference line
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR)')
    plt.title('Multiclass Receiver Operating Characteristic (ROC) Curves')
    plt.legend(loc="lower right")
    if save_folder:
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        save_path = os.path.join(save_folder, 'aucroc' + ".png")
        plt.savefig(save_path)
        print(f"Figures saved at: {save_path}")
    plt.show()
