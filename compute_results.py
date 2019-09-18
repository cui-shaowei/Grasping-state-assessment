from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import sklearn as sk

# '''
y_true = np.array(np.load('checkpoint_C3D35_1_8_512/best_targets.npy'))
y_pred = np.array(np.load('checkpoint_C3D35_1_8_512/best_pred.npy'))

print("Precision", sk.metrics.precision_score(y_true, y_pred,average='macro'))
print( "Recall", sk.metrics.recall_score(y_true, y_pred,average='macro'))
print( "f1_score", sk.metrics.f1_score(y_true, y_pred,average='macro'))

# print("Precision", sk.metrics.precision_score(y_true, y_pred,average='weighted'))
# print( "Recall", sk.metrics.recall_score(y_true, y_pred,average='weighted'))
# print( "f1_score", sk.metrics.f1_score(y_true, y_pred,average='weighted'))