# --------------------------------------------------------------
# Model Evaluation step of the pipeline run
# --------------------------------------------------------------

# +
# Import required classes from Azureml
from azureml.core import Run
import argparse
from azureml.core import Workspace
from azureml.core.model import Model

import os
import pandas as pd
#import matplotlib.pyplot as plt
# #%matplotlib inline

from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_auc_score
# -


# Create a plot
import matplotlib.pyplot as plt
# %matplotlib inline

# Get the context of the experiment run
new_run = Run.get_context()


# Access the Workspace
ws = new_run.experiment.workspace
print("Workspace accessed...")

# Get parameters
parser = argparse.ArgumentParser()
parser.add_argument("--testdata", type=str)
args = parser.parse_args()
print("arguments accessed and printing...")
print(args.testdata)

path = os.path.join(args.testdata, 'x_test.csv')
X_test = pd.read_csv(path)

path = os.path.join(args.testdata, 'y_test.csv')
Y_test = pd.read_csv(path)

path = os.path.join(args.testdata, 'y_predict.csv')
Y_predict = pd.read_csv(path)


import joblib
obj_file = os.path.join(args.testdata, 'rfcModel.pkl')
rfc = joblib.load(obj_file)
score = rfc.score(X_test, Y_test)

fpr, tpr, thresholds = roc_curve(Y_test, Y_predict)
roc_auc = auc(fpr, tpr)
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'y', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()

precision_score = precision_score(Y_test, Y_predict)
recall_score = recall_score(Y_test, Y_predict)
f1_score = f1_score(Y_test, Y_predict)
roc_auc_score = roc_auc_score(Y_test, Y_predict)

#Registering the  model in azure ml
model = Model.register(workspace=ws, model_path=obj_file, model_name="RandomForestModel")

# Evaluate the RFC model
cm = confusion_matrix(Y_test, Y_predict)
#cl = classification_report(Y_test, Y_predict)

#confusion matrix
cm_dict = {"schema_type": "confusion_matrix",
           "schema_version": "v1",
           "data": {"class_labels": ["N", "Y"],
                    "matrix": cm.tolist()}
           }

#logging the metrics
new_run.log_confusion_matrix("ConfusionMatrix", cm_dict)
new_run.log("precision_score", precision_score)
new_run.log("recall_score", recall_score)
new_run.log("recall_score", f1_score)
new_run.log("Score", score)
new_run.get_metrics()
# Log the plot to the run.  To log an arbitrary image, use the form run.log_image(name, path='./image_path.png')
new_run.log_image(name='Receiver Operating Characteristic', plot=plt)


# Complete the run
new_run.complete()


