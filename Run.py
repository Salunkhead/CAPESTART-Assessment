import logging
import os
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.model_selection import cross_validate
# from sentence_transformers import SentenceTransformer
# sbert_model = SentenceTransformer('bert-base-nli-mean-tokens')
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
logging.getLogger('tensorflow').disabled = True
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


df = pd.read_csv('Dataset/articles.csv', header=None, encoding='latin-1')  # Reading the dataset
print("df", df)
df1 = df.fillna(0)
arr = np.array(df1)
headers = arr[0]
data = arr[1:]


print("Pre-processing...........")
def Process(dta):
    lab = dta[:,6]
    uni = np.unique(lab)
    label = []
    for i in range(len(lab)):
        for j in range(len(uni)):
            if (lab[i]==uni[j]):
                label.append(j)
    # np.savetxt("Processed/Label.csv", label, delimiter=',', fmt='%s')

    content = []
    data = dta[:,5]
    for i in range(len(data)):
        sence = data[i].replace("<p>", "")
        sentence = sence.replace("</p>", "")
        content.append(sentence)
    return content, label

Prcsd_data, label = Process(data)  # Pre-Processing the dataset



print("Vectorizing")
def Vectorize(sentences):
    # sentence_embeddings = sbert_model.encode(sentences)
    # np.save('Processed/vect_data', sentence_embeddings)
    vect_data = np.load('Processed/vect_data.npy', allow_pickle=True)
    return vect_data
Vec_data = Vectorize(Prcsd_data)  # Vectorization of the dataset


print("Classification......")
def classify(data, label):
    global cr, pm
    # if run:
    X_train, X_test, y_train, y_test = train_test_split(data, label, test_size=0.2, random_state=42)
    mdl = SVC(kernel="rbf",
              C=3)
    mdl.fit(X_train, y_train)
    # joblib.dump(mdl, 'svm_model.pkl')
    pred = mdl.predict(X_test)

    scores = cross_validate(mdl, data, label, cv=4,
                            scoring=['accuracy', 'precision_macro', 'recall_macro', 'f1_macro'])
    cr_val = np.array(
        [scores['test_accuracy'], scores['test_precision_macro'], scores['test_recall_macro'],
         scores['test_f1_macro']])

    cr_df = pd.DataFrame(cr_val, columns=['K-Fold1', 'K-Fold2', 'K-Fold3', 'K-Fold4'], index=['Accuracy', 'Precision', 'Recall', 'F1'])
    print('Cross validation Results')
    print(cr_df.to_markdown())

    pm = np.array([accuracy_score(y_test, pred), precision_score(y_test, pred, average='weighted'),
                   recall_score(y_test, pred, average='weighted'),
                   f1_score(y_test, pred, average='weighted')])
 

classify(Vec_data, label)
