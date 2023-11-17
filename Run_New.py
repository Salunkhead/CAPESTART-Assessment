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


df = pd.read_csv('Dataset/unknown_articles.csv', header=None, encoding='latin-1')  # Reading the dataset
print("df", df)
df1 = df.fillna(0)
arr = np.array(df1)
headers = arr[0]
data = arr[1:]



from urllib.request import Request, urlopen

req = Request(
    url='http://australianaviation.com.au/2018/10/a-competitive-edge-50-years-of-the-australian-army-aviation-corps/',
    headers={'User-Agent': 'Mozilla/5.0'}
)
webpage = urlopen(req).read()


