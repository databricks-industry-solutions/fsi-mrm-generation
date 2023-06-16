# Databricks notebook source
import pandas as pd
import numpy as np
from scipy.stats import kstest
from scipy.stats import ks_2samp

# COMMAND ----------

import warnings
warnings.filterwarnings("ignore")

# COMMAND ----------

import seaborn as sns
import matplotlib.pyplot as plt

# COMMAND ----------

from hyperopt import hp, fmin, tpe, SparkTrials, STATUS_OK, space_eval

from sklearn.datasets import make_regression
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.metrics import average_precision_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_predict, cross_val_score, KFold
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline

from xgboost import XGBClassifier
from xgboost import XGBRegressor

import mlflow
from mlflow.models import infer_signature

# COMMAND ----------

_ = sql("CREATE CATALOG IF NOT EXISTS fsgtm")
_ = sql("DROP DATABASE IF EXISTS fsgtm.mrm CASCADE")
_ = sql("CREATE DATABASE fsgtm.mrm")

# COMMAND ----------

df = pd.read_csv("data/german_credit_data_risk.csv", index_col=0)
df = df.rename({
  'Age' : 'AGE',
  'Sex' : 'SEX',
  'Job' : 'JOB',
  'Housing': 'HOUSING',
  'Saving accounts': 'SAVING_ACCOUNT',
  'Checking account': 'CHECKING_ACCOUNT',
  'Credit amount': 'CREDIT_AMOUNT',
  'Duration': 'DURATION',
  'Purpose': 'PURPOSE',
  'Risk': 'RISK'
}, axis=1)

spark.createDataFrame(df).write.format('delta').saveAsTable('fsgtm.mrm.german_credit_data_risk')
