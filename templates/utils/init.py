# Databricks notebook source
import pandas as pd
import numpy as np
from scipy.stats import kstest
from scipy.stats import ks_2samp

# COMMAND ----------

import warnings
warnings.filterwarnings("ignore")

# COMMAND ----------

import mlflow
# ensure models will be governed by UC
mlflow.set_registry_uri('databricks-uc')

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

from mlflow.models import infer_signature

# COMMAND ----------

catalog = 'users'
schema = dbutils.notebook.entry_point.getDbutils().notebook().getContext().userName().get()
schema = schema.split('@')[0].replace('.', '_').lower()
table_name = 'credit_data'
volume_name = 'landing_zone'

# COMMAND ----------

_ = sql(f"CREATE CATALOG IF NOT EXISTS {catalog}")
_ = sql(f"CREATE DATABASE IF NOT EXISTS {catalog}.{schema}")
_ = sql(f"CREATE VOLUME IF NOT EXISTS {catalog}.{schema}.{volume_name}")

# COMMAND ----------

import shutil
shutil.copy('data/german_credit_data_risk.csv', f'/Volumes/{catalog}/{schema}/{volume_name}/german_credit_data_risk.csv')

# COMMAND ----------

from pyspark.sql import functions as F

volume_df = (
  spark
    .read
    .format('csv')
    .option('header', 'true')
    .option('inferSchema', 'true')
    .load(f'/Volumes/{catalog}/{schema}/{volume_name}')
    .select(
      F.col('Age').alias('AGE'),
      F.col('Sex').alias('SEX'),
      F.col('Job').alias('JOB'),
      F.col('Housing').alias('HOUSING'),
      F.col('Saving accounts').alias('SAVING_ACCOUNT'),
      F.col('Checking account').alias('CHECKING_ACCOUNT'),
      F.col('Credit amount').alias('CREDIT_AMOUNT'),
      F.col('Duration').alias('DURATION'),
      F.col('Purpose').alias('PURPOSE'),
      F.col('Risk').alias('RISK')
    )
)

volume_df.write.mode('overwrite').format('delta').saveAsTable(f'{catalog}.{schema}.{table_name}')
