# Databricks notebook source
# DBTITLE 1,Install dependencies
# MAGIC %pip install hyperopt xgboost databricks-sdk -q

# COMMAND ----------

# DBTITLE 1,Fix databricks-sdk namespace collision
import sys
import glob

# FIX: Databricks serverless has a namespace package collision.
# '/databricks/python_shell/lib/databricks/' defines the 'databricks' package but does NOT
# include the SDK subpackage. We extend databricks.__path__ so 'databricks.sdk' resolves
# to the site-packages copy installed by pip. This is required for MLflow's internal
# artifact resolution (DatabricksSdkArtifactRepository) to work.
import databricks
_sdk_paths = glob.glob('/databricks/python/lib/python*/site-packages/databricks')
for _p in _sdk_paths:
    if _p not in databricks.__path__:
        databricks.__path__.insert(0, _p)

# Verify
from databricks.sdk import WorkspaceClient  # noqa: F401
print(f"databricks-sdk resolved OK: {databricks.sdk.__file__}")

# COMMAND ----------

import pandas as pd
import numpy as np
from scipy.stats import kstest
from scipy.stats import ks_2samp

# COMMAND ----------

import warnings
warnings.filterwarnings("ignore")

# COMMAND ----------

import mlflow
# Unity Catalog is the default model registry on modern Databricks runtimes

# COMMAND ----------

import seaborn as sns
import matplotlib.pyplot as plt

# COMMAND ----------

from hyperopt import hp, fmin, tpe, Trials, STATUS_OK, space_eval

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

catalog = spark.sql("SELECT current_catalog()").first()[0]
schema = spark.sql("SELECT current_user()").first()[0]
schema = schema.split('@')[0].replace('.', '_').lower()
table_name = 'credit_data'
volume_name = 'landing_zone'

# COMMAND ----------

# Catalog is derived from session default — no need to create it
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {catalog}.{schema}")
spark.sql(f"CREATE VOLUME IF NOT EXISTS {catalog}.{schema}.{volume_name}")

# COMMAND ----------

# DBTITLE 1,Copy data file to volume (portable path)
import shutil
import os

# Dynamically resolve the project data path from the calling notebook's location.
# When invoked via %run from the main notebook, notebookPath() returns the caller's path.
# This ensures portability — no hardcoded user paths or workspace references.
_nb_path = dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get()
_templates_dir = '/Workspace' + '/'.join(_nb_path.split('/')[:-1])
_data_path = os.path.join(_templates_dir, 'data', 'german_credit_data_risk.csv')
shutil.copy(_data_path, f'/Volumes/{catalog}/{schema}/{volume_name}/german_credit_data_risk.csv')

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
