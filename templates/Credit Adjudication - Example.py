# Databricks notebook source
# *********
# DEMO ONLY
# *********
 
_ = sql("CREATE CATALOG IF NOT EXISTS fsgtm")
_ = sql("DROP DATABASE IF EXISTS fsgtm.mrm CASCADE")
_ = sql("CREATE DATABASE fsgtm.mrm")

import pandas as pd

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

# COMMAND ----------

# MAGIC %md
# MAGIC #### External Data Sources
# MAGIC
# MAGIC The external data contains personal information on clients with saving and/or checking accounts. Overall, 1,000 observations are included in the dataset. The following describes the different variables along with the features of the dataset.
# MAGIC
# MAGIC <br>
# MAGIC
# MAGIC + Age (numeric)
# MAGIC + Sex (text: male, female)
# MAGIC + Job (numeric: 0 - unskilled and non-resident, 1 - unskilled and resident, 2 - skilled, 3 - highly skilled)
# MAGIC + Housing (text: own, rent, or free)
# MAGIC + Saving accounts (text - little, moderate, quite rich, rich)
# MAGIC + Checking account (numeric, in DM - Deutsch Mark)
# MAGIC + Credit amount (numeric, in DM)
# MAGIC + Duration (numeric, in month)
# MAGIC + Purpose (text: car, furniture/equipment, radio/TV, domestic appliances, repairs, education, business, vacation/others)

# COMMAND ----------

df_credit = spark.read.table('fsgtm.mrm.german_credit_data_risk').toPandas()
df_credit.loc[df_credit['RISK'] =='good', 'RISK_EN'] = 0 
df_credit.loc[df_credit['RISK'] =='bad', 'RISK_EN'] = 1 
displayHTML(df_credit.head().to_html())

# COMMAND ----------

# MAGIC %md
# MAGIC ### Target Variable Definition
# MAGIC
# MAGIC The model is designed to predict the likelihood of a loan defaulting. The target variable (good/bad) is defined using the information in the extracted dataset. The target variable defines the loans status as ‘good’ or ‘bad’. A ‘good’ status means a good credit performance, i.e., the client did not default during the observation period, whereas a ‘bad’ status means a default occurred during the observation period.
# MAGIC In the modeling code, ‘good’ is identified as ‘0’, and ‘bad’ is identified as ‘1’. The following figures depict the target variable distribution (percentage of good/bad), according to the different variables.
# MAGIC

# COMMAND ----------

import seaborn as sns
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(8,5))

sns.countplot(ax=ax, x=df_credit["RISK"], palette=["steelblue", "coral"], alpha=0.7)

plt.xlabel('Risk variable')
plt.ylabel('Count')
plt.title('Target variable distribution')
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC #### Eligible Population
# MAGIC
# MAGIC The following table provides descriptive statistics on the eligible population for the model development, which includes 1,000 observations, in total. Descriptive statistics apply to the overall population, without any data treatment such as exclusion or sampling. ‘NaN’ mostly appears when trying to compute statistics on categorical variables; hence, they may be ignored.

# COMMAND ----------

displayHTML(df_credit.describe(include='all').to_html())

# COMMAND ----------

# MAGIC %md
# MAGIC #### Good-Bad Observations
# MAGIC
# MAGIC The following provides statistics on the ‘good’ and ‘bad’ observations. Overall, 700 ‘good’ and 300 ‘bad’ observations are found in the dataset. Histograms of ‘good’ and ‘bad’ observations are plotted below. 

# COMMAND ----------

# using matplotlib visualisations
plt.figure(figsize = (12, 6))

# plot distribution for repaid loan with kernel density estimate
sns.kdeplot(df_credit.loc[df_credit['RISK'] == 'good', 'AGE'], label = 'good credit', color='steelblue')
plt.hist(df_credit.loc[df_credit['RISK'] == 'good', 'AGE'], bins=25, alpha=0.25, color='steelblue', density=True)

# plot distribution for default credit with kernel density estimate
sns.kdeplot(df_credit.loc[df_credit['RISK'] == 'bad', 'AGE'], label = 'bad credit', color='coral')
plt.hist(df_credit.loc[df_credit['RISK'] == 'bad', 'AGE'], bins=25, alpha=0.25, color='coral', density=True)

plt.legend()
plt.xlabel('Customer Age')
plt.ylabel('Density')
plt.title('Age distribution')

# COMMAND ----------

# MAGIC %md
# MAGIC ### Sampling Methodology
# MAGIC
# MAGIC Two different datasets, training and validation, were created for the modeling purpose. More specifically, a stratified random sampling methodology was used to sample the original dataset: About 80% was used to train the model, and the remaining 20% was considered for the model performance assessment. The tables below present descriptive statistics on the datasets.

# COMMAND ----------

from sklearn.model_selection import train_test_split

def sampling(df, target, sample_rate=None, sample_seeds=None, sample_tech=None):
    if sample_rate:
        sample_rate = int(sample_rate) / 100
    if sample_seeds:
        sample_seeds = int(sample_seeds)
    if sample_tech:
        predictor = list(set(df.columns) - set(target))
        if sample_tech == 'Random':
            df_sample = df.sample(random_state=sample_seeds, frac=sample_rate)
        elif sample_tech == 'Stratified':
            df_train, df_sample, df_train_target, df_sample_target = train_test_split(df[predictor], df[target],
                                                                                      stratify=df[target],
                                                                                      test_size=sample_rate)
            df_sample[target] = df_sample_target
        else:
            df_sample = df
    else:
        df_sample = df
    return df_sample

# COMMAND ----------

df_sample = sampling(
  df_credit,
  'RISK_EN',
  sample_rate='80',
  sample_seeds='33987',
  sample_tech='Stratified'
)

displayHTML(df_sample.describe().to_html())

# COMMAND ----------

# MAGIC %md
# MAGIC #### Variable Reduction
# MAGIC
# MAGIC Preliminary analyses were conducted to support the variable selection process. More precisely, the variables were plotted according to different bin categories to assess their probability density. Moreover, weights of evidence (WOEs) which measure the relative risk of each bin within each variable were also calculated and evaluated as part of the variable selection process. The results of the probability density and WOEs for each variable are showed below. 

# COMMAND ----------

import numpy as np

def describe_data_g_targ(dat_df, target_var, logbase=np.e):
    num=dat_df.shape[0]
    n_targ = sum(dat_df[target_var]==1)
    n_ctrl = sum(dat_df[target_var]==0)
    assert n_ctrl+ n_targ == num
    base_rate = n_targ/num
    base_odds = n_targ/n_ctrl
    lbm = 1/np.log(logbase)
    base_log_odds = np.log(base_odds)*lbm
    nll_null = -(dat_df[target_var] * np.log(base_rate)*lbm + (1-dat_df[target_var])*np.log(1-base_rate)*lbm).sum()
    logloss_null = nll_null/num
    
    print("********************************")
    print("Number of records:", num)
    print("Target count:", n_targ)
    print("Target rate:", base_rate)
    print("Target odds:", base_odds)
    print("Target log odds:", base_log_odds)
    print("Null model negative log-likelihood:", nll_null)
    print("Null model LogLoss:", logloss_null)
    print("********************************")

    return {'num':num, 'n_targ':n_targ, 'base_rate':base_rate, 'base_odds':base_odds, 
            'base_log_odds':base_log_odds, 'nll_null':nll_null, 'logloss_null':logloss_null}

# COMMAND ----------

describe_data_records = describe_data_g_targ(df_credit,'RISK_EN')
describe_data_df = pd.DataFrame.from_dict(describe_data_records, orient='index', columns=['value'])
displayHTML(describe_data_df.to_html())

# COMMAND ----------

def woe_discrete(df, cat_variabe_name, y_df):
    df = pd.concat([df[cat_variabe_name], y_df], axis = 1)
    df = pd.concat([df.groupby(df.columns.values[0], as_index = False)[df.columns.values[1]].count(),
                    df.groupby(df.columns.values[0], as_index = False)[df.columns.values[1]].mean()], axis =1)
    df = df.iloc[:, [0, 1, 3]]
    df.columns = [df.columns.values[0], 'n_obs', 'prop_bad']
    df['prop_n_obs'] = df['n_obs'] / df['n_obs'].sum()
    df['n_bad'] = df['prop_bad'] * df['n_obs']
    df['n_good'] = (1 - df['prop_bad']) * df['n_obs']
    df['prop_n_good'] = df['n_good'] / df['n_good'].sum()
    df['prop_n_bad'] = df['n_bad'] / df['n_bad'].sum()
    df['WoE'] = np.log(df['prop_n_bad'] / df['prop_n_good'])
    df = df.sort_values(['WoE'])
    df = df.reset_index(drop = True)
    df['IV'] = (df['prop_n_bad'] - df['prop_n_good']) * df['WoE']
    df['Atr_IV'] = df['IV'].sum()
    return df

# COMMAND ----------

 for i in df_credit.columns:
  if i not in ['RISK', 'RISK_EN']:
    display(woe_discrete(df_credit[[i]],i,df_credit['RISK_EN']))
    # univariate_sc_plot(df_credit,i,'Risk_en', n_cuts=3)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Final Variable Reduction
# MAGIC
# MAGIC For the final variable reduction, intervals were created for some continuous variables such as the age, whereas dummies were created for categorical variables such the sex, housing, etc. Results of the analyses are presented below.

# COMMAND ----------

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.model_selection import train_test_split

interval = (18, 25, 35, 60, 120)

cats = ['Student', 'Young', 'Adult', 'Senior']
df_credit["AGE_CAT"] = pd.cut(df_credit['AGE'], interval, labels=cats)

df_credit['SAVING_ACCOUNT'] = df_credit['SAVING_ACCOUNT'].fillna('no_inf')
df_credit['CHECKING_ACCOUNT'] = df_credit['CHECKING_ACCOUNT'].fillna('no_inf')

df_credit = df_credit.merge(pd.get_dummies(df_credit['PURPOSE'], drop_first=True, prefix='PURPOSE'), left_index=True, right_index=True)
df_credit = df_credit.merge(pd.get_dummies(df_credit['SEX'], drop_first=True, prefix='SEX'), left_index=True, right_index=True)
df_credit = df_credit.merge(pd.get_dummies(df_credit['HOUSING'], drop_first=True, prefix='HOUSING'), left_index=True, right_index=True)
df_credit = df_credit.merge(pd.get_dummies(df_credit['SAVING_ACCOUNT'], drop_first=True, prefix='SAVING'), left_index=True, right_index=True)
df_credit = df_credit.merge(pd.get_dummies(df_credit["CHECKING_ACCOUNT"], drop_first=True, prefix='CHECK'), left_index=True, right_index=True)
df_credit = df_credit.merge(pd.get_dummies(df_credit["AGE_CAT"], drop_first=True, prefix='AGE_CAT'), left_index=True, right_index=True)

# excluding the missing columns
del df_credit["SAVING_ACCOUNT"]
del df_credit["CHECKING_ACCOUNT"]
del df_credit["PURPOSE"]
del df_credit["SEX"]
del df_credit["HOUSING"]
del df_credit["AGE_CAT"]
del df_credit["RISK"]

feature_red = SelectKBest(f_classif, k=15).fit(df_credit[[i for i in df_credit.columns if i not in ['RISK','RISK_EN']]],df_credit['RISK_EN'])
X_new = feature_red.transform(df_credit[[i for i in df_credit.columns if i not in ['RISK','RISK_EN']]])

# COMMAND ----------

cols_idxs = feature_red.get_support(indices=True)
train_data_prepared = df_credit.iloc[:,cols_idxs]
train_data_prepared.describe()

X = train_data_prepared[[i for i in train_data_prepared.columns if i not in ['RISK','RISK_EN']]]
y = df_credit['RISK_EN']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state=42)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Model Selection
# MAGIC
# MAGIC For the model selection and estimation, a 10 fold cross-validation procedure is used to compare and select among different alternative models. The following models were trained using hyperopt for hyper parameter tuning.
# MAGIC
# MAGIC <br>
# MAGIC
# MAGIC + K Neighbors
# MAGIC + XGBoost
# MAGIC
# MAGIC Confusion matrices (or error matrices) will be produced for model comparison purposes. Indeed, these matrices easily allow the visualization of the performance of the different models, in terms of actual vs. predicted classes. 

# COMMAND ----------

# we broadcast our dataset to be used by our executors
X_train_broadcast = sc.broadcast(X_train)
X_test_broadcast = sc.broadcast(X_test)
y_train_broadcast = sc.broadcast(y_train)
y_test_broadcast = sc.broadcast(y_test)

# COMMAND ----------

from hyperopt import hp, fmin, tpe, SparkTrials, STATUS_OK, space_eval
from sklearn.pipeline import Pipeline
from sklearn.metrics import average_precision_score
from sklearn.model_selection import cross_val_predict, cross_val_score, KFold

from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier

# we delegate hyperparameter tuning and model selection to hyperopt, distributing that process
# across multiple spark executors

space = hp.choice('classifiers', [
    {
    'model':KNeighborsClassifier(),
    'params':{
        'model__n_neighbors': hp.choice('knc.n_neighbors', range(2,10)),
        'model__algorithm': hp.choice('knc.algorithm', ['auto', 'ball_tree', 'kd_tree']),
        'model__metric': hp.choice('knc.metric', ['chebyshev', 'minkowski'])
    }
    },
    {
    'model': XGBClassifier(eval_metric='logloss', verbosity=0),
    'params': {
        'model__max_depth' : hp.choice('xgb.max_depth', range(5, 30, 1)),
        'model__learning_rate' : hp.quniform('xgb.learning_rate', 0.01, 0.5, 0.1),
        'model__n_estimators' : hp.choice('xgb.n_estimators', range(5, 50, 1)),
        'model__reg_lambda' : hp.uniform ('xgb.reg_lambda', 0, 1),
        'model__reg_alpha' : hp.uniform ('xgb.reg_alpha', 0, 1)
    }
    }
])

# COMMAND ----------

# We define our objective function whose loss we have to minimize
def objective(args):
    # Initialize our model pipeline
    pipeline = Pipeline(steps=[('model', args['model'])])
    pipeline.set_params(**args['params'])
    score = cross_val_score(pipeline, X_train_broadcast.value, y_train_broadcast.value, cv=10, n_jobs=-1, error_score=0.99)
    return {'loss': 1 - np.median(score), 'status': STATUS_OK}

# COMMAND ----------

import mlflow
from mlflow.models import infer_signature
import warnings
warnings.filterwarnings("ignore")

max_evals=50

with mlflow.start_run(run_name='credit adjudication select', nested=True):
  
  # Hyperopts SparkTrials() records all the model and run artifacts.
  trials = SparkTrials(parallelism=20)

  # Fmin will call the objective function with selective param set. 
  # The choice of algorithm will narrow the searchspace.
  best_classifier = fmin(
    objective, 
    space, 
    algo=tpe.suggest,
    max_evals=max_evals, 
    trials=trials
  )

  best_params = space_eval(space, best_classifier)

# COMMAND ----------

best_model = best_params['model']
displayHTML("<p>Best model is [{}]</p>".format(type(best_model).__name__))

# COMMAND ----------

description = """
A 10 fold cross-validation procedure was used to select the best model and hyperparameters across multiple techniques.
Our model selection included XGBoost and K nearest neighbors.
This run was evaluated as our best run that maximizes `cross_val_score`.
"""

with mlflow.start_run(run_name='credit adjudication', description=description) as run:

  # retrieve mlflow run Id
  run_id = run.info.run_id

  # Log model name
  mlflow.set_tag('Model name', type(best_model).__name__)
  mlflow.set_tag('Model complexity', 'MEDIUM')
  mlflow.set_tag('Model explainability', 'MEDIUM')

  # Log model parameters
  mlflow.log_params(best_model.get_params())

  # Train and log our final model
  signature = infer_signature(X_test, y_test)
  model = best_model.fit(X_train, y_train)
  mlflow.sklearn.log_model(model, 'model', signature=signature)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Accuracy Ratio Test
# MAGIC
# MAGIC To better assess the models’ performance, different accuracy tests including, the accuracy ratio, the precision test, the recall test and the F1 test were performed. Results of these tests are showed in the following tables.

# COMMAND ----------

seed=42
kfold = KFold(n_splits=10, random_state=seed, shuffle=True)

def get_accuracy_ratio(model):
    cv_accuracy = cross_val_score(model, X_train, y_train, cv=kfold, scoring='accuracy').mean()
    cv_precision = cross_val_score(model, X_train, y_train, cv=kfold, scoring='precision').mean()
    cv_recall = cross_val_score(model, X_train, y_train, cv=kfold, scoring='recall').mean()
    cv_f1 = cross_val_score(model, X_train, y_train, cv=kfold, scoring='f1').mean()
    return {
      'cv_accuracy': cv_accuracy,
      'cv_precision': cv_precision,
      'cv_recall': cv_recall,
      'cv_f1': cv_f1
    }

# COMMAND ----------

client = mlflow.tracking.MlflowClient()
metrics = get_accuracy_ratio(model)
for metric in metrics.keys():
  client.log_metric(run_id, metric, metrics[metric])

metrics_df = pd.DataFrame.from_dict(metrics, orient='index', columns=['value'])
displayHTML(metrics_df.to_html())

# COMMAND ----------

# MAGIC %md
# MAGIC #### Kolmogorov-Smirnov Test
# MAGIC
# MAGIC In addition to the aforementioned performance tests, the KS test was also performed, and results are the following.

# COMMAND ----------

from scipy.stats import kstest
from scipy.stats import ks_2samp

def get_kolmogorov():
  data1 = X_train.iloc[:,0]
  data2 = X_train.iloc[:,1]
  test_results = ks_2samp(data1,data2)
  return {
      'ks_statistic': test_results.statistic,
      'ks_pvalue': test_results.pvalue,
    }

# COMMAND ----------

metrics = get_kolmogorov()
for metric in metrics.keys():
  client.log_metric(run_id, metric, metrics[metric])

metrics_df = pd.DataFrame.from_dict(metrics, orient='index', columns=['value'])
displayHTML(metrics_df.to_html())

# COMMAND ----------

# MAGIC %md
# MAGIC #### Confusion matrix

# COMMAND ----------

from sklearn.metrics import confusion_matrix

def get_confusion_matrix(model):
    y_pred = cross_val_predict(model, X_train, y_train, cv=kfold)
    group_names = ['True Neg', 'False Pos', 'False Neg', 'True Pos']
    cf_matrix = confusion_matrix(y_train, y_pred)
    group_counts = ['{0:0.0f}'.format(value) for value in cf_matrix.flatten()]
    group_percentages = ['{0:.2%}'.format(value) for value in cf_matrix.flatten()/np.sum(cf_matrix)]
    labels = [f'{v1}\n{v2}\n{v3}' for v1, v2, v3 in zip(group_names,group_counts,group_percentages)]
    labels = np.asarray(labels).reshape(2,2)
    plt.figure(figsize=(8, 5))
    sns.heatmap(cf_matrix, annot=labels, fmt='', cmap='Blues')
    plt.title('Confusion matrix')
    plt.tight_layout()
    plt.savefig('matrix.png')
    plt.show()

# COMMAND ----------

get_confusion_matrix(model)
client.log_artifact(run_id, 'matrix.png', 'images')

# COMMAND ----------

# MAGIC %md
# MAGIC ### Sensitivity Analysis
# MAGIC
# MAGIC A sensitivity analysis was conducted to identify the key variables that mostly impact the model results. For instance, a 5% increase in the following variables, age, credit amount, duration, purpose (if domestic appliances and furniture/equipment) was performed, and the impact reasonableness was assessed. Sensitivity analyses results are showed below.

# COMMAND ----------

from sklearn.datasets import make_regression
import pandas as pd
from xgboost import XGBRegressor
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.pyplot as plt

class Simulate:
    def __init__(self, obs, var):
        self.obs = obs
        self.var = var

    def simulate_increase(self, model, percentage):
        baseline = model.predict(self.obs)
        plus = {}
        for ivar in self.var:
            X_plus = self.obs.copy()
            X_plus[ivar] = X_plus[ivar] + X_plus[ivar] * (percentage / 100)
            plus[ivar] = model.predict(X_plus)
        b = pd.DataFrame(
            plus, index=['simulated'
                         ]).T.reset_index().rename(columns={'index': 'test'})
        b['baseline'] = baseline[0]
        return b

# COMMAND ----------

def sensitivity(d, **kwargs):

    fig, ax = plt.subplots(figsize = (10, 8))
    sns.barplot(x='test', y='simulated', data=d, palette='pastel', ax=ax)
    ax.axhline(d['baseline'].values[0], color='grey', linestyle='--', linewidth=2)
    ax.plot([0, 0], [-100, -100], color='grey', linestyle='--', linewidth=2, label='baseline')

    maxi = int(d['simulated'].max() + d['simulated'].max() * 0.1)
    mini = int(d['simulated'].min() - d['simulated'].min() * 0.1)
    ax.set_ylim([mini, maxi])

    ax.set_xlabel('Simulated variables')
    ax.set_ylabel('Target value')
    ax.set_title(kwargs.get('title'))
    ax.legend()

    ax.grid(axis='y', linewidth=.3)
    sns.despine(offset=10, trim=True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('sensitivity.png')
    plt.show()

# COMMAND ----------

from xgboost import XGBRegressor

VAR_OPTIMIZE = [i for i in X_train.columns[:5]]
PERC = 5
ROW = X_train.iloc[[29]]

mlflow.autolog(disable=True)
model = XGBRegressor()
model.fit(X_train, y_train)
S = Simulate(obs=ROW, var=VAR_OPTIMIZE)
d = S.simulate_increase(model=model, percentage=PERC)
sensitivity(d, title=f'Impact of a {PERC}% increase in target value')
client.log_artifact(run_id, 'sensitivity.png', 'images')

# COMMAND ----------

# MAGIC %md
# MAGIC ### Register model
# MAGIC Finally, we will log all evidence required to trigger an independant review of our modeling approach. We show how to do so programmatically, though this process could be done manually from the MLFlow UI.

# COMMAND ----------

model_uri = "runs:/{}/model".format(run_id)
model_name = "credit_adjudication"
model_version_description = """{} was selected as the best approach over {} different combinations of models and hyperparameters.
Different techniques included K Nearest neighbors and XGBoost.""".format(type(best_model).__name__, max_evals)

# Register model
result = mlflow.register_model(model_uri, model_name)
model_version = result.version

# Update model description
client.update_model_version(name=model_name, version=model_version, description=model_version_description,)

# Update model tags
client.set_model_version_tag(name=model_name, version=model_version, key='Model name', value=type(best_model).__name__)
client.set_model_version_tag(name=model_name, version=model_version, key='Model selection', value='HYPEROPT')
client.set_model_version_tag(name=model_name, version=model_version, key='Model complexity', value='MEDIUM')
client.set_model_version_tag(name=model_name, version=model_version, key='Model explainability', value='MEDIUM')

# COMMAND ----------

# *********
# DEMO ONLY
# *********
 
model_registered_description="""Co-developped with EY, This model is a simple example of how organisations could standardize their approach to AI by defining a series of steps that any data science team ought to address prior to a model validation. Although not exhaustive, this shows that most of the questions required by IVU process for a given use case (Credit adjudication) could be addressed upfront to reduce the friction between regulatory imposed silos, increase model validation success rate and drammatically reduce time from exploration to productionization of AI use cases."""

# Update parent model description (done for the purpose of demo, should have been set upfront)
client.update_registered_model(name=model_name, description=model_registered_description)

# Update parent model tags
client.set_registered_model_tag(name=model_name, key='Model materiality', value='HIGH')
client.set_registered_model_tag(name=model_name, key='Model review', value='REQUESTED')

# COMMAND ----------

# *********
# DEMO ONLY
# *********

# Transition previous iterations to archive (for the purpose of demo)
for model in client.search_model_versions("name='{}'".format(model_name)):
  if model.current_stage == 'Production':
    client.transition_model_version_stage(name=model_name, version=int(model.version), stage="Archived")

# COMMAND ----------

# Transition model to next stage
client.transition_model_version_stage(name=model_name, version=model_version, stage="Production")
