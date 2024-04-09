# Databricks notebook source
# MAGIC %md
# MAGIC # Credit Risk Adjudication Model
# MAGIC
# MAGIC This notebook is a simple example of how organisations could standardize their approach to AI by defining a series of steps that any data science team ought to address prior to a model validation. Although not exhaustive, this shows that most of the questions required by IVU process could be addressed upfront (at model development phase) to reduce the friction between regulatory imposed silos, increase model validation success rate and drammatically reduce time from exploration to productionization of AI use cases. 
# MAGIC
# MAGIC <br>
# MAGIC
# MAGIC <a href="https://www.ey.com/en_ca" target="_blank">
# MAGIC   <img src='https://assets.ey.com/content/dam/ey-sites/ey-com/en_gl/generic/logos/20170526-EY-Digital-Brand.svg' width=100> 
# MAGIC </a>

# COMMAND ----------

# MAGIC %md
# MAGIC ## Executive Summary
# MAGIC This notebook demonstrates the use of Machine Learning for credit risk adjudication model. We will be loading a publicly available dataset and evaluate multiple modelling techniques and parameter tuning using hyperopts in order to select the best approach balancing between model explainability and model accuracy.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1 Introduction

# COMMAND ----------

# MAGIC %md
# MAGIC ### 1.1 Model Background and Initiation
# MAGIC The motivations behind this modeling effort is to showcase Lakehouse capabilities combined with EY expertise as it relates to model risk management. The goal is not to build the best model nor to showcase latest state of the art AI capabilities. 

# COMMAND ----------

# MAGIC %md
# MAGIC ### 1.2 Model Purpose
# MAGIC The purpose of this document is to provide a detailed description of the new retail credit adjudication model.

# COMMAND ----------

# MAGIC %md
# MAGIC ### 1.3 Model Portfolio
# MAGIC The MLflow Model Registry component is a centralized model store, set of APIs, and UI, to collaboratively manage the full lifecycle of an MLflow Model. It provides model lineage (which MLflow experiment and run produced the model), model versioning, stage transitions (for example from staging to production), and annotations. Used as a backbone of our model risk management solution accelerator, this becomes the de facto place to register both machine learning and non machine learning models. 

# COMMAND ----------

# MAGIC %md
# MAGIC ### 1.4 Model Risk Rating
# MAGIC Credit Risk Model would give creditors, analysts, and portfolio managers a way of ranking borrowers based on their creditworthiness and default risk. Any issue on the model output would have financial consequences, leading to a relative `HIGH` model materiality. 

# COMMAND ----------

# MAGIC %md
# MAGIC ### 1.5 Model Log of Changes
# MAGIC We captured all different models and previous versions using MLFlow. Model development history is available through the MLFlow registry UI / API.

# COMMAND ----------

# MAGIC %md
# MAGIC ### 1.6 Business-Driven Risk Considerations
# MAGIC *Note: Explain the business risks that are explored and assessed during the model development process, and how they are accounted for in the final model (outputs). Describe and justify any mitigation action (plan) that helps reduce the business-driven risk.*

# COMMAND ----------

# MAGIC %md
# MAGIC ### 1.7 Economic and Market Outlook
# MAGIC *Note: Explain how the current and forward-looking overall economic conditions may impact the business line and subsequently the model outcome.*

# COMMAND ----------

# MAGIC %md
# MAGIC ### 1.8 Model Development Process
# MAGIC *Note: Describe the overall model development process, the different milestones of the process, along with the roles and responsibilities of the stakeholders involved at each of these key steps.*

# COMMAND ----------

# MAGIC %md
# MAGIC ### 1.9 Economic and Market Outlook
# MAGIC *Note: Explain how the current and forward-looking overall economic conditions may impact the business line and subsequently the model outcome.*

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2 Data

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2.1 Borrower Definition
# MAGIC *Note: Describe the borrowers’ categories of the model portfolio/population. For instance, whether the model applies to borrowers with a certain range of exposure, within a geographical area, or with a minimum/maximum of total asset (e.g., when the model also applies to SMEs). It outlines the borrower identification process in the data bases as well.*

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2.2 Data Sources

# COMMAND ----------

# MAGIC %md
# MAGIC #### 2.2.1 Internal Data Sources
# MAGIC *Note: Describe the internal data sources, as well as their appropriateness with the model purpose and model population.*

# COMMAND ----------

# MAGIC %run ./utils/init

# COMMAND ----------

# MAGIC %md
# MAGIC #### 2.2.2 External Data Sources
# MAGIC
# MAGIC The external data contains personal information on clients with saving and/or checking accounts. Overall, 1,000 observations are included in the dataset. The following describes the different variables along with the features of the dataset. We display a few records below as well as table statistics.
# MAGIC
# MAGIC <br>
# MAGIC
# MAGIC + `AGE` (numeric)
# MAGIC + `SEX` (text: male, female)
# MAGIC + `JOB` (numeric: 0 - unskilled and non-resident, 1 - unskilled and resident, 2 - skilled, 3 - highly skilled)
# MAGIC + `HOUSING` (text: own, rent, or free)
# MAGIC + `SAVING_ACCOUNT` (text - little, moderate, quite rich, rich)
# MAGIC + `CHECKING_ACCOUNT` (numeric, in DM - Deutsch Mark)
# MAGIC + `CREDIT_AMOUNT` (numeric, in DM)
# MAGIC + `DURATION` (numeric, in month)
# MAGIC + `PURPOSE` (text: car, furniture/equipment, radio/TV, domestic appliances, repairs, education, business, vacation/others)

# COMMAND ----------

# When you train a model on a table in Unity Catalog, you can track the lineage of the model to the upstream dataset(s) it was trained and evaluated on. For that purpose, we load our data as a mlflow dataset.
mlflow_dataset = mlflow.data.load_delta(table_name=f'{catalog}.{schema}.{table_name}')

# COMMAND ----------

df_credit = mlflow_dataset.df.toPandas()
df_credit.loc[df_credit['RISK'] =='good', 'RISK_EN'] = 0 
df_credit.loc[df_credit['RISK'] =='bad', 'RISK_EN'] = 1 
df_credit.head().drop('RISK_EN', axis=1)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2.3 Data Historical Coverage and Suitability
# MAGIC *Note: Describe the data extraction process, along with the period spanned by the data and the statistics on the extracted observations. The section should not only evidence that the extracted data reflects the business practices and experiences, but is also suitable for the model purpose, modeling methodology and modeling assumptions.*

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2.4 Modeling Timeframes

# COMMAND ----------

# MAGIC %md
# MAGIC #### 2.4.1 Timeframe Concepts
# MAGIC *Note: Explain the different concepts of the modeling timeframes used for the model development, specifically the observation period, the lag period, along with the performance period.*

# COMMAND ----------

# MAGIC %md
# MAGIC #### 2.4.2 Determination of the Performance and Lag Periods
# MAGIC *Note: Describe the determination process of the lag and performance periods, including the judgemental considerations that were used. Provide a justification of the selections and their consistency with the model product and the observed borrowers’ experience. Explain the different concepts of the modeling timeframes used for the model development, specifically the observation period, the lag period, along with the performance period.*

# COMMAND ----------

# MAGIC %md
# MAGIC #### 2.4.3 Modeling Timeframes
# MAGIC *Note: Describe the different modeling timeframes that were finally selected (i.e., the corresponding periods to the concepts explained in Section 2.4.1) for the model development and validation.*

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2.5 Target Variable Definition
# MAGIC
# MAGIC The model is designed to predict the likelihood of a loan defaulting. The target variable `RISK` (good/bad) is defined using the information in the extracted dataset. The target variable defines the loans status as ‘good’ or ‘bad’. A ‘good’ status means a good credit performance, i.e., the client did not default during the observation period, whereas a ‘bad’ status means a default occurred during the observation period. In the modeling code, ‘good’ is identified as ‘0’, and ‘bad’ is identified as ‘1’ and encoded as our `RISK_EN` column. The following figures depict the target variable distribution (percentage of good/bad), according to the different variables.

# COMMAND ----------

# using matplotlib visualisations
fig, ax = plt.subplots(figsize=(8,5))

# use seaborn for visualization
sns.countplot(ax=ax, x=df_credit["RISK"], palette=["steelblue", "coral"], alpha=0.7)

# add labels
plt.xlabel('Risk variable')
plt.ylabel('Count')
plt.title('Target variable distribution')

# display graph
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2.6 Modeling Populations

# COMMAND ----------

# MAGIC %md
# MAGIC #### 2.6.1 Eligible Population
# MAGIC The following table provides descriptive statistics on the eligible population for the model development, which includes 1,000 observations, in total. Descriptive statistics apply to the overall population, without any data treatment such as exclusion or sampling. ‘NaN’ mostly appears when trying to compute statistics on categorical variables; hence, they may be ignored.

# COMMAND ----------

df_credit.describe(include='all')

# COMMAND ----------

# MAGIC %md
# MAGIC #### 2.6.2 Good-Bad Observations
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

# add labels
plt.legend()
plt.xlabel('Customer Age')
plt.ylabel('Density')
plt.title('Age distribution')

# display graph
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC #### 2.6.3 Indeterminate Observations
# MAGIC *Note: Describe and provide statistics on observations that cannot be classified as good or bad observations.*

# COMMAND ----------

# MAGIC %md
# MAGIC #### 2.6.4 Statistically Inferred Performance Data
# MAGIC *Note: Describe the observations whose performance could not be observed (e.g.,indeterminate observations), the reject inference technique used to infer the performance. The reason supporting the selected technique, along with the considered population should be described as well.*

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2.7 Data Exclusions and Treatment
# MAGIC *Note: Describe exclusions and any treatments (e.g., outlier and missing value treatment, and application of floors and caps) applied to the data, along with the supporting justification.*

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2.8 Sampling Methodology
# MAGIC Two different datasets, training and validation, were created for the modeling purpose. More specifically, a stratified random sampling methodology was used to sample the original dataset: About 80% was used to train the model, and the remaining 20% was considered for the model performance assessment. The tables below present descriptive statistics on the datasets.

# COMMAND ----------

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

df_sample.describe()

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2.9 Modeling Data Assessment
# MAGIC *Note: Describe the final dataset that will be used for the model development. Describe the data quality, using statistics and graphs, describe any data limitations and their potential impact on the model
# MAGIC output.*

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3 Model Development

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3.1 Methodology Selection
# MAGIC *Note: Describe the modeling methodology selection process. More specifically, first present and compare the different alternatives through the literature and industry practice review, and then explain
# MAGIC the rationale behind the selected approach. In addition, outline the mathematical definitions and equations, along with the assumptions and limitations of the selected modeling methodology.*

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3.2 Model Segmentation
# MAGIC *Note: Describe the model segmentation process, including the judgemental considerations, the statistical analyses, and the supporting rationale for the selected segments.*

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3.3 Model Variable Selection
# MAGIC *Note: Describe the variable selection process from the initial list until the selected variables. The statistical analyses with their results and the business considerations should be described in the corresponding sub-sections below. Only relevant and applicable sub-sections should documented. Additional analyses or tests may be added.*

# COMMAND ----------

# MAGIC %md
# MAGIC #### 3.3.1 Variable Reduction
# MAGIC Preliminary analyses were conducted to support the variable selection process. More precisely, the variables were plotted according to different bin categories to assess their probability density. Moreover, weights of evidence (WOEs) which measure the relative risk of each bin within each variable were also calculated and evaluated as part of the variable selection process. The results of the probability density and WOEs for each variable are showed below. 

# COMMAND ----------

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
    return {'num':num, 'n_targ':n_targ, 'base_rate':base_rate, 'base_odds':base_odds, 
            'base_log_odds':base_log_odds, 'nll_null':nll_null, 'logloss_null':logloss_null}

# COMMAND ----------

describe_data_records = describe_data_g_targ(df_credit,'RISK_EN')
describe_data_df = pd.DataFrame.from_dict(describe_data_records, orient='index', columns=['value'])
describe_data_df

# COMMAND ----------

# MAGIC %run ./utils/woe

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
    univariate_sc_plot(df_credit,i,'RISK_EN', n_cuts=3)

# COMMAND ----------

# MAGIC %md
# MAGIC #### 3.3.2 Final Variable Reduction
# MAGIC For the final variable reduction, intervals were created for some continuous variables such as the age, whereas dummies were created for categorical variables such the sex, housing, etc. Results of the analyses are presented below.

# COMMAND ----------

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

feature_red = SelectKBest(f_classif, k=15) \
  .fit(
    df_credit[[i for i in df_credit.columns if i not in ['RISK','RISK_EN']]],
    df_credit['RISK_EN']
  )

X_new = feature_red.transform(df_credit[[i for i in df_credit.columns if i not in ['RISK','RISK_EN']]])

# COMMAND ----------

cols_idxs = feature_red.get_support(indices=True)
train_data_prepared = df_credit.iloc[:,cols_idxs]
train_data_prepared.describe()

X = train_data_prepared[[i for i in train_data_prepared.columns if i not in ['RISK','RISK_EN']]]
y = df_credit['RISK_EN']

# split dataset between training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state=42)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3.4 Model Estimation
# MAGIC
# MAGIC For the model selection and estimation, a 10 fold cross-validation procedure is used to compare and select among different alternative models. The following models were trained using hyperopt for hyper parameter tuning.
# MAGIC <br>
# MAGIC
# MAGIC + K Neighbors
# MAGIC + XGBoost
# MAGIC
# MAGIC Confusion matrices (or error matrices) will be produced for model comparison purposes. Indeed, these matrices easily allow the visualization of the performance of the different models, in terms of actual vs. predicted classes. 

# COMMAND ----------

# broadcast our dataset to be used by our executors
X_train_broadcast = sc.broadcast(X_train)
X_test_broadcast = sc.broadcast(X_test)
y_train_broadcast = sc.broadcast(y_train)
y_test_broadcast = sc.broadcast(y_test)

# COMMAND ----------

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

# unpersist our datasets now that we trained best model
X_train_broadcast.unpersist(blocking = False)
X_test_broadcast.unpersist(blocking = False)
y_train_broadcast.unpersist(blocking = False)
y_test_broadcast.unpersist(blocking = False)

# COMMAND ----------

description = """A 10 fold cross-validation procedure was used to select the best model and hyperparameters across multiple techniques.
Our model selection included XGBoost and K nearest neighbors and selected {} as best fit.
This run was evaluated as our best run that maximizes `cross_val_score`.
""".format(type(best_model).__name__)

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

  # Log dataset used for lineage
  # See https://mlflow.org/docs/latest/python_api/mlflow.html?highlight=log_input#mlflow.log_input
  mlflow.log_input(mlflow_dataset, context='training')

# COMMAND ----------

# ensure random numbers can be reproduced
seed=42

# define our 10 fold cross validation for metrics
kfold = KFold(n_splits=10, random_state=seed, shuffle=True)

# COMMAND ----------

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
    plt.savefig('/tmp/mrmgen_matrix.png')
    plt.show()

# COMMAND ----------

# create MLFlow client to update our experiment with additional metrics / graphs
client = mlflow.tracking.MlflowClient()

# plot confusion matrix for best model
get_confusion_matrix(model)

# equally store the same alongside MLFlow experiment for audit purpose
client.log_artifact(run_id, '/tmp/mrmgen_matrix.png', 'images')

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3.5 Model Scaling
# MAGIC *Note: Describe the model scaling process. More specifically, cover the selection of the scaling equations and parameters, as well as the expert judgements that were considered. Display and interpret the model final results.*

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4 Model Performance Assessment
# MAGIC *Note: Thoroughly assess the model performance in this section. Each sub-section is designed to cover particular dimension that is assessed, outline the analysis or statistical test that is performed and
# MAGIC provide the results interpretation. Keep only relevant and applicable sub-sections. Add additional analyses or tests.*

# COMMAND ----------

# MAGIC %md
# MAGIC ### 4.1 Output Analysis

# COMMAND ----------

# MAGIC %md
# MAGIC ### 4.2 Discriminatory Power Testing

# COMMAND ----------

# MAGIC %md
# MAGIC #### 4.2.1 Accuracy Ratio Test
# MAGIC
# MAGIC To better assess the models’ performance, different accuracy tests including, the accuracy ratio, the precision test, the recall test and the F1 test were performed. Results of these tests are showed in the following tables.

# COMMAND ----------

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

metrics = get_accuracy_ratio(model)
for metric in metrics.keys():
  # log metrics on MLFlow experiment
  client.log_metric(run_id, metric, metrics[metric])

# COMMAND ----------

# MAGIC %md
# MAGIC #### 4.2.2 Kolmogorov-Smirnov Test
# MAGIC
# MAGIC In addition to the aforementioned performance tests, the KS test was also performed, and results are the following.

# COMMAND ----------

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
  # log metrics on MLFlow experiment
  client.log_metric(run_id, metric, metrics[metric])

# COMMAND ----------

# MAGIC %md
# MAGIC ### 4.3 Sensitivity Analysis
# MAGIC A sensitivity analysis was conducted to identify the key variables that mostly impact the model results. For instance, a 5% increase in the following variables, age, credit amount, duration, purpose (if domestic appliances and furniture/equipment) was performed, and the impact reasonableness was assessed. Sensitivity analyses results are showed below.

# COMMAND ----------

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
    plt.savefig('/tmp/mrmgen_sensitivity.png')
    plt.show()

# COMMAND ----------

VAR_OPTIMIZE = [i for i in X_train.columns[:5]]
PERC = 5
ROW = X_train.iloc[[29]]

# disable mlflow autologging, this is not a model but a validation
mlflow.autolog(disable=True)
model = XGBRegressor()
model.fit(X_train, y_train)

S = Simulate(obs=ROW, var=VAR_OPTIMIZE)
d = S.simulate_increase(model=model, percentage=PERC)

# COMMAND ----------

# display graph sensitivity here for the purpose of this documentation
sensitivity(d, title=f'Impact of a {PERC}% increase in target value')

# equally store the same alongside MLFlow experiment for audit purpose
client.log_artifact(run_id, '/tmp/mrmgen_sensitivity.png', 'images')

# COMMAND ----------

# MAGIC %md
# MAGIC ### 4.4 Population Stability Analysis

# COMMAND ----------

# MAGIC %md
# MAGIC ### 4.5 Benchmarking
# MAGIC For the benchmarking, please refer to the section of the model estimation results, where different models were trained, and the results were compared using confusion matrices. 

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5 Model Assumptions and Limitations

# COMMAND ----------

# MAGIC %md
# MAGIC ### 5.1 Model Assumptions
# MAGIC *Note: Describe the key assumptions made throughout the model development process and provide evidence to support their reasonableness and soundness.*

# COMMAND ----------

# MAGIC %md
# MAGIC ### 5.2 Model Limitations
# MAGIC *Note: Describe the key model limitations, their potential impact on the model, as well as the corresponding mitigation action plan(s) to reduce the model risk.*

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6 Model Ongoing Monitoring

# COMMAND ----------

# MAGIC %md
# MAGIC ### 6.1 Ongoing Performance Assessment
# MAGIC *Note: Describe the ongoing model performance monitoring plan. Cover the statistical tests (including e.g., the frequency and acceptance thresholds) that will be performed on an ongoing basis to
# MAGIC ensure the model is still performing adequately.*

# COMMAND ----------

# MAGIC %md
# MAGIC ### 6.2 Documentation Review
# MAGIC *Note: Describe the conditions or types of model changes that trigger the model documentation review, as well as the key components that need to be reviewed.*

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7 References

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8 Model registry
# MAGIC Finally, we will log all evidence required to trigger an independant review of our modeling approach. We show how to do so programmatically, though this process could be done manually from the MLFlow UI.

# COMMAND ----------

model_name = 'credit_adjudication'
model_fqdn = f'{catalog}.{schema}.{model_name}'

# COMMAND ----------

# DBTITLE 0,Credit Model Registration Workflow
model_uri = "runs:/{}/model".format(run_id)
model_version_description = """This version of credit adjudication model was built for the purpose of unity catalog demo. Model was co-developped between EY and Databricks, finding {} as best fit model trained against {} different experiments.
All experiments are tracked and available on MLFlow experiment tracker.""".format(type(best_model).__name__, max_evals)

# Register model
result = mlflow.register_model(model_uri, model_fqdn)
model_version = result.version

# Update model description
client.update_model_version(name=model_fqdn, version=model_version, description=model_version_description)

# Update model tags
client.set_model_version_tag(name=model_fqdn, version=model_version, key='model_type', value=type(best_model).__name__)
client.set_model_version_tag(name=model_fqdn, version=model_version, key='model_selection', value='HYPEROPT')
client.set_model_version_tag(name=model_fqdn, version=model_version, key='model_complexity', value='MEDIUM')
client.set_model_version_tag(name=model_fqdn, version=model_version, key='model_explainability', value='MEDIUM')

# COMMAND ----------

model_registered_description="""Co-developped with EY, This model is a simple example of how organisations could standardize their approach to AI by defining a series of steps that any data science team ought to address prior to a model validation. Although not exhaustive, this shows that most of the questions required by IVU process for a given use case (Credit adjudication) could be addressed upfront to reduce the friction between regulatory imposed silos, increase model validation success rate and drammatically reduce time from exploration to productionization of AI use cases."""

# Update parent model description (done for the purpose of demo, should have been set upfront)
client.update_registered_model(name=model_fqdn, description=model_registered_description)

# Update parent model tags
client.set_registered_model_tag(name=model_fqdn, key='model_materiality', value='HIGH')
client.set_registered_model_tag(name=model_fqdn, key='model_review', value='REQUESTED')

# COMMAND ----------

client.set_registered_model_alias(name=model_fqdn, version=model_version, alias='production')

# COMMAND ----------

# MAGIC %md
# MAGIC Upon completion of this notebook, one can run utility library as follows and get the resulting [PDF](https://github.com/databricks-industry-solutions/fsi-mrm-generation/blob/main/templates/Credit%20Adjudication%20-%20Output.pdf) document
# MAGIC
# MAGIC ```
# MAGIC pip install -r requirements.txt
# MAGIC python databricks.py \
# MAGIC     --db-workspace my-workspace-url \
# MAGIC     --db-token my-workspace-token \
# MAGIC     --model-name my-model-name \
# MAGIC     --model-version my-model-version \
# MAGIC     --output my-model-output.pdf
# MAGIC ```

# COMMAND ----------

# MAGIC %md
# MAGIC <div style="text-align: center; margin-top: 30px;">
# MAGIC     <img src='https://assets.ey.com/content/dam/ey-sites/ey-com/en_gl/generic/logos/20170526-EY-Digital-Brand.svg' alt="Logo" height="100px">
# MAGIC     <br>
# MAGIC     <br>
# MAGIC     <em>Disclaimer: The views and opinions expressed in this blog are those of the authors and do not necessarily reflect the policy or position of EY.</em>
# MAGIC </div>

# COMMAND ----------


