# Databricks notebook source
# MAGIC %md
# MAGIC # Credit Risk Adjudication
# MAGIC
# MAGIC This notebook is a simple example of how organisations could standardize their approach to AI by defining a series of steps that any data science team ought to address prior to a model validation. Although not exhaustive, this shows that most of the questions required by IVU process could be addressed upfront (at model development phase) to reduce the friction between regulatory imposed silos, increase model validation success rate and drammatically reduce time from exploration to productionization of AI use cases. 
# MAGIC
# MAGIC <br>
# MAGIC
# MAGIC <a href="https://www.ey.com/en_ca" target="_blank">
# MAGIC   <img src='https://assets.ey.com/content/dam/ey-sites/ey-com/en_gl/generic/logos/20170526-EY-Digital-Brand.svg' width=150> 
# MAGIC </a>

# COMMAND ----------

# MAGIC %md
# MAGIC ## Executive Summary
# MAGIC
# MAGIC **Note:* Provide a summary that sums up the model purpose, scope of usage, along with the target portfolio. The
# MAGIC model risk ranking, the model limits, and the future developments or enhancements (if applicable)
# MAGIC should be highlighted as well.*
# MAGIC
# MAGIC **[Your text goes here]**

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1 Introduction

# COMMAND ----------

# MAGIC %md
# MAGIC ### 1.1 Model Background and Initiation
# MAGIC
# MAGIC *Note: Describe the history behind the model development. If applicable, describe at high level the different significant changes that were previously made and provide the reason for this current development. It clarifies whether it is a new model or significant changes are being brought to an existing model. This section also should describe the model working group composition and how it was established, if applicable.*
# MAGIC
# MAGIC **[Your text goes here]**

# COMMAND ----------

# MAGIC %md
# MAGIC ### 1.2 Model Purpose
# MAGIC
# MAGIC *Note: Describe the model purpose and scope of usage. More precisely, clarify what the model is designed for, along with the model users, the model interdependencies, and any limitations to the model use.*
# MAGIC
# MAGIC The purpose of this document is to provide a detailed description of the new retail credit adjudication model.

# COMMAND ----------

# MAGIC %md
# MAGIC ### 1.3 Model Portfolio
# MAGIC
# MAGIC *Note: Describe the underlying business product(s) and portfolio of the model. Provide any relevant information about the recent significant changes and trends in the portfolio. Outline key statistics on the portfolio evolution, regarding e.g., the number of borrowers, the amount of exposures, etc.*
# MAGIC
# MAGIC **[Your text goes here]**

# COMMAND ----------

# MAGIC %md
# MAGIC ### 1.4 Model Risk Rating
# MAGIC
# MAGIC *Note: Provide and justify the model risk ranking.*
# MAGIC
# MAGIC **[Your text goes here]**

# COMMAND ----------

# MAGIC %md
# MAGIC ### 1.5 Model Log of Changes (if applicable)
# MAGIC
# MAGIC *Note: Briefly describe all the previous (minor and major) changes that were made to the model,
# MAGIC including the date, justification and how they improved the model performance or use.*
# MAGIC
# MAGIC **[Your text goes here]**

# COMMAND ----------

# MAGIC %md
# MAGIC ### 1.6 Business-Driven Risk Considerations
# MAGIC
# MAGIC *Note: Explain the business risks that are explored and assessed during the model development process, and how they are accounted for in the final model (outputs). Describe and justify any mitigation action (plan) that helps reduce the business-driven risk.*
# MAGIC
# MAGIC **[Your text goes here]**

# COMMAND ----------

# MAGIC %md
# MAGIC ### 1.7 Economic and Market Outlook
# MAGIC
# MAGIC *Note: Explain how the current and forward-looking overall economic conditions may impact the business line and subsequently the model outcome.*
# MAGIC
# MAGIC **[Your text goes here]**

# COMMAND ----------

# MAGIC %md
# MAGIC ### 1.8 Model Development Process
# MAGIC
# MAGIC *Note: Describe the overall model development process, the different milestones of the process, along with the roles and responsibilities of the stakeholders involved at each of these key steps.*
# MAGIC
# MAGIC **[Your text goes here]**

# COMMAND ----------

# MAGIC %md
# MAGIC ### 1.9 Economic and Market Outlook
# MAGIC
# MAGIC *Note: Explain how the current and forward-looking overall economic conditions may impact the business line and subsequently the model outcome.*
# MAGIC
# MAGIC **[Your text goes here]**

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2 Data

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2.1 Borrower Definition
# MAGIC
# MAGIC *Note: Describe the borrowers’ categories of the model portfolio/population. For instance, whether the model applies to borrowers with a certain range of exposure, within a geographical area, or with a minimum/maximum of total asset (e.g., when the model also applies to SMEs). It outlines the borrower identification process in the data bases as well.*
# MAGIC
# MAGIC **[Your text goes here]**

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2.2 Data Sources

# COMMAND ----------

# MAGIC %md
# MAGIC #### 2.2.1 Internal Data Sources
# MAGIC *Note: Describe the internal data sources, as well as their appropriateness with the model purpose and model population.*
# MAGIC
# MAGIC **[Your text goes here]**

# COMMAND ----------

# MAGIC %md
# MAGIC #### 2.2.2 External Data Sources
# MAGIC *Note: Describe the external data sources, as well as their appropriateness with the internal definitions and practices, model purpose and model population.*
# MAGIC
# MAGIC The external data contains personal information on clients with saving and/or checking accounts. Overall, 1,000 observations are included in the dataset. The following describes the different variables along with the features of the dataset.
# MAGIC <br><br>
# MAGIC
# MAGIC *	Age (numeric);
# MAGIC *	Sex (text: male, female);
# MAGIC *	Job (numeric: 0 - unskilled and non-resident, 1 - unskilled and resident, 2 - skilled, 3 - highly skilled);
# MAGIC *	Housing (text: own, rent, or free);
# MAGIC *	Saving accounts (text - little, moderate, quite rich, rich);
# MAGIC *	Checking account (numeric, in DM - Deutsch Mark);
# MAGIC *	Credit amount (numeric, in DM);
# MAGIC *	Duration (numeric, in month);
# MAGIC *	Purpose (text: car, furniture/equipment, radio/TV, domestic appliances, repairs, education, business, vacation/others).

# COMMAND ----------

import pandas as pd
import numpy as np
import seaborn as sns
import plotly.offline as py 

py.init_notebook_mode(connected=True) # allow us to work with offline plotly version

import plotly.graph_objs as go
import plotly.tools as tls

import warnings
from collections import Counter
from xgboost import XGBClassifier

# COMMAND ----------

df_credit = pd.read_csv("data/german_credit_data_risk.csv", index_col=0)
df_credit.loc[df_credit['Risk'] =='good', 'Risk_en'] = 0 
df_credit.loc[df_credit['Risk'] =='bad', 'Risk_en'] = 1 
df_credit.describe(include='all')

# COMMAND ----------

df_credit.head()

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2.3 Data Historical Coverage and Suitability
# MAGIC *Note: Describe the data extraction process, along with the period spanned by the data and the statistics on the extracted observations. The section should not only evidence that the extracted data reflects the business practices and experiences, but is also suitable for the model purpose, modeling methodology and modeling assumptions.*
# MAGIC
# MAGIC **[Your text goes here]**

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2.4 Modeling Timeframes

# COMMAND ----------

# MAGIC %md
# MAGIC #### 2.4.1 Timeframe Concepts
# MAGIC *Note: Explain the different concepts of the modeling timeframes used for the model development, specifically the observation period, the lag period, along with the performance period.*
# MAGIC
# MAGIC **[Your text goes here]**

# COMMAND ----------

# MAGIC %md
# MAGIC #### 2.4.2 Determination of the Performance and Lag Periods
# MAGIC
# MAGIC *Note: Describe the determination process of the lag and performance periods, including the judgemental considerations that were used. Provide a justification of the selections and their consistency with the model product and the observed borrowers’ experience. Explain the different concepts of the modeling timeframes used for the model development, specifically the observation period, the lag period, along with the performance period.*
# MAGIC
# MAGIC **[Your text goes here]**

# COMMAND ----------

# MAGIC %md
# MAGIC #### 2.4.3 Modeling Timeframes
# MAGIC
# MAGIC *Note: Describe the different modeling timeframes that were finally selected (i.e., the corresponding periods to the concepts explained in Section 2.4.1) for the model development and validation.*
# MAGIC
# MAGIC **[Your text goes here]**

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2.5 Target Variable Definition
# MAGIC
# MAGIC *Note: Define the target variable (i.e., bad/default) and the other possible outcomes such as good and intermediate status.*
# MAGIC
# MAGIC The model is designed to predict the likelihood of a loan defaulting. The target variable (good/bad) is defined using the information in the extracted dataset. The target variable defines the loans status as ‘good’ or ‘bad’. A ‘good’ status means a good credit performance, i.e., the client did not default during the observation period, whereas a ‘bad’ status means a default occurred during the observation period.
# MAGIC In the modeling code, ‘good’ is identified as ‘0’, and ‘bad’ is identified as ‘1’. The following figures depict the target variable distribution (percentage of good/bad), according to the different variables.
# MAGIC

# COMMAND ----------

trace0 = go.Bar(
  x = df_credit[df_credit["Risk"]== 'good']["Risk"].value_counts().index.values,
  y = df_credit[df_credit["Risk"]== 'good']["Risk"].value_counts().values,
  name='Good credit'
)

trace1 = go.Bar(
  x = df_credit[df_credit["Risk"]== 'bad']["Risk"].value_counts().index.values,
  y = df_credit[df_credit["Risk"]== 'bad']["Risk"].value_counts().values,
  name='Bad credit'
)

data = [trace0, trace1]
layout = go.Layout()

layout = go.Layout(
    yaxis=dict(title='Count'),
    xaxis=dict(title='Risk Variable'),
    title='Target variable distribution'
)

fig = go.Figure(data=data, layout=layout)
fig.data[0].marker.line.width = 4
fig.data[0].marker.line.color = "black"
fig.data[1].marker.line.width = 4
fig.data[1].marker.line.color = "black"

displayHTML(py.plot(fig, filename='grouped-bar', output_type='div'))

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2.6 Modeling Populations

# COMMAND ----------

# MAGIC %md
# MAGIC #### 2.6.1 Eligible Population
# MAGIC
# MAGIC *Note: Describe and provide statistics on all the extracted population that is qualified for the model development, without any treatment yet (e.g., without exclusions).*
# MAGIC
# MAGIC The following table provides descriptive statistics on the eligible population for the model development, which includes 1,000 observations, in total. Descriptive statistics apply to the overall population, without any data treatment such as exclusion or sampling. ‘NaN’ mostly appears when trying to compute statistics on categorical variables; hence, they may be ignored.

# COMMAND ----------

df_credit.describe(include='all')

# COMMAND ----------

# MAGIC %md
# MAGIC #### 2.6.2 Good-Bad Observations
# MAGIC
# MAGIC *Note: Describe and provide statistics on observations that can classified as good or bad observations.*
# MAGIC
# MAGIC The following provides statistics on the ‘good’ and ‘bad’ observations. Overall, 700 ‘good’ and 300 ‘bad’ observations are found in the dataset. Histograms of ‘good’ and ‘bad’ observations are plotted below. 

# COMMAND ----------

df_good = df_credit.loc[df_credit["Risk"] == 'good']['Age'].values.tolist()
df_bad = df_credit.loc[df_credit["Risk"] == 'bad']['Age'].values.tolist()
df_age = df_credit['Age'].values.tolist()

trace0 = go.Histogram(
    x=df_good,
    histnorm='probability',
    name="Good Credit"
)
trace1 = go.Histogram(
    x=df_bad,
    histnorm='probability',
    name="Bad Credit"
)
trace2 = go.Histogram(
    x=df_age,
    histnorm='probability',
    name="Overall Age"
)

fig = tls.make_subplots(rows=2, cols=2, specs=[[{}, {}], [{'colspan': 2}, None]],
                          subplot_titles=('Good','Bad', 'General Distribuition'))

fig.append_trace(trace0, 1, 1)
fig.append_trace(trace1, 1, 2)
fig.append_trace(trace2, 2, 1)

fig['layout'].update(showlegend=True, title='Age Distribuition', bargap=0.05)
displayHTML(py.plot(fig, filename='custom-sized-subplot-with-subplot-titles',output_type='div'))

# COMMAND ----------

# MAGIC %md
# MAGIC #### 2.6.3 Indeterminate Observations
# MAGIC
# MAGIC *Note: Describe and provide statistics on observations that cannot be classified as good or bad observations.*
# MAGIC
# MAGIC **[Your text goes here]**

# COMMAND ----------

# MAGIC %md
# MAGIC #### 2.6.4 Statistically Inferred Performance Data
# MAGIC
# MAGIC *Note: Describe the observations whose performance could not be observed (e.g.,indeterminate observations), the reject inference technique used to infer the performance. The reason supporting the selected technique, along with the considered population should be described as well.*
# MAGIC
# MAGIC **[Your text goes here]**

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2.7 Data Exclusions and Treatment
# MAGIC
# MAGIC *Note: Describe exclusions and any treatments (e.g., outlier and missing value treatment, and application of floors and caps) applied to the data, along with the supporting justification.*
# MAGIC
# MAGIC **[Your text goes here]**

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2.8 Sampling Methodology
# MAGIC *Note: Describe the data sampling methodology, along with the supporting justification. Provide evidence that the in-sample and out-of-sample populations have enough data points to allow for reliable
# MAGIC statistical test results.*
# MAGIC
# MAGIC Two different datasets, training and validation, were created for the modeling purpose. More specifically, a stratified random sampling methodology was used to sample the original dataset: About 80% was used to train the model, and the remaining 20% was considered for the model performance assessment. The tables below present descriptive statistics on the datasets.

# COMMAND ----------

def sampling(df, target, sample_rate=None, sample_seeds=None, sample_tech=None):
    from sklearn.model_selection import train_test_split
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
df_sample = sampling(df_credit,'Risk_en',sample_rate="80",sample_seeds='33987',sample_tech='Stratified')
df_sample.describe()

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2.9 Modeling Data Assessment
# MAGIC *Note: Describe the final dataset that will be used for the model development. Describe the data quality, using statistics and graphs, describe any data limitations and their potential impact on the model
# MAGIC output.*
# MAGIC
# MAGIC **[Your text goes here]**

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3 Model Development

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3.1 Methodology Selection
# MAGIC *Note: Describe the modeling methodology selection process. More specifically, first present and compare the different alternatives through the literature and industry practice review, and then explain
# MAGIC the rationale behind the selected approach. In addition, outline the mathematical definitions and equations, along with the assumptions and limitations of the selected modeling methodology.*
# MAGIC
# MAGIC **[Your text goes here]**

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3.2 Model Segmentation
# MAGIC
# MAGIC *Note: Describe the model segmentation process, including the judgemental considerations, the statistical analyses, and the supporting rationale for the selected segments.*
# MAGIC
# MAGIC **[Your text goes here]**

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3.3 Model Variable Selection
# MAGIC
# MAGIC *Note: Describe the variable selection process from the initial list until the selected variables. The statistical analyses with their results and the business considerations should be described in the corresponding sub-sections below. Only relevant and applicable sub-sections should documented. Additional analyses or tests may be added.*
# MAGIC
# MAGIC **[Your text goes here]**

# COMMAND ----------

# MAGIC %md
# MAGIC #### 3.3.1 Variable Reduction
# MAGIC
# MAGIC *Note: Describe the first step of the variable selection process, which primarily consists in narrowing down the initial list of variables. Describe each of the listed topics below that are applicable in your approach and add others that were used*
# MAGIC <br/><br/>
# MAGIC
# MAGIC * *Business Considerations*
# MAGIC * *WOE Analysis*
# MAGIC * *Binning Process*
# MAGIC * *Information Value Analysis*
# MAGIC * *Variable Clustering*
# MAGIC * *Additional Considerations (if applicable)*
# MAGIC
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
    #min_bin_ct = np.ceil(1/base_rate).astype(int)
    
    print("Number of records:", num)
    print("Target count:", n_targ)
    print("Target rate:",base_rate)
    print("Target odds:",base_odds)
    print("Target log odds:",base_log_odds)
    print("Null model negative log-likelihood:",nll_null)
    print("Null model LogLoss:",logloss_null)
    
    print("")
    return {'num':num, 'n_targ':n_targ, 'base_rate':base_rate, 'base_odds':base_odds, 
            'base_log_odds':base_log_odds, 'nll_null':nll_null, 'logloss_null':logloss_null}
    
display(describe_data_g_targ(df_credit,'Risk_en'))

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
    #df['diff_prop_good'] = df['prop_good'].diff().abs()
    #df['diff_WoE'] = df['WoE'].diff().abs()
    df['IV'] = (df['prop_n_bad'] - df['prop_n_good']) * df['WoE']
    df['Atr_IV'] = df['IV'].sum()
    return df
  
for i in df_credit.columns:
  if i not in ['Risk', 'Risk_en']:
    display(woe_discrete(df_credit[[i]],i,df_credit['Risk_en']))
    univariate_sc_plot(df_credit,i,'Risk_en', n_cuts=3)

# COMMAND ----------

# MAGIC %md
# MAGIC #### 3.3.2 Final Variable Reduction
# MAGIC
# MAGIC *Note: Describe the additional step(s) to select and refine the model variables after the initial list is reduced.*
# MAGIC <br/><br/>
# MAGIC
# MAGIC * *Univariate Analysis*
# MAGIC * *Multivariate and Multicollinearity Analysis*
# MAGIC * *Additional Considerations (if applicable)*
# MAGIC
# MAGIC For the final variable reduction, intervals were created for some continuous variables such as the age, whereas dummies were created for categorical variables such the sex, housing, etc. Results of the analyses are presented below.

# COMMAND ----------

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.model_selection import train_test_split

interval = (18, 25, 35, 60, 120)

cats = ['Student', 'Young', 'Adult', 'Senior']
df_credit["Age_cat"] = pd.cut(df_credit.Age, interval, labels=cats)
print(df_credit['Age_cat'])
df_credit['Saving accounts'] = df_credit['Saving accounts'].fillna('no_inf')
df_credit['Checking account'] = df_credit['Checking account'].fillna('no_inf')

#Purpose to Dummies Variable
df_credit = df_credit.merge(pd.get_dummies(df_credit.Purpose, drop_first=True, prefix='Purpose'), left_index=True, right_index=True)
#Sex feature in dummies
df_credit = df_credit.merge(pd.get_dummies(df_credit.Sex, drop_first=True, prefix='Sex'), left_index=True, right_index=True)
# Housing get dummies
df_credit = df_credit.merge(pd.get_dummies(df_credit.Housing, drop_first=True, prefix='Housing'), left_index=True, right_index=True)
# Housing get Saving Accounts
df_credit = df_credit.merge(pd.get_dummies(df_credit["Saving accounts"], drop_first=True, prefix='Savings'), left_index=True, right_index=True)
# Housing get Risk
#df_credit = df_credit.merge(pd.get_dummies(df_credit.Risk, prefix='Risk'), left_index=True, right_index=True)
# Housing get Checking Account
df_credit = df_credit.merge(pd.get_dummies(df_credit["Checking account"], drop_first=True, prefix='Check'), left_index=True, right_index=True)
# Housing get Age categorical
df_credit = df_credit.merge(pd.get_dummies(df_credit["Age_cat"], drop_first=True, prefix='Age_cat'), left_index=True, right_index=True)

#Excluding the missing columns
del df_credit["Saving accounts"]
del df_credit["Checking account"]
del df_credit["Purpose"]
del df_credit["Sex"]
del df_credit["Housing"]
del df_credit["Age_cat"]
del df_credit["Risk"]

feature_red = SelectKBest(f_classif, k=15).fit(df_credit[[i for i in df_credit.columns if i not in ['Risk','Risk_en']]],df_credit['Risk_en'])
X_new = feature_red.transform(df_credit[[i for i in df_credit.columns if i not in ['Risk','Risk_en']]])

print(feature_red.get_feature_names_out)

cols_idxs = feature_red.get_support(indices=True)
train_data_prepared = df_credit.iloc[:,cols_idxs]
train_data_prepared.describe()

X = train_data_prepared[[i for i in train_data_prepared.columns if i not in ['Risk','Risk_en']]]
y = df_credit['Risk_en']

# Spliting X and y into train and test version
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state=42)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3.4 Model Estimation
# MAGIC
# MAGIC *Note: Describe the model estimation methodology and its suitability with the model purpose and the modeling data. Outline and interpret the specifications and estimation results.*
# MAGIC
# MAGIC For the model selection and estimation, a 10-fold cross-validation procedure was used to compare and select among different alternative models. The following models were trained and compared:
# MAGIC
# MAGIC •	Logistic regression;
# MAGIC
# MAGIC •	Stochastic gradient descent;
# MAGIC
# MAGIC •	K-neighbors;
# MAGIC
# MAGIC •	Decision tree;
# MAGIC
# MAGIC •	Multinomial naive Bayes;
# MAGIC
# MAGIC •	Random forest;
# MAGIC
# MAGIC •	Support vector;
# MAGIC
# MAGIC •	Extreme gradient boosting.
# MAGIC
# MAGIC Confusion matrices (or error matrices) were then produced for model comparison purposes. Indeed, these matrices easily allow the visualization of the performance of the different models, in terms of actual vs. predicted classes. The resulting confusion matrices are presented below.
# MAGIC

# COMMAND ----------

from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_predict, cross_val_score, KFold
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

clf_xgb = XGBClassifier()

# to feed the random state
seed = 7

# prepare models
models = []

models.append((clf_xgb))

fig, axes = plt.subplots(1,1, figsize=(15, 8))

for model in models:
    kfold = KFold(n_splits=10, random_state=seed, shuffle=True) # use kfold cross validation for training
    y_train_pred = cross_val_predict(model, X_train, y_train, cv=kfold) # cross_val_predict will train the model and give y-values
    cf_matrix = confusion_matrix(y_train, y_train_pred)

    group_names = ['True Neg', 'False Pos', 'False Neg', 'True Pos']
    group_counts = ['{0:0.0f}'.format(value) for value in cf_matrix.flatten()]
    group_percentages = ['{0:.2%}'.format(value) for value in cf_matrix.flatten()/np.sum(cf_matrix)]
    labels = [f'{v1}\n{v2}\n{v3}' for v1, v2, v3 in zip(group_names,group_counts,group_percentages)]
    labels = np.asarray(labels).reshape(2,2)

    sns.heatmap(cf_matrix, annot=labels, fmt='', cmap='Blues') # visual confusion matrix of all models

fig.suptitle("Confusion Matrices", fontsize=30)
plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3.5 Model Scaling
# MAGIC
# MAGIC *Note: Describe the model scaling process. More specifically, cover the selection of the scaling equations and parameters, as well as the expert judgements that were considered. Display and interpret the model final results.*
# MAGIC
# MAGIC **[Your text goes here]**

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4 Model Performance Assessment
# MAGIC
# MAGIC *Note: Thoroughly assess the model performance in this section. Each sub-section is designed to cover particular dimension that is assessed, outline the analysis or statistical test that is performed and
# MAGIC provide the results interpretation. Keep only relevant and applicable sub-sections. Add additional analyses or tests.*
# MAGIC
# MAGIC **[Your text goes here]**

# COMMAND ----------

# MAGIC %md
# MAGIC ### 4.1 Output Analysis
# MAGIC **[Your text goes here]**

# COMMAND ----------

# MAGIC %md
# MAGIC ### 4.2 Discriminatory Power Testing
# MAGIC
# MAGIC **[Your text goes here]**

# COMMAND ----------

# MAGIC %md
# MAGIC #### 4.2.1 Accuracy Ratio Test
# MAGIC
# MAGIC To better assess the models’ performance, different accuracy tests including, the accuracy ratio, the precision test, the recall test and the F1 test were performed. Results of these tests are showed in the following tables.

# COMMAND ----------

from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score, KFold

# to feed the random state
seed = 7

# prepare models
models = []

models.append(('XGB', clf_xgb))

# evaluate each model in turn
# creating a list for the kfold CV metric scores
accuracy_results = []
precision_results = []
recall_results = []
f1_results = []

# creating a list of the average metric scores
avg_accuracy_results = []
avg_precision_results = []
avg_recall_results = []
avg_f1_results = []

names = []

for name, model in models:
        kfold = KFold(n_splits=10, random_state=seed, shuffle=True)
        cv_accuracy = cross_val_score(model, X_train, y_train, cv=kfold, scoring='accuracy')
        cv_precision = cross_val_score(model, X_train, y_train, cv=kfold, scoring='precision')
        cv_recall = cross_val_score(model, X_train, y_train, cv=kfold, scoring='recall')
        cv_f1 = cross_val_score(model, X_train, y_train, cv=kfold, scoring='f1')
        accuracy_results.append(cv_accuracy)
        precision_results.append(cv_precision)
        recall_results.append(cv_recall)
        f1_results.append(cv_f1)

        avg_accuracy_results.append(cv_accuracy.mean())
        avg_precision_results.append(cv_precision.mean())
        avg_recall_results.append(cv_recall.mean())
        avg_f1_results.append(cv_f1.mean())

        names.append(name)

clf_metrics_data = {
    "Accuracy": avg_accuracy_results,
    "Precision": avg_precision_results,
    "Recall": avg_recall_results,
    "F1": avg_f1_results
}

clf_metrics_df = pd.DataFrame(clf_metrics_data, index=names) # Creating a dataframe of all the metrics for each model
clf_metrics_df.index.name = "Model"

clf_metrics_df

# COMMAND ----------

# MAGIC %md
# MAGIC #### 4.2.2 Kolmogorov-Smirnov Test
# MAGIC
# MAGIC In addition to the aforementioned performance tests, the KS test was also performed, and results are the following.

# COMMAND ----------

from scipy.stats import kstest
from scipy.stats import ks_2samp

data1 = X_train.iloc[:,0]
data2 = X_train.iloc[:,1]
ks_2samp(data1,data2)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 4.3 Sensitivity Analysis
# MAGIC
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

    @staticmethod
    def plot_simulation(d, **kwargs):
        fig, ax = plt.subplots()
        sns.barplot(x='test', y='simulated', data=d, palette='deep', ax=ax)
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
        plt.tight_layout()
        plt.show()

model = XGBRegressor()
model.fit(X_train, y_train)

VAR_OPTIMIZE = [i for i in X_train.columns[:5]]
PERC = 5
ROW = X_train.iloc[[29]]

S = Simulate(obs=ROW, var=VAR_OPTIMIZE)
d = S.simulate_increase(model=model, percentage=PERC)
S.plot_simulation(d, title=f'Impact of a {PERC}% increase of {VAR_OPTIMIZE} in target value')

# COMMAND ----------

# MAGIC %md
# MAGIC ### 4.4 Population Stability Analysis
# MAGIC
# MAGIC **[Your text goes here]**

# COMMAND ----------

# MAGIC %md
# MAGIC ### 4.5 Benchmarking
# MAGIC
# MAGIC
# MAGIC For the benchmarking, please refer to the section of the model estimation results, where different models were trained, and the results were compared using confusion matrices. 

# COMMAND ----------

from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_predict, cross_val_score, KFold
from sklearn.metrics import confusion_matrix

clf_lr = LogisticRegression(random_state=42)
clf_sgd = SGDClassifier(random_state=42)
clf_knn = KNeighborsClassifier()
clf_cart = DecisionTreeClassifier(random_state=42)
clf_nb = MultinomialNB()
clf_rf = RandomForestClassifier(random_state=42)
clf_svm = SVC(random_state=42)
clf_xgb = XGBClassifier()

# to feed the random state
seed = 7

# prepare models
models = []
models.append((clf_lr))
models.append((clf_sgd))
models.append((clf_knn))
models.append((clf_cart))
models.append((clf_nb))
models.append((clf_rf))
models.append((clf_svm))
models.append((clf_xgb))

fig, axes = plt.subplots(2, 4, figsize=(15, 8))

for model, ax in zip(models, axes.flatten()):
    kfold = KFold(n_splits=10, random_state=seed, shuffle=True) # use kfold cross validation for training
    y_train_pred = cross_val_predict(model, X_train, y_train, cv=kfold) # cross_val_predict will train the model and give y-values
    cf_matrix = confusion_matrix(y_train, y_train_pred)

    group_names = ['True Neg', 'False Pos', 'False Neg', 'True Pos']
    group_counts = ['{0:0.0f}'.format(value) for value in cf_matrix.flatten()]
    group_percentages = ['{0:.2%}'.format(value) for value in cf_matrix.flatten()/np.sum(cf_matrix)]
    labels = [f'{v1}\n{v2}\n{v3}' for v1, v2, v3 in zip(group_names,group_counts,group_percentages)]
    labels = np.asarray(labels).reshape(2,2)

    sns.heatmap(cf_matrix, annot=labels, fmt='', cmap='Blues', ax=ax) # visual confusion matrix of all models
    ax.title.set_text(type(model).__name__)

#plt.title("Confusion Matrices")
fig.suptitle("Confusion Matrices", fontsize=30)
plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5 Model Assumptions and Limitations

# COMMAND ----------

# MAGIC %md
# MAGIC ### 5.1 Model Assumptions
# MAGIC *Note: Describe the key assumptions made throughout the model development process and provide evidence to support their reasonableness and soundness.*
# MAGIC
# MAGIC
# MAGIC **[Your text goes here]**

# COMMAND ----------

# MAGIC %md
# MAGIC ### 5.2 Model Limitations
# MAGIC *Note: Describe the key model limitations, their potential impact on the model, as well as the corresponding mitigation action plan(s) to reduce the model risk.*
# MAGIC
# MAGIC **[Your text goes here]**

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6 Model Ongoing Monitoring

# COMMAND ----------

# MAGIC %md
# MAGIC ### 6.1 Ongoing Performance Assessment
# MAGIC *Note: Describe the ongoing model performance monitoring plan. Cover the statistical tests (including e.g., the frequency and acceptance thresholds) that will be performed on an ongoing basis to
# MAGIC ensure the model is still performing adequately.*
# MAGIC
# MAGIC **[Your text goes here]**

# COMMAND ----------

# MAGIC %md
# MAGIC ### 6.2 Documentation Review
# MAGIC *Note: Describe the conditions or types of model changes that trigger the model documentation review, as well as the key components that need to be reviewed.*
# MAGIC
# MAGIC **[Your text goes here]**

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7 References
# MAGIC
# MAGIC **[Your text goes here]**

# COMMAND ----------

# MAGIC %md
# MAGIC <div style="text-align: center; margin-top: 30px;">
# MAGIC     <img src='https://raw.githubusercontent.com/databricks-industry-solutions/fsi-mrm-generation/main/templates/figs/ey_logo.png' height="50px">
# MAGIC     <br>
# MAGIC     <em>Disclaimer: The views and opinions expressed in this blog are those of the authors and do not necessarily reflect the policy or position of EY.</em>
# MAGIC </div>
