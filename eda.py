
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from sklearn import tree
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report


import sklearn.ensemble as se
from sklearn import svm

df = pd.read_excel('/content/Telco_customer_churn.xlsx')
df.head()

df['Churn Label'].value_counts()/np.float(len(df))

## print

df.columns = df.columns.str.replace(' ', '')

dd  = df.loc[df['TotalCharges'] != ' ']
dd['TotalCharges'] = dd['TotalCharges'].astype('float')
dd['TotalCharges'] = dd['TotalCharges'].astype('float')


plt.figure(figsize=(24,20))

plt.subplot(4, 2, 1)
fig = dd.boxplot(column='MonthlyCharges')
fig.set_title('')
fig.set_ylabel('Monthly Charges')


plt.subplot(4, 2, 2)
fig = dd['MonthlyCharges'].hist(bins=20)
fig.set_xlabel('Monthly Charges')
fig.set_ylabel('Charges')


plt.subplot(4, 2, 3)
fig = dd.boxplot(column='TotalCharges')
fig.set_title('')
fig.set_ylabel('Total Charges')


plt.subplot(4, 2, 4)
fig = dd['TotalCharges'].hist(bins=20)
fig.set_xlabel('Total Charges')
fig.set_ylabel('Charges')

plt.subplot(4, 2, 5)
fig = dd.boxplot(column='CLTV')
fig.set_title('')
fig.set_ylabel('Monthly CLTV')

plt.subplot(4, 2, 6)
fig = dd['CLTV'].hist(bins=20)
fig.set_xlabel('CLTV')
fig.set_ylabel('VAlue')


plt.subplot(4, 2, 7)
fig = dd.boxplot(column='TenureMonths')
fig.set_title('')
fig.set_ylabel('Months')

plt.subplot(4, 2, 8)
fig = dd['TenureMonths'].hist(bins=20)
fig.set_xlabel('Tenure Months')
fig.set_ylabel('Months')


plt.figure(figsize=(5,5))

sns.set(rc={'figure.figsize':(20,15)})
fig, axs = plt.subplots(ncols=3)
sns.countplot(x='PaymentMethod', hue='ChurnLabel',  data=dd ,ax=axs[0])

sns.countplot(x='StreamingTV', hue='ChurnLabel', data=dd,ax=axs[1])

sns.countplot(x='Gender', hue='ChurnLabel',  data=dd,ax=axs[2])


fig, axs = plt.subplots(ncols=3)
sns.countplot(x='Contract', hue='ChurnLabel',  data=dd ,ax=axs[0])


sns.countplot(x='TechSupport', hue='ChurnLabel', data=dd ,ax=axs[1])


sns.countplot(x='Partner', hue='ChurnLabel', data=dd ,ax=axs[2])


fig, axs = plt.subplots(ncols=3)
sns.countplot(x='PhoneService', hue='ChurnLabel',  data=dd ,ax=axs[0])
plt.xticks(rotation=45)

sns.countplot(x='MultipleLines', hue='ChurnLabel',  data=dd ,ax=axs[1])
plt.xticks(rotation=45)

sns.countplot(x='InternetService', hue='ChurnLabel', data=dd ,ax=axs[2])
plt.xticks(rotation=45)


g = sns.FacetGrid(dd, col="PaymentMethod", height=5, aspect=1)
g.map(sns.barplot, "SeniorCitizen", "TotalCharges")
