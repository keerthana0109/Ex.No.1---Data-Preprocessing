# Ex.No.1---Data-Preprocessing
## AIM:

To perform Data preprocessing in a data set downloaded from Kaggle

##REQUIPMENTS REQUIRED:
Hardware – PCs
Anaconda – Python 3.7 Installation / Google Colab /Jupiter Notebook

## RELATED THEORETICAL CONCEPT:

Kaggle :
Kaggle, a subsidiary of Google LLC, is an online community of data scientists and machine learning practitioners. Kaggle allows users to find and publish data sets, explore and build models in a web-based data-science environment, work with other data scientists and machine learning engineers, and enter competitions to solve data science challenges.

Data Preprocessing:

Pre-processing refers to the transformations applied to our data before feeding it to the algorithm. Data Preprocessing is a technique that is used to convert the raw data into a clean data set. In other words, whenever the data is gathered from different sources it is collected in raw format which is not feasible for the analysis.
Data Preprocessing is the process of making data suitable for use while training a machine learning model. The dataset initially provided for training might not be in a ready-to-use state, for e.g. it might not be formatted properly, or may contain missing or null values.Solving all these problems using various methods is called Data Preprocessing, using a properly processed dataset while training will not only make life easier for you but also increase the efficiency and accuracy of your model.

Need of Data Preprocessing :

For achieving better results from the applied model in Machine Learning projects the format of the data has to be in a proper manner. Some specified Machine Learning model needs information in a specified format, for example, Random Forest algorithm does not support null values, therefore to execute random forest algorithm null values have to be managed from the original raw data set.
Another aspect is that the data set should be formatted in such a way that more than one Machine Learning and Deep Learning algorithm are executed in one data set, and best out of them is chosen.


## ALGORITHM:
Importing the libraries
Importing the dataset
Taking care of missing data
Encoding categorical data
Normalizing the data
Splitting the data into test and train

## PROGRAM:
```
from google.colab import files
uploaded = files.upload()

#importing required packages
import pandas as pd
import io

#importing our dataset
df=pd.read_csv(io.BytesIO(uploaded['data.csv']))
df.isnull().sum() #identifying redundant or missing values

x = df['Calories'].mean() # finding the mean for 'Calories' col
df['Calories'].fillna(x,inplace = True) # Replacing the null values with neighbouring mean value
print(df)

print(df.describe())

print(df.isnull().sum())
# It shows that we have no missing values

import seaborn as sns
import matplotlib.pyplot as plt

# To identify the outliers by visualizing the box plot and scatter plot
sns.boxplot(df['Calories'])

y = df['Duration']
plt.scatter(df['Calories'],y)

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

x=df.iloc[:,:-1].values
y=df.iloc[:,-1].values

scaler = MinMaxScaler()
df1 = pd.DataFrame(scaler.fit_transform(df))

xtrain,ytrain,xtest,ytest=train_test_split(x,y,test_size=0.2,random_state=2)
print(len(xtrain))
print(len(xtest))

print(xtest)

sc = StandardScaler()
df1 = sc.fit_transform(df)

print(df1)
```


## OUTPUT:
/ Show the result/

## RESULT
/Type your result here/
