## Introduction


"Wine has been a part of civilized life for some seven thousand years. It is the only beverage that feeds the body, soul and spirit of man and at the same time stimulates the mind..." 
This is a quote by Robert Mondavi

Having read that , I was wondering if we could predict the quality of wine when the values of its features are given to us. While doing so manually takes many years of experience,expertise and deep knowledge in the field of wine, its a seconds job with machine learning. Using suitable model and training the model using suitable dataset yields us the answer.

![Image](https://miro.medium.com/max/575/0*OGDB-JY7IDYQE48U.jpg)

Wine quality prediction can be solved as classification or regression problem. Here we are solving it as a classification problem. The main objective of this project is to experiment with the various classifier models and pick out the model yielding highest accuracy. We have made use of Kaggle's red wine dataset in this project. Click [here](https://github.com/Panchami28/Wine-quality-prediction/blob/master/datasets_4458_8204_winequality-red.csv) to download the dataset. In this dataset , quality of red wine is determined by 11 input variables namely 
fixed acidity,
volatile acidity,
citric acid,
residual sugar,
chlorides,
free sulfur dioxide,
total sulfur dioxide,
density,
pH,
sulfates,
alcohol.


## Setup

### Importing the required modules
We start out by importing the standard libraries

```markdown
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
%matplotlib inline
```
### Loading the dataset
Dataset is then loaded into the notebook and converted to a dataframe

```markdown
df=pd.read_csv('datasets_4458_8204_winequality-red.csv')
```

### Gaining information about the dataset
Reading the data and gaining information
```markdown
df.head()
print("Rows, columns: " + str(df.shape))
df.describe()
df.info()
# Checking for missing Values
print(df.isna().sum())
```
There are no missing values in this dataset.Hence it is not necessary to drop any columns. 

### Exploratory data analysis (EDA)
Now that we came to know about the data that we are using for tarining the model, let us perform some EDA (Exploratory data aanalysis) to understand the relationship between various input variables and its variation with the output

```markdown
fig = plt.figure(figsize = (10,6))
sns.barplot(x = 'quality', y = 'fixed acidity', data = df)
```
Output:
<img src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAlcAAAFzCAYAAAAT7iw5AAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAALEgAACxIB0t1+/AAAADh0RVh0U29mdHdhcmUAbWF0cGxvdGxpYiB2ZXJzaW9uMy4yLjIsIGh0dHA6Ly9tYXRwbG90bGliLm9yZy+WH4yJAAAVHklEQVR4nO3df7DldX3f8dfbXQmwRcGyU+qPuOhQrU0Tf+xYDYk6ElN/oKbGTKQDaekk2Ez9EfKDMeNMjW3SmRLsmJg0KVGJiahNCDbBKtFRMdG2xF3E+gPTWlRkw8oyBhH8Aci7f9xDs7uF3cPu53vPOfc+HjN3zv2ec/Z+33NmZ/d5vz+ruwMAwBgPWvQAAAAbibgCABhIXAEADCSuAAAGElcAAAOJKwCAgbYueoD9nXzyyb1jx45FjwEAcFi7d+++pbu3H/z8UsXVjh07smvXrkWPAQBwWFX1pft63m5BAICBxBUAwEDiCgBgIHEFADCQuAIAGEhcAQAMJK4AAAYSVwAAA4krAICBxBUAwEDiCgBgIHEFADDQUt24GQDYPC644ILs3bs3p5xySi688MJFjzOMuDpKG/UvBgBMbe/evdmzZ8+ixxhOXB2ljfoXAwA4Mo65AgAYSFwBAAwkrgAABhJXAAADiSsAgIHEFQDAQOIKAGAgcQUAMJCLiAJA3HGDccQVAMQdNxjHbkEAgIHEFQDAQOIKAGAgcQUAMJC4AgAYSFwBAAwkrgAABhJXAAADuYgoAAvxGz93xaJHOMCtt9zx/x6XabZXvOGFix6BB8iWKwCAgcQVAMBA4goAYCBxBQAwkLgCABhIXAEADCSuAAAGElcAAAOJKwCAgVyhHQCSbDvmIQc8bkS/cvZLFz3CAb5689fWHvfetFSzvfbtlx3VnxdXAJDk9Me+ZNEjsEGIK4AldMEFF2Tv3r055ZRTcuGFFy56HOABEFcAS2jv3r3Zs2fPoscAjoAD2gEABhJXAAADiSsAgIFW7pirp/zC7y16hAOccMvXsyXJDbd8falm2/2rP7HoEWClfOQZz1z0CAf45tYtSVW+eeONSzXbM//sI4seAZbeysUVsP6cuQYwP3EFHJYz1wDmJ65YObaiALDMxBUrx1YUAJaZswUBAAay5QqW0OlvOn3RIxzgmFuPyYPyoHz51i8v1Wwfe+XHFj3CZE7sPuARWB3iCmAJnf2dexY9AnCEJt0tWFXnV9VnqurTVfXOqjp2yvUBACzaZHFVVY9I8qokO7v7e5JsSfKyqdYHALAMpj6gfWuS46pqa5Ljk/zVxOsDAFioyeKqu/ckuSjJDUluSvK17n7/VOsDptPHd+7Zdk/6eAdXAxzOlLsFT0ry4iSnJnl4km1VdfZ9vO+8qtpVVbv27ds31TjAUbjr9Lty53PuzF2n37XoUQCW3pS7BX8oyRe6e19335Xk8iTff/Cbuvvi7t7Z3Tu3b98+4TgAANObMq5uSPK0qjq+qirJGUmum3B9AAALN+UxV1cnuSzJNUk+NVvXxVOtDwBgGUx6EdHufl2S1025DgCAZeLeggAAA4krAICB3FuQw7rh3/zDRY9wgLu/+rAkW3P3V7+0VLN997/+1KJHAGAJ2HIFADCQuAIAGEhcAQAMJK4AAAYSVwAAA4krAICBXIoBAFiIY7c86IDHjUJcAQAL8aS/fcKiR5jExkpFAIAFE1cAAAOJKwCAgcQVAMBADmg/Svccs+2ARwBgcxNXR+mO03540SNsOicfe0+Su2ePALBcxBUr5+e/99ZFjwAA98sxVwAAA4krAICBxBUAwEDiCgBgIHEFADCQuAIAGEhcAQAMJK4AAAYSVwAAA4krAICBxBUAwEDiCgBgIHEFADCQuAIAGEhcAQAMJK4AAAYSVwAAA4krAICBxBUAwEDiCgBgIHEFADCQuAIAGEhcAQAMJK4AAAYSVwAAA4krAICBxBUAwEDiCgBgIHEFADCQuAIAGEhcAQAMJK4AAAYSVwAAA4krAICBxBUAwEDiCgBgIHEFADCQuAIAGEhcAQAMNGlcVdWJVXVZVX2uqq6rqqdPuT4AgEXbOvHP/7UkV3b3S6vqmCTHT7w+AICFmiyuquqhSZ6R5J8nSXffmeTOqdYHALAMptwteGqSfUkuqapPVNWbq2rbwW+qqvOqaldV7dq3b9+E4wAATG/KuNqa5MlJfqu7n5TkjiSvOfhN3X1xd+/s7p3bt2+fcBwAgOlNGVc3Jrmxu6+eLV+WtdgCANiwJour7t6b5MtV9bjZU2ck+exU6wMAWAZTny34yiSXzs4UvD7JuROvDwBgoQ4bV1W1O8lbk7yju//6gfzw7r42yc4jnA0AYOXMs1vwx5M8PMnHq+pdVfWPq6omngsAYCUdNq66+/Pd/dokfy/JO7K2FetLVfX6qnrY1AMCAKySuQ5or6rvTfKGJL+a5I+S/FiS25J8aLrRAABWz7zHXN2a5C1JXtPd3569dHVVnT7lcAAAq2aeswV/rLuv3/+Jqjq1u7/Q3S+ZaC4AgJU0z27By+Z8DgBg07vfLVdV9fgk/yDJQ6tq/y1UD0ly7NSDAQCsokPtFnxckjOTnJjkhfs9//UkPzXlUAAAq+p+46q7/zjJH1fV07v7v6/jTAAAK+tQuwUv6O4Lk/zTqjrr4Ne7+1WTTgYAsIIOtVvwutnjrvUYBABgIzjUbsErZo9vW79xAABW26F2C16RpO/v9e5+0SQTAQCssEPtFrxo9viSJKckefts+awkX5lyKACAVXWo3YIfSZKqekN379zvpSuqynFYAAD3YZ4rtG+rqsfcu1BVpybZNt1IAACra557C56f5Kqquj5JJXl0kpdPOhUAwIo6bFx195VVdVqSx8+e+lx3f3vasQAAVtOhzhZ8dnd/6KD7CibJY6sq3X35xLMBAKycQ225emaSD+XA+wreq5OIKwCAgxzqbMHXzR7PXb9xAABW22HPFqyqf1dVJ+63fFJV/fK0YwEArKZ5LsXwvO6+9d6F7v7rJM+fbiQAgNU1T1xtqarvunehqo5L8l2HeD8AwKY1z3WuLk3ywaq6ZLZ8bhI3cwYAuA/zXOfq31fV/0xyxuypf9vdfzrtWAAAq2meLVfp7vcled/EswAArLx5zhZ8WlV9vKpur6o7q+o7VXXbegwHALBq5jmg/TeSnJXkfyc5LslPJvnNKYcCAFhV88RVuvvzSbZ093e6+5Ikz512LACA1TTPMVffqKpjklxbVRcmuSlzRhkAwGYzTySdM3vfK5LckeRRSX50yqEAAFbVPJdi+NLs228lef204wAArDa79wAABhJXAAADiSsAgIHu95irqroiSd/f6939okkmAgBYYYc6oP2i2eNLkpyS5O2z5bOSfGXKoQAAVtX9xlV3fyRJquoN3b1zv5euqKpdk08GALCC5jnmaltVPebehao6Ncm26UYCAFhd81yh/fwkV1XV9UkqyaOTvHzSqQAAVtQ8FxG9sqpOS/L42VOf6+5vTzsWAMBqOuxuwao6PskvJHlFd38yyXdX1ZmTTwYAsILmOebqkiR3Jnn6bHlPkl+ebCIAgBU2T1w9trsvTHJXknT3N7J27BUAAAeZJ67urKrjMrugaFU9NoljrgAA7sM8Zwv+UpIrkzyqqi5NcnqSc6ccCgBgVc1ztuD7q2p3kqdlbXfgq7v7lsknAwBYQfOcLfj7Se7u7v/a3e/J2kVFPzj9aAAAq2eeY64+muTqqnp+Vf1Ukg8keeO0YwEArKZ5dgv+p6r6TJIPJ7klyZO6e+/kkwEArKB5dguek+StSX4iye8meW9Vfd/EcwEArKR5zhb80SQ/0N03J3lnVb07yduSPHHSyQAAVtA8uwV/5KDlv6iqp043EgDA6rrfuKqqC7r7wqp6U2YXED3Iq6YbCwBgNR1qy9VnZ4+71mMQAICN4FBx9eNJ3pPkxO7+tSNdQVVtyVqg7enuM4/05wAArIJDnS34lKp6eJJ/UVUnVdXD9v96AOt4dZLrjm5MAIDVcKgtV7+d5INJHpNkd9ZufXOvnj1/SFX1yCQvSPIrSX72yMcEAFgN97vlqrt/vbv/fpK3dvdjuvvU/b4OG1Yzb0xyQZJ7RgwLALDsDnsR0e7+6SP5wVV1ZpKbu3v3Yd53XlXtqqpd+/btO5JVAQAsjXnuLXikTk/yoqr6YpJ3JXl2Vb394Dd198XdvbO7d27fvn3CcQAApjdZXHX3L3b3I7t7R5KXJflQd5891foAAJbBlFuuAAA2nXnuLXjUuvuqJFetx7oAABbJlisAgIHEFQDAQOIKAGAgcQUAMJC4AgAYSFwBAAwkrgAABhJXAAADiSsAgIHEFQDAQOIKAGAgcQUAMJC4AgAYSFwBAAwkrgAABhJXAAADiSsAgIHEFQDAQOIKAGAgcQUAMJC4AgAYSFwBAAwkrgAABhJXAAADiSsAgIHEFQDAQOIKAGAgcQUAMJC4AgAYSFwBAAwkrgAABhJXAAADiSsAgIHEFQDAQOIKAGAgcQUAMJC4AgAYSFwBAAwkrgAABhJXAAADiSsAgIHEFQDAQOIKAGAgcQUAMJC4AgAYSFwBAAwkrgAABhJXAAADiSsAgIHEFQDAQOIKAGAgcQUAMJC4AgAYSFwBAAwkrgAABhJXAAADiSsAgIEmi6uqelRVfbiqPltVn6mqV0+1LgCAZbF1wp99d5Kf6+5rquqEJLur6gPd/dkJ1wkAsFCTbbnq7pu6+5rZ919Pcl2SR0y1PgCAZbAux1xV1Y4kT0py9X28dl5V7aqqXfv27VuPcQAAJjN5XFXV30ryR0l+prtvO/j17r64u3d2987t27dPPQ4AwKQmjauqenDWwurS7r58ynUBACyDKc8WrCRvSXJdd/+HqdYDALBMptxydXqSc5I8u6qunX09f8L1AQAs3GSXYujujyapqX4+AMAycoV2AICBxBUAwEDiCgBgIHEFADCQuAIAGEhcAQAMJK4AAAYSVwAAA4krAICBxBUAwEDiCgBgIHEFADCQuAIAGEhcAQAMJK4AAAYSVwAAA4krAICBxBUAwEDiCgBgIHEFADCQuAIAGEhcAQAMJK4AAAYSVwAAA4krAICBxBUAwEDiCgBgIHEFADCQuAIAGEhcAQAMJK4AAAYSVwAAA4krAICBxBUAwEDiCgBgIHEFADCQuAIAGEhcAQAMJK4AAAYSVwAAA4krAICBxBUAwEDiCgBgIHEFADCQuAIAGEhcAQAMJK4AAAYSVwAAA4krAICBxBUAwEDiCgBgIHEFADCQuAIAGEhcAQAMJK4AAAYSVwAAA00aV1X13Kr6y6r6fFW9Zsp1AQAsg8niqqq2JPnNJM9L8oQkZ1XVE6ZaHwDAMphyy9VTk3y+u6/v7juTvCvJiydcHwDAwk0ZV49I8uX9lm+cPQcAsGFVd0/zg6temuS53f2Ts+Vzkvyj7n7FQe87L8l5s8XHJfnLSQaa1slJbln0EJuMz3z9+czXn898/fnM198qf+aP7u7tBz+5dcIV7knyqP2WHzl77gDdfXGSiyecY3JVtau7dy56js3EZ77+fObrz2e+/nzm628jfuZT7hb8eJLTqurUqjomycuS/MmE6wMAWLjJtlx1991V9Yokf5pkS5K3dvdnplofAMAymHK3YLr7vUneO+U6lsRK79ZcUT7z9eczX38+8/XnM19/G+4zn+yAdgCAzcjtbwAABhJXR6Gqjq2qv6iqT1bVZ6rq9YueaTOoqi1V9Ymqes+iZ9ksquqLVfWpqrq2qnYtep7NoKpOrKrLqupzVXVdVT190TNtZFX1uNnf73u/bquqn1n0XBtdVZ0/+//z01X1zqo6dtEzjWC34FGoqkqyrbtvr6oHJ/lokld39/9Y8GgbWlX9bJKdSR7S3Wcuep7NoKq+mGRnd6/qtWhWTlW9Lcmfd/ebZ2dcH9/dty56rs1gdvu2PVm7NuOXFj3PRlVVj8ja/5tP6O5vVtUfJHlvd//uYic7erZcHYVec/ts8cGzL7U6oap6ZJIXJHnzomeBqVTVQ5M8I8lbkqS77xRW6+qMJP9HWK2LrUmOq6qtSY5P8lcLnmcIcXWUZruork1yc5IPdPfVi55pg3tjkguS3LPoQTaZTvL+qto9u6sC0zo1yb4kl8x2gb+5qrYteqhN5GVJ3rnoITa67t6T5KIkNyS5KcnXuvv9i51qDHF1lLr7O939xKxdgf6pVfU9i55po6qqM5Pc3N27Fz3LJvQD3f3kJM9L8q+q6hmLHmiD25rkyUl+q7uflOSOJK9Z7Eibw2wX7IuS/OGiZ9noquqkJC/O2i8TD0+yrarOXuxUY4irQWab7D+c5LmLnmUDOz3Ji2bH/7wrybOr6u2LHWlzmP2Gme6+Ocm7kzx1sRNteDcmuXG/LeGXZS22mN7zklzT3V9Z9CCbwA8l+UJ37+vuu5JcnuT7FzzTEOLqKFTV9qo6cfb9cUmek+Rzi51q4+ruX+zuR3b3jqxttv9Qd2+I33KWWVVtq6oT7v0+yQ8n+fRip9rYuntvki9X1eNmT52R5LMLHGkzOSt2Ca6XG5I8raqOn50gdkaS6xY80xCTXqF9E/i7Sd42O7PkQUn+oLtdHoCN5u8keffav33ZmuQd3X3lYkfaFF6Z5NLZbqrrk5y74Hk2vNkvD89J8vJFz7IZdPfVVXVZkmuS3J3kE9kgV2t3KQYAgIHsFgQAGEhcAQAMJK4AAAYSVwAAA4krAICBxBWwKVTVjqr69Oz7nVX167Pvn1VVG+LChcBycJ0rYNPp7l1Jds0Wn5Xk9iT/bWEDARuKLVfA0quq11bV/6qqj1bVO6vq56vqqqraOXv95Nltke7dQvXnVXXN7Ov/2yo121r1nqrakeRfJjm/qq6tqh+sqi9U1YNn73vI/ssA87DlClhqVfWUrN3u6IlZ+zfrmiSHunn3zUme093fqqrTsnYrk5339cbu/mJV/XaS27v7otn6rkrygiT/Zbbey2f3PQOYiy1XwLL7wSTv7u5vdPdtSf7kMO9/cJLfqapPJfnDJE94gOt7c/7mVjPnJrnkAf55YJOz5QpYVXfnb35BPHa/589P8pUk3zd7/VsP5Id298dmuxaflWRLd7tJNfCA2HIFLLs/S/IjVXVcVZ2Q5IWz57+Y5Cmz71+63/sfmuSm7r4nyTlJthzm5389yQkHPfd7Sd4RW62AIyCugKXW3dck+c9JPpnkfUk+PnvpoiQ/XVWfSHLyfn/kPyb5Z1X1ySSPT3LHYVZxRZJ/cu8B7bPnLk1yUtaO1wJ4QKq7Fz0DwNyq6pey3wHoE63jpUle3N3nTLUOYONyzBXAfqrqTUmel+T5i54FWE22XAEADOSYKwCAgcQVAMBA4goAYCBxBQAwkLgCABhIXAEADPR/AabmevpQV/lsAAAAAElFTkSuQmCC" width="324" height="324">


Here we see that fixed acidity does not give any specification to classify the quality
Similarly barplots are plotted for various input attributes

```markdown
#Here we see that the volatile acidity decreases as the quality of
wine increases
fig = plt.figure(figsize = (10,6))
sns.barplot(x = 'quality', y = 'volatile acidity', data = df)

#Citric acid has an increasing trend with qulaity
fig = plt.figure(figsize = (10,6))
sns.barplot(x = 'quality', y = 'citric acid', data = df)

#Not much variation
fig = plt.figure(figsize = (10,6))
sns.barplot(x = 'quality', y = 'residual sugar', data = df)
```

In the dataset that we have chosen, quality of wine is given using a number between 1 and 10. Now we plot a histogram to check out the count of each quality of wine
```markdown
fig = px.histogram(df,x='quality')
fig.show()
```
![Image](https://towardsdatascience.com/predicting-wine-quality-with-several-classification-techniques-179038ea6434)
For the output of histogram check the jupyter notebook in [github]()

### Preprocessing the data
Now that sufficient data analysis has been done, some data preprocessing has to be done. Since its a classification problem we take another attribute named good quality that has only 2 states (i.e. 1 for good quality an 0 for bad quality). Wine is considered to be of good quality if 'x' value is greater than or equal to 7. Otherwise it is considered to be of bad quality.

```markdown
df['goodquality'] = [1 if x >= 7 else 0 for x in df['quality']]
#Checking the proportion
df['goodquality'].value_counts()
```
Output:

0  -  1382

1  -  217


Next we separate feature variables and target variable
```markdown
X = df.drop(['quality','goodquality'], axis = 1)
y = df['goodquality']
```

Normalizing the feature variables is a useful technique to transform attributes with a Gaussian distribution and differing means and standard deviations to a standard Gaussian distribution with a mean of 0 and a standard deviation of 1.We can standardize data using scikit-learn with the StandardScaler class.
```markdown
from sklearn.preprocessing import StandardScaler
#X_features = X
X = StandardScaler().fit_transform(X)
```

Then we split the dataset wherein 70% of data is used to train the model and the remaining 30% of it is used for testing.
```markdown
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3, random_state=0)
```

### Performing ML algorithm
### 1)Logistic regression:
In logistic regression,the dependent variable is binary in nature having data coded as either 1 (stands for success/yes) or 0 (stands for failure/no).
```markdown
from sklearn.linear_model import LogisticRegression
logmodel = LogisticRegression()
logmodel.fit(X_train,y_train)
predictions = logmodel.predict(X_test)

from sklearn.metrics import confusion_matrix,classification_report,accuracy_score
print(confusion_matrix(y_test,predictions))
print(accuracy_score(y_test,predictions))
print(classification_report(y_test,predictions))
```
OUTPUT:

Confusion matrix:
```markdown
[[412  18]
[ 31  19]]
```
 
 Accuracy score:
`0.8979166666666667`
 
Classification report:
```markdown
                precision    recall  f1-score   support

           0       0.93      0.96      0.94       430
           1       0.51      0.38      0.44        50

    accuracy                           0.90       480                                                    `    
   macro avg       0.72      0.67      0.69       480
weighted avg       0.89      0.90      0.89       480
```

### 2) SVM
A support vector machine (SVM) is a supervised machine learning model that uses classification algorithms for two-group classification problems. After giving an SVM model sets of labeled training data for each category, they're able to categorize new text. So you're working on a text classification problem.

```markdown
from sklearn.svm import SVC
model = SVC()
model.fit(X_train,y_train)
predictions = model.predict(X_test)

from sklearn.metrics import confusion_matrix,classification_report,accuracy_score
print(confusion_matrix(y_test,predictions))
print(accuracy_score(y_test,predictions))
print(classification_report(y_test,predictions))
```
OUTPUT:

Confusion matrix:
```markdown
  TP   FP
[[419  11]
[ 32  18]]
  FN   TN
```
 
 Classification report:
 ```markdown
              precision    recall  f1-score   support

           0       0.93      0.97      0.95       430
           1       0.62      0.36      0.46        50

    accuracy                           0.91       480
   macro avg       0.77      0.67      0.70       480
weighted avg       0.90      0.91      0.90       480
```

Let's try to increase the accuracy by finding the best parameters for our SVC model
```markdown
from sklearn.model_selection import GridSearchCV

param = {
    'C': [0.1,0.8,0.9,1,1.1,1.2,1.3,1.4],
    'kernel':['linear', 'rbf'],
    'gamma' :[0.1,0.8,0.9,1,1.1,1.2,1.3,1.4]
}
grid_svc = GridSearchCV(SVC(), param_grid=param,refit=True,verbose=3)
```

```markdown
grid_svc.fit(X_train, y_train)
#Best parameters for our svc model
grid_svc.best_params_
```

Output:
```markdown
{'C': 1.4, 'gamma': 0.8, 'kernel': 'rbf'}
```

```markdown
grid_pred=grid_svc.predict(X_test)
print(confusion_matrix(y_test,grid_pred))
print(accuracy_score(y_test,grid_pred))
print(classification_report(y_test,grid_pred))
```

OUTPUT:
Confusion matrix:
```markdown
[[421   9]
 [ 27  23]]
```
 
Accuracy score:
`0.925`
 
 Classification report:
```markdown
              precision    recall  f1-score   support

           0       0.94      0.98      0.96       430
           1       0.72      0.46      0.56        50

    accuracy                           0.93       480
   macro avg       0.83      0.72      0.76       480
weighted avg       0.92      0.93      0.92       480
```

### 3) Random forest
Random forest is a supervised learning algorithm. The "forest" it builds, is an ensemble of decision trees, usually trained with the “bagging” method. The general idea of the bagging method is that a combination of learning models increases the overall result.

```markdown
from sklearn.ensemble import RandomForestClassifier
model2 = RandomForestClassifier(random_state=1)
model2.fit(X_train, y_train)
y_pred2 = model2.predict(X_test)
print(confusion_matrix(y_test,y_pred2))
print(accuracy_score(y_test,y_pred2))
print(classification_report(y_test, y_pred2))
```
OUTPUT:

Confusion matrix:
```markdown
[[414  16]
[ 21  29]]
```
Accuracy score:
`0.9229166666666667`

Classification report:
```markdown
                precision    recall  f1-score   support

           0       0.95      0.96      0.96       430
           1       0.64      0.58      0.61        50

    accuracy                           0.92       480
   macro avg       0.80      0.77      0.78       480
weighted avg       0.92      0.92      0.92       480
```
