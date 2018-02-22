---
layout: single
permalink: proj2/
title: &title "Project 2"
author_profile: true
---
<img src="https://i.imgur.com/RDye8eA.png" style="float: left; margin: 15px; height: 80px">

### Regression and Classification with the Ames Housing Data
---

<img src="/assets/images/proj2.jpg" width="100%">

This project uses the [Ames housing data recently made available on kaggle](https://www.kaggle.com/c/house-prices-advanced-regression-techniques).  

Data Science often involves modelling and prediction based on a dataset. In this project, techniques such as regression and classification are explored. Python packages used for this dataset include: 
1. numpy
2. pandas
3. matplotlib
4. seaborn
5. scikit-learn
6. imb-learn

## <img src="https://i.imgur.com/wLPdKgZ.png" style="float: left; height: 25px"> 1. Estimating the value of homes from fixed characteristics


### 1.1 Overview of dataset using pandas .describe()


```python
house_res.describe(include='all')
```

<iframe src="https://jerylcodes.github.io/proj2/table1/" height="400" width="600" overflow="auto"></iframe> 

### 1.2 Fixed characteristics as observed from the data description file:

<iframe src="https://jerylcodes.github.io/proj2/table2/" height="400" width="600" overflow="auto"></iframe> 

### 1.3 Feature engineering
A good practice before performing any modelling or classification is to examine your dataset and look for salient features that may be aggregated or perform any factorisation or binarisation (one-hot encoding) for qualtitative data. A brief feature engineering workflow may be as follows:

1. Sum columns that can be aggregated
2. Binarise columns
3. Drop columns with low variance
4. Get dummies for categorical columns  

A good feature engineering step that you may consider is to remove quantitative data columns with near 0 variance. This illustrates that that particular column has minimal impact to your prediction or classification. Here is a sample python code used to show columns with near 0 variance.


```python
# near zero variance

def nearZeroVariance(X, freqCut = 95 / 5, uniqueCut = 10):
    '''
    Determine predictors with near zero or zero variance.
    Inputs:
    X: pandas data frame
    freqCut: the cutoff for the ratio of the most common value to the second most common value
    uniqueCut: the cutoff for the percentage of distinct values out of the number of total samples
    Returns a tuple containing a list of column names: (zeroVar, nzVar)
    '''

    colNames = X.columns.values.tolist()
    freqRatio = dict()
    uniquePct = dict()

    for names in colNames:
        counts = (
            (X[names])
            .value_counts()
            .sort_values(ascending = False)
            .values
            )

        if len(counts) == 1:
            freqRatio[names] = -1
            uniquePct[names] = (len(counts) / len(X[names])) * 100
            continue

        freqRatio[names] = counts[0] / counts[1]
        uniquePct[names] = (len(counts) / len(X[names])) * 100

    zeroVar = list()
    nzVar = list()
    for k in uniquePct.keys():
        if freqRatio[k] == -1:
            zeroVar.append(k)

        if uniquePct[k] < uniqueCut and freqRatio[k] > freqCut:
            nzVar.append(k)

    return(zeroVar, nzVar)
```


```python
zerovartest = house_res1.loc[:,['LotFrontage','LotArea','MasVnrArea', 'TotalBsmtSF','GrLivArea',\
                                'GarageArea','WoodDeckSF', 'PoolArea', 'MiscVal']]
nearZeroVariance(zerovartest)
```




    ([], ['MasVnrArea', 'PoolArea', 'MiscVal'])


## <img src="https://i.imgur.com/wLPdKgZ.png" style="float: left; height: 25px"> 2. Preparing your dataset for prediction/classification

### 2.1 Train-test split and normalisation
Before running your model, it is important to split your data into train and test datasets. It is also important to normalise any data columns whenever necessary. In our case, we may use scikit-learn's __train-test split__ module to split our dataset; and its __StandardScaler__ module to normalise our data.


```python
# Lasso regression  
lasso = Lasso(alpha=optimal_lasso.alpha_)

lasso_scores = cross_val_score(lasso, Xs, y, cv=10)

print lasso_scores
print np.mean(lasso_scores)
```

    [ 0.88388639  0.8378792   0.83228373  0.73855977  0.7982018   0.7768981
      0.82231355  0.77187934  0.51414779  0.83227006]
    0.780831973138
    

## <img src="https://i.imgur.com/wLPdKgZ.png" style="float: left; height: 25px"> 3. Model for regression

### 3.1 Lasso regression to predict house prices
The lasso regression applies regularisation to data columns such that certain columns may not be as information to predict your variable of interest. This is particularly useful for datasets with large amounts of qualitative features.


```python
lasso.fit(Xs, y)
```




    Lasso(alpha=1098.3164314643716, copy_X=True, fit_intercept=True,
       max_iter=1000, normalize=False, positive=False, precompute=False,
       random_state=None, selection='cyclic', tol=0.0001, warm_start=False)




```python
# top 20 features after lasso
lasso_coefs = pd.DataFrame({'variable':X.columns,
                            'coef':lasso.coef_,
                            'abs_coef':np.abs(lasso.coef_)})

lasso_coefs.sort_values('abs_coef', inplace=True, ascending=False)

lasso_coefs.head(20)
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>abs_coef</th>
      <th>coef</th>
      <th>variable</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>3</th>
      <td>27801.501124</td>
      <td>27801.501124</td>
      <td>GrLivArea</td>
    </tr>
    <tr>
      <th>52</th>
      <td>13227.923817</td>
      <td>13227.923817</td>
      <td>Neighborhood_NridgHt</td>
    </tr>
    <tr>
      <th>93</th>
      <td>13146.851630</td>
      <td>13146.851630</td>
      <td>GarageCars_3</td>
    </tr>
    <tr>
      <th>2</th>
      <td>9522.574031</td>
      <td>9522.574031</td>
      <td>TotalBsmtSF</td>
    </tr>
    <tr>
      <th>58</th>
      <td>7605.301512</td>
      <td>7605.301512</td>
      <td>Neighborhood_StoneBr</td>
    </tr>
    <tr>
      <th>51</th>
      <td>7503.234427</td>
      <td>7503.234427</td>
      <td>Neighborhood_NoRidge</td>
    </tr>
    <tr>
      <th>57</th>
      <td>6852.984978</td>
      <td>6852.984978</td>
      <td>Neighborhood_Somerst</td>
    </tr>
    <tr>
      <th>7</th>
      <td>5878.558372</td>
      <td>-5878.558372</td>
      <td>BsmtFullBath_0</td>
    </tr>
    <tr>
      <th>19</th>
      <td>5098.838853</td>
      <td>-5098.838853</td>
      <td>MSSubClass_90</td>
    </tr>
    <tr>
      <th>82</th>
      <td>4719.729393</td>
      <td>4719.729393</td>
      <td>TotRmsAbvGrd_10</td>
    </tr>
    <tr>
      <th>53</th>
      <td>4426.366451</td>
      <td>-4426.366451</td>
      <td>Neighborhood_OldTown</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4359.232517</td>
      <td>4359.232517</td>
      <td>GarageArea</td>
    </tr>
    <tr>
      <th>72</th>
      <td>4212.180965</td>
      <td>-4212.180965</td>
      <td>BedroomAbvGr_5</td>
    </tr>
    <tr>
      <th>43</th>
      <td>4058.672481</td>
      <td>-4058.672481</td>
      <td>Neighborhood_Edwards</td>
    </tr>
    <tr>
      <th>8</th>
      <td>3902.417073</td>
      <td>-3902.417073</td>
      <td>fullbath2&lt;</td>
    </tr>
    <tr>
      <th>21</th>
      <td>3650.970712</td>
      <td>-3650.970712</td>
      <td>MSSubClass_160</td>
    </tr>
    <tr>
      <th>5</th>
      <td>3590.483741</td>
      <td>3590.483741</td>
      <td>WoodDeckSF</td>
    </tr>
    <tr>
      <th>1</th>
      <td>3211.026312</td>
      <td>3211.026312</td>
      <td>LotArea</td>
    </tr>
    <tr>
      <th>41</th>
      <td>2830.346250</td>
      <td>2830.346250</td>
      <td>Neighborhood_CollgCr</td>
    </tr>
    <tr>
      <th>9</th>
      <td>2691.396233</td>
      <td>-2691.396233</td>
      <td>halfbath_0</td>
    </tr>
  </tbody>
</table>
</div>

## <img src="https://i.imgur.com/wLPdKgZ.png" style="float: left; height: 25px"> 4. Classify records into abnormal or normal sale

### 4.1 Caveat: Imbalanced dataset
In some cases, your dataset might present itself to be imbalanced. This has large implications towards the building of our models. In the case of classification, an over-representation of a particular class may skew the classification towards the majority class. To mitigate this problem, it is advisable to perform certain sampling techniques in order to balance out the classes.  

For this dataset, we will be exploring 2 imbalanced dataset sampling techniques: 
1. SMOTE - Synthetic Minority Oversampling TEchnique
2. Combination of SMOTE and TOMEK - tomek link undersampling

### 4.1.1 SMOTE

Synthetic minority oversampling technique
- For each minority point, compute nearest neighbours
- draw line to nearest neighbours
- synthetically add a new point as minority

### 4.1.2 Combination, SMOTE and tomek


Tomek link (undersampling)
- a pair of samples is considered tomek link if they are nearest neighbour and of differing class
- the majority class of the tomek link is then removed (under sample)

### 4.2 Logistic Regression Classification
We will next use logistic regression to classify our dataset into abnormal or normal housing sale.


```python
param = {'penalty':['l1','l2'] ,\
         'C': list(np.linspace(0.1,1,num=10))}
```


```python
clf = GridSearchCV(LogisticRegression(),param, cv=5)
clf.fit(X_res,y_res)
```




    GridSearchCV(cv=5, error_score='raise',
           estimator=LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
              intercept_scaling=1, max_iter=100, multi_class='ovr', n_jobs=1,
              penalty='l2', random_state=None, solver='liblinear', tol=0.0001,
              verbose=0, warm_start=False),
           fit_params={}, iid=True, n_jobs=1,
           param_grid={'penalty': ['l1', 'l2'], 'C': [0.10000000000000001, 0.20000000000000001, 0.30000000000000004, 0.40000000000000002, 0.5, 0.59999999999999998, 0.70000000000000007, 0.80000000000000004, 0.90000000000000002, 1.0]},
           pre_dispatch='2*n_jobs', refit=True, scoring=None, verbose=0)



### 4.3 Logistic Regression after SMOTE
#### Classification report


```python
from imblearn.metrics import classification_report_imbalanced
print classification_report_imbalanced(y_test, y_pred,target_names=['normal','abnormal'])
```

                       pre       rec       spe        f1       geo       iba       sup
    
         normal       0.94      0.79      0.40      0.86      0.35      0.13       444
       abnormal       0.13      0.40      0.79      0.20      0.35      0.11        35
    
    avg / total       0.88      0.76      0.43      0.81      0.35      0.13       479
    
    

### 4.4 Logistic Regression after SMOTE + TOMEK 
#### Classification report


```python
print classification_report_imbalanced(y_test, y_pred,target_names=['normal','abnormal'])
```

                       pre       rec       spe        f1       geo       iba       sup
    
         normal       0.95      0.77      0.49      0.85      0.37      0.15       444
       abnormal       0.14      0.49      0.77      0.22      0.37      0.12        35
    
    avg / total       0.89      0.75      0.51      0.80      0.37      0.14       479
    
    
