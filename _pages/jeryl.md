---
permalink: /project 01/
title: "Project 01"
ads: false
share: false
---


<img src="https://i.imgur.com/u1otXPC.png" style="float: left; margin: 15px; height: 80px">

# Project 1

### Exploratory Data Analysis (EDA)

---
### Problem statement  

Your hometown mayor just created a new data analysis team to give policy advice, and the administration recruited _you_ via LinkedIn to join it. Unfortunately, due to budget constraints, for now the "team" is just you...

The mayor wants to start a new initiative to move the needle on one of two separate issues: high school education outcomes, or drug abuse in the community.

Also unfortunately, that is the entirety of what you've been told. And the mayor just went on a lobbyist-funded fact-finding trip in the Bahamas. In the meantime, you got your hands on two national datasets: one on SAT scores by state, and one on drug use by age. Start exploring these to look for useful patterns and possible hypotheses!

---

This project is focused on exploratory data analysis, aka "EDA". EDA is an essential part of the data science analysis pipeline. Failure to perform EDA before modeling is almost guaranteed to lead to bad models and faulty conclusions. What you do in this project are good practices for all projects going forward, especially those after this bootcamp!

### Import packages


```python
import numpy as np
import scipy.stats as stats
import csv
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

import seaborn as sns

# this line tells jupyter notebook to put the plots in the notebook rather than saving them to file.
%matplotlib inline

# this line makes plots prettier on mac retina screens. If you don't have one it shouldn't do anything.
%config InlineBackend.figure_format = 'retina'
```

<img src="https://i.imgur.com/wLPdKgZ.png" style="float: left; margin: 25px 15px 0px 0px; height: 25px">

## 1. Load the `sat_scores.csv` dataset and describe it

---
### 1.1 Load the file with the `csv` module and put it in a Python dictionary


```python
satcsv = './sat_scores.csv'
drugcsv = './drug-use-by-age.csv'

datasat = pd.read_csv(satcsv)
```


```python
print datasat.info()
datasat.head()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 52 entries, 0 to 51
    Data columns (total 4 columns):
    State     52 non-null object
    Rate      52 non-null int64
    Verbal    52 non-null int64
    Math      52 non-null int64
    dtypes: int64(3), object(1)
    memory usage: 1.7+ KB
    None
    




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
      <th>State</th>
      <th>Rate</th>
      <th>Verbal</th>
      <th>Math</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>CT</td>
      <td>82</td>
      <td>509</td>
      <td>510</td>
    </tr>
    <tr>
      <th>1</th>
      <td>NJ</td>
      <td>81</td>
      <td>499</td>
      <td>513</td>
    </tr>
    <tr>
      <th>2</th>
      <td>MA</td>
      <td>79</td>
      <td>511</td>
      <td>515</td>
    </tr>
    <tr>
      <th>3</th>
      <td>NY</td>
      <td>77</td>
      <td>495</td>
      <td>505</td>
    </tr>
    <tr>
      <th>4</th>
      <td>NH</td>
      <td>72</td>
      <td>520</td>
      <td>516</td>
    </tr>
  </tbody>
</table>
</div>




```python
datasat.describe()
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
      <th>Rate</th>
      <th>Verbal</th>
      <th>Math</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>52.000000</td>
      <td>52.000000</td>
      <td>52.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>37.153846</td>
      <td>532.019231</td>
      <td>531.500000</td>
    </tr>
    <tr>
      <th>std</th>
      <td>27.301788</td>
      <td>33.236225</td>
      <td>36.014975</td>
    </tr>
    <tr>
      <th>min</th>
      <td>4.000000</td>
      <td>482.000000</td>
      <td>439.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>9.000000</td>
      <td>501.000000</td>
      <td>504.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>33.500000</td>
      <td>526.500000</td>
      <td>521.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>63.500000</td>
      <td>562.000000</td>
      <td>555.750000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>82.000000</td>
      <td>593.000000</td>
      <td>603.000000</td>
    </tr>
  </tbody>
</table>
</div>



### 1.2 Make a pandas DataFrame object with the SAT dictionary, and another with the pandas `.read_csv()` function


```python
sat_dict = {'state': ['CT', 'NJ','MA','NY','NH'],
           'rate': [82,81,79,77,72],
           'verbal':[509,499,511,495,520],
           'math':[510,513,515,505,516]}
```


```python
sat_pd= pd.DataFrame(data=sat_dict)
sat_pd.dtypes
```




    math       int64
    rate       int64
    state     object
    verbal     int64
    dtype: object




```python
datasat.dtypes
```




    State     object
    Rate       int64
    Verbal     int64
    Math       int64
    dtype: object



### 1.3 Look at the first ten rows of the DataFrame: what does our data describe?

From now on, use the DataFrame loaded from the file using the `.read_csv()` function.

Use the `.head(num)` built-in DataFrame function, where `num` is the number of rows to print out.

You are not given a "codebook" with this data, so you will have to make some (very minor) inference.


```python
datasat.head(10)
datasat.drop(51, axis=0, inplace=True)
```

### 1.4 Describing the data
The sat scores contain 4 columns of information : 
1. State: Refers to states in the U.S.
2. Rate: A percentage possibly referring to SAT passing rate of children
3. Verbal: Verbal Scores in the range of 482 - 593
4. Math: Math Scores in the range of 439 - 603

<img src="https://i.imgur.com/wLPdKgZ.png" style="float: left; margin: 25px 15px 0px 0px; height: 25px">

## 2. Create a "data dictionary" based on the data

---


```python
# Create a data dictionary using pandas dataframe
sat_ddict = {'column':datasat.columns.tolist(),
            'description':['States of the U.S.','Passing rate of SAT','Verbal SAT scores','MATH SAT scores'],
            'data type':['string','integer','integer','integer']}
sat_ddf = pd.DataFrame(sat_ddict)

# Print the shape and data dictionary.
print 'Shape of SAT dataset:', datasat.shape
sat_ddf
```

    Shape of SAT dataset: (51, 4)
    




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
      <th>column</th>
      <th>data type</th>
      <th>description</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>State</td>
      <td>string</td>
      <td>States of the U.S.</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Rate</td>
      <td>integer</td>
      <td>Passing rate of SAT</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Verbal</td>
      <td>integer</td>
      <td>Verbal SAT scores</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Math</td>
      <td>integer</td>
      <td>MATH SAT scores</td>
    </tr>
  </tbody>
</table>
</div>



<img src="https://i.imgur.com/wLPdKgZ.png" style="float: left; margin: 25px 15px 0px 0px; height: 25px">

## 3. Plot the data using seaborn

---

### 3.1 Using seaborn's `distplot`, plot the distributions for each of `Rate`, `Math`, and `Verbal`


```python
# Plot rate distplot

ratep = sns.distplot(datasat['Rate'], kde=False, bins=10)
ratep.set(xlabel='SAT passing rate')
ratep.set_title('Frequency distribution of SAT passing rate')

```




    Text(0.5,1,u'Frequency distribution of SAT passing rate')




![png](output_16_1.png)



```python
# Plot math distplot

mathp = sns.distplot(datasat['Math'], kde=False,bins=20, color='blue')
mathp.set(xlabel='SAT Math Scores')
mathp.set_title('Frequency distribution of SAT Math Scores')

```




    Text(0.5,1,u'Frequency distribution of SAT Math Scores')




![png](output_17_1.png)



```python
# Plot verbal distplot

verbp = sns.distplot(datasat['Verbal'], kde=False,bins=20, color='orange')
verbp.set(xlabel='SAT Verbal Scores')
verbp.set_title('Frequency distribution of SAT Verbal Scores')

```




    Text(0.5,1,u'Frequency distribution of SAT Verbal Scores')




![png](output_18_1.png)


### 3.2 Using seaborn's `pairplot`, show the joint distributions for each of `Rate`, `Math`, and `Verbal`


```python
sns.pairplot(datasat, vars=['Rate', 'Math', 'Verbal'], hue='State')
```




    <seaborn.axisgrid.PairGrid at 0x454d09e8>




![png](output_20_1.png)


## Intepretation of pair plot
**1. Rate seems to be negatively correlated with both math and verbal scores.**  
From plots 2 and 3 in the first row, as scores increase, rate decreases.

**2. Math and verbal appears to be positively correlated.**  
From plots (row 2, plot 3 and row 3, plot 2), as either variable increases, the other seems to increase as well.

**3. Adding hue dimension (assigned to state) may yield interesting information about the clustering of states according to SAT scores.**

<img src="https://i.imgur.com/wLPdKgZ.png" style="float: left; margin: 25px 15px 0px 0px; height: 25px">

## 4. Plot the data using built-in pandas functions.

---

### 4.1 Plot a stacked histogram with `Verbal` and `Math` using pandas


```python
datasat[['Verbal','Math']].plot.hist(stacked=True, alpha=0.5,figsize=(12,12), bins=20)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x4d3185f8>




![png](output_23_1.png)


### 4.2 Plot `Verbal` and `Math` on the same chart using boxplots


```python
databox = datasat.sort_values(by=['Verbal','Math']).plot(kind='box', y=['Verbal','Math'], figsize=(10,10),title='SAT score distribution')
databox.set_ylabel('Score')
databox.set_xlabel('Index')
```




    Text(0.5,0,u'Index')




![png](output_25_1.png)


** Advantages of box plot **

Able to visualise spread of a feature as well as outliers quickly. 

** Problems with plotting rate in the same box plot: **

As the feature 'rate' has a smaller range of values compared to Verbal and Math, it will be unwise to plot rate on the same box plot as it may affect the visualisation of the data negatively.

<img src="https://i.imgur.com/wLPdKgZ.png" style="float: left; margin: 25px 15px 0px 0px; height: 25px">

### 4.3 Plot `Verbal`, `Math`, and `Rate` appropriately on the same boxplot chart


```python
# divide each feature by its own mean, to normalise all features
datanorm = datasat.copy()
print datanorm.mean()

datanorm['Raten'] = datanorm['Rate'].map(lambda x: (x-37.153846)/np.std(datanorm['Rate']))
datanorm['Verbaln'] = datanorm['Verbal'].map(lambda x: (x-532.019231)/np.std(datanorm['Verbal']))
datanorm['Mathn'] = datanorm['Math'].map(lambda x: (x-531.5)/np.std(datanorm['Math']))
```

    Rate       37.000000
    Verbal    532.529412
    Math      531.843137
    dtype: float64
    


```python
datanormp = datanorm.sort_values(by=['Verbaln','Mathn']).plot(kind='box', y=['Verbaln','Mathn','Raten'], figsize=(10,10),title='SAT score distribution')
```


![png](output_29_0.png)


### Justification

By dividing each feature by its own mean, it transforms each feature to be on the same scale (i.e., with a mean of 1 each). With this transformation, comparisons between the spread of data among features can be more intuitive.

<img src="https://i.imgur.com/wLPdKgZ.png" style="float: left; margin: 25px 15px 0px 0px; height: 25px">

## 5. Create and examine subsets of the data

---
### 5.1 Find the list of states that have `Verbal` scores greater than the average of `Verbal` scores across states

How many states are above the mean? What does this tell you about the distribution of `Verbal` scores?


```python
verb_average = datasat['Verbal'].mean()
print 'verbal mean is ', verb_average
datasat.loc[datasat['Verbal']>verb_average]
```

    verbal mean is  532.529411765
    




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
      <th>State</th>
      <th>Rate</th>
      <th>Verbal</th>
      <th>Math</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>26</th>
      <td>CO</td>
      <td>31</td>
      <td>539</td>
      <td>542</td>
    </tr>
    <tr>
      <th>27</th>
      <td>OH</td>
      <td>26</td>
      <td>534</td>
      <td>439</td>
    </tr>
    <tr>
      <th>28</th>
      <td>MT</td>
      <td>23</td>
      <td>539</td>
      <td>539</td>
    </tr>
    <tr>
      <th>30</th>
      <td>ID</td>
      <td>17</td>
      <td>543</td>
      <td>542</td>
    </tr>
    <tr>
      <th>31</th>
      <td>TN</td>
      <td>13</td>
      <td>562</td>
      <td>553</td>
    </tr>
    <tr>
      <th>32</th>
      <td>NM</td>
      <td>13</td>
      <td>551</td>
      <td>542</td>
    </tr>
    <tr>
      <th>33</th>
      <td>IL</td>
      <td>12</td>
      <td>576</td>
      <td>589</td>
    </tr>
    <tr>
      <th>34</th>
      <td>KY</td>
      <td>12</td>
      <td>550</td>
      <td>550</td>
    </tr>
    <tr>
      <th>35</th>
      <td>WY</td>
      <td>11</td>
      <td>547</td>
      <td>545</td>
    </tr>
    <tr>
      <th>36</th>
      <td>MI</td>
      <td>11</td>
      <td>561</td>
      <td>572</td>
    </tr>
    <tr>
      <th>37</th>
      <td>MN</td>
      <td>9</td>
      <td>580</td>
      <td>589</td>
    </tr>
    <tr>
      <th>38</th>
      <td>KS</td>
      <td>9</td>
      <td>577</td>
      <td>580</td>
    </tr>
    <tr>
      <th>39</th>
      <td>AL</td>
      <td>9</td>
      <td>559</td>
      <td>554</td>
    </tr>
    <tr>
      <th>40</th>
      <td>NE</td>
      <td>8</td>
      <td>562</td>
      <td>568</td>
    </tr>
    <tr>
      <th>41</th>
      <td>OK</td>
      <td>8</td>
      <td>567</td>
      <td>561</td>
    </tr>
    <tr>
      <th>42</th>
      <td>MO</td>
      <td>8</td>
      <td>577</td>
      <td>577</td>
    </tr>
    <tr>
      <th>43</th>
      <td>LA</td>
      <td>7</td>
      <td>564</td>
      <td>562</td>
    </tr>
    <tr>
      <th>44</th>
      <td>WI</td>
      <td>6</td>
      <td>584</td>
      <td>596</td>
    </tr>
    <tr>
      <th>45</th>
      <td>AR</td>
      <td>6</td>
      <td>562</td>
      <td>550</td>
    </tr>
    <tr>
      <th>46</th>
      <td>UT</td>
      <td>5</td>
      <td>575</td>
      <td>570</td>
    </tr>
    <tr>
      <th>47</th>
      <td>IA</td>
      <td>5</td>
      <td>593</td>
      <td>603</td>
    </tr>
    <tr>
      <th>48</th>
      <td>SD</td>
      <td>4</td>
      <td>577</td>
      <td>582</td>
    </tr>
    <tr>
      <th>49</th>
      <td>ND</td>
      <td>4</td>
      <td>592</td>
      <td>599</td>
    </tr>
    <tr>
      <th>50</th>
      <td>MS</td>
      <td>4</td>
      <td>566</td>
      <td>551</td>
    </tr>
  </tbody>
</table>
</div>



### 5.2 Find the list of states that have `Verbal` scores greater than the median of `Verbal` scores across states


```python
verb_med = datasat['Verbal'].median()
print "verbal median is ", verb_med
datasat.loc[datasat['Verbal']>verb_med]
```

    verbal median is  527.0
    




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
      <th>State</th>
      <th>Rate</th>
      <th>Verbal</th>
      <th>Math</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>26</th>
      <td>CO</td>
      <td>31</td>
      <td>539</td>
      <td>542</td>
    </tr>
    <tr>
      <th>27</th>
      <td>OH</td>
      <td>26</td>
      <td>534</td>
      <td>439</td>
    </tr>
    <tr>
      <th>28</th>
      <td>MT</td>
      <td>23</td>
      <td>539</td>
      <td>539</td>
    </tr>
    <tr>
      <th>30</th>
      <td>ID</td>
      <td>17</td>
      <td>543</td>
      <td>542</td>
    </tr>
    <tr>
      <th>31</th>
      <td>TN</td>
      <td>13</td>
      <td>562</td>
      <td>553</td>
    </tr>
    <tr>
      <th>32</th>
      <td>NM</td>
      <td>13</td>
      <td>551</td>
      <td>542</td>
    </tr>
    <tr>
      <th>33</th>
      <td>IL</td>
      <td>12</td>
      <td>576</td>
      <td>589</td>
    </tr>
    <tr>
      <th>34</th>
      <td>KY</td>
      <td>12</td>
      <td>550</td>
      <td>550</td>
    </tr>
    <tr>
      <th>35</th>
      <td>WY</td>
      <td>11</td>
      <td>547</td>
      <td>545</td>
    </tr>
    <tr>
      <th>36</th>
      <td>MI</td>
      <td>11</td>
      <td>561</td>
      <td>572</td>
    </tr>
    <tr>
      <th>37</th>
      <td>MN</td>
      <td>9</td>
      <td>580</td>
      <td>589</td>
    </tr>
    <tr>
      <th>38</th>
      <td>KS</td>
      <td>9</td>
      <td>577</td>
      <td>580</td>
    </tr>
    <tr>
      <th>39</th>
      <td>AL</td>
      <td>9</td>
      <td>559</td>
      <td>554</td>
    </tr>
    <tr>
      <th>40</th>
      <td>NE</td>
      <td>8</td>
      <td>562</td>
      <td>568</td>
    </tr>
    <tr>
      <th>41</th>
      <td>OK</td>
      <td>8</td>
      <td>567</td>
      <td>561</td>
    </tr>
    <tr>
      <th>42</th>
      <td>MO</td>
      <td>8</td>
      <td>577</td>
      <td>577</td>
    </tr>
    <tr>
      <th>43</th>
      <td>LA</td>
      <td>7</td>
      <td>564</td>
      <td>562</td>
    </tr>
    <tr>
      <th>44</th>
      <td>WI</td>
      <td>6</td>
      <td>584</td>
      <td>596</td>
    </tr>
    <tr>
      <th>45</th>
      <td>AR</td>
      <td>6</td>
      <td>562</td>
      <td>550</td>
    </tr>
    <tr>
      <th>46</th>
      <td>UT</td>
      <td>5</td>
      <td>575</td>
      <td>570</td>
    </tr>
    <tr>
      <th>47</th>
      <td>IA</td>
      <td>5</td>
      <td>593</td>
      <td>603</td>
    </tr>
    <tr>
      <th>48</th>
      <td>SD</td>
      <td>4</td>
      <td>577</td>
      <td>582</td>
    </tr>
    <tr>
      <th>49</th>
      <td>ND</td>
      <td>4</td>
      <td>592</td>
      <td>599</td>
    </tr>
    <tr>
      <th>50</th>
      <td>MS</td>
      <td>4</td>
      <td>566</td>
      <td>551</td>
    </tr>
  </tbody>
</table>
</div>



### Comparison between verbal > average and verbal > median

The number of states with verbal > average and verbal > median is 24 which is equal. Suggests that verbal has quite a symmetrical distribution.

### 5.3 Create a column that is the difference between the `Verbal` and `Math` scores


```python
datasat['ver_diff'] = datasat['Verbal'] - datasat['Math']

```

### 5.4 Create two new DataFrames showing states with the greatest difference between scores

1. Your first DataFrame should be the 10 states with the greatest gap between `Verbal` and `Math` scores where `Verbal` is greater than `Math`. It should be sorted appropriately to show the ranking of states.
2. Your second DataFrame will be the inverse: states with the greatest gap between `Verbal` and `Math` such that `Math` is greater than `Verbal`. Again, this should be sorted appropriately to show rank.
3. Print the header of both variables, only showing the top 3 states in each.


```python
dataasc = datasat.sort_values('ver_diff',ascending=False)
dataasc.head(3)
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
      <th>State</th>
      <th>Rate</th>
      <th>Verbal</th>
      <th>Math</th>
      <th>ver_diff</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>27</th>
      <td>OH</td>
      <td>26</td>
      <td>534</td>
      <td>439</td>
      <td>95</td>
    </tr>
    <tr>
      <th>50</th>
      <td>MS</td>
      <td>4</td>
      <td>566</td>
      <td>551</td>
      <td>15</td>
    </tr>
    <tr>
      <th>29</th>
      <td>WV</td>
      <td>18</td>
      <td>527</td>
      <td>512</td>
      <td>15</td>
    </tr>
  </tbody>
</table>
</div>




```python
datades = datasat.sort_values('ver_diff',ascending=True)
datades.head(3)
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
      <th>State</th>
      <th>Rate</th>
      <th>Verbal</th>
      <th>Math</th>
      <th>ver_diff</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>21</th>
      <td>HI</td>
      <td>52</td>
      <td>485</td>
      <td>515</td>
      <td>-30</td>
    </tr>
    <tr>
      <th>23</th>
      <td>CA</td>
      <td>51</td>
      <td>498</td>
      <td>517</td>
      <td>-19</td>
    </tr>
    <tr>
      <th>1</th>
      <td>NJ</td>
      <td>81</td>
      <td>499</td>
      <td>513</td>
      <td>-14</td>
    </tr>
  </tbody>
</table>
</div>




<img src="https://i.imgur.com/wLPdKgZ.png" style="float: left; margin: 25px 15px 0px 0px; height: 25px">
## 6. Examine summary statistics
---

Checking the summary statistics!


### 6.1 Create the correlation matrix of your variables (excluding `State`).


```python
datasat.corr()
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
      <th>Rate</th>
      <th>Verbal</th>
      <th>Math</th>
      <th>ver_diff</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Rate</th>
      <td>1.000000</td>
      <td>-0.888121</td>
      <td>-0.773419</td>
      <td>-0.098671</td>
    </tr>
    <tr>
      <th>Verbal</th>
      <td>-0.888121</td>
      <td>1.000000</td>
      <td>0.899909</td>
      <td>0.044527</td>
    </tr>
    <tr>
      <th>Math</th>
      <td>-0.773419</td>
      <td>0.899909</td>
      <td>1.000000</td>
      <td>-0.395574</td>
    </tr>
    <tr>
      <th>ver_diff</th>
      <td>-0.098671</td>
      <td>0.044527</td>
      <td>-0.395574</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>



Rate seems to have a negative correlation with the verbal and math variables.  
Verbal and math seems to be positively correlated.

### 6.2 Use pandas'  `.describe()` built-in function on your DataFrame


```python
datasat.describe()
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
      <th>Rate</th>
      <th>Verbal</th>
      <th>Math</th>
      <th>ver_diff</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>51.000000</td>
      <td>51.000000</td>
      <td>51.000000</td>
      <td>51.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>37.000000</td>
      <td>532.529412</td>
      <td>531.843137</td>
      <td>0.686275</td>
    </tr>
    <tr>
      <th>std</th>
      <td>27.550681</td>
      <td>33.360667</td>
      <td>36.287393</td>
      <td>15.839811</td>
    </tr>
    <tr>
      <th>min</th>
      <td>4.000000</td>
      <td>482.000000</td>
      <td>439.000000</td>
      <td>-30.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>9.000000</td>
      <td>501.000000</td>
      <td>503.000000</td>
      <td>-6.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>33.000000</td>
      <td>527.000000</td>
      <td>525.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>64.000000</td>
      <td>562.000000</td>
      <td>557.500000</td>
      <td>4.500000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>82.000000</td>
      <td>593.000000</td>
      <td>603.000000</td>
      <td>95.000000</td>
    </tr>
  </tbody>
</table>
</div>



<span style="color:blue">Write-up:</span>
1. Count
    * Number of values in this column
2. Mean
    * Average of all non-null values in column
3. Std
    * Standard deviation of values in column
4. Min
    * Smallest value in column
5. 25%, 50%, 75%
    * 25<sup>th</sup>, 50<sup>th</sup>, 75<sup>th</sup> Percentile of values in column
6. Max
    * Biggest value in column

### 6.3 Assign and print the _covariance_ matrix for the dataset

1. Describe how the covariance matrix is different from the correlation matrix.
2. What is the process to convert the covariance into the correlation?
3. Why is the correlation matrix preferred to the covariance matrix for examining relationships in your data?


```python
datasat.cov()
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
      <th>Rate</th>
      <th>Verbal</th>
      <th>Math</th>
      <th>ver_diff</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Rate</th>
      <td>759.04</td>
      <td>-816.280000</td>
      <td>-773.220000</td>
      <td>-43.060000</td>
    </tr>
    <tr>
      <th>Verbal</th>
      <td>-816.28</td>
      <td>1112.934118</td>
      <td>1089.404706</td>
      <td>23.529412</td>
    </tr>
    <tr>
      <th>Math</th>
      <td>-773.22</td>
      <td>1089.404706</td>
      <td>1316.774902</td>
      <td>-227.370196</td>
    </tr>
    <tr>
      <th>ver_diff</th>
      <td>-43.06</td>
      <td>23.529412</td>
      <td>-227.370196</td>
      <td>250.899608</td>
    </tr>
  </tbody>
</table>
</div>



<span style="color:blue">Answers to prompt questions:</span>
1. Covariance measures how 2 variables vary together while correlation measures how one variable is dependent on the other.
2. Correlation can be obtained by dividing the covariance with the product of standard deviation of the 2 variables.
3. Correlation will be able to tell us when a variable increases, whether the other increases or decreases.


<img src="https://i.imgur.com/wLPdKgZ.png" style="float: left; margin: 25px 15px 0px 0px; height: 25px">

## 7. Performing EDA on "drug use by age" data.

---

### 7.1


```python
df1 = pd.read_csv(drugcsv)

df1.head(1)
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
      <th>age</th>
      <th>n</th>
      <th>alcohol-use</th>
      <th>alcohol-frequency</th>
      <th>marijuana-use</th>
      <th>marijuana-frequency</th>
      <th>cocaine-use</th>
      <th>cocaine-frequency</th>
      <th>crack-use</th>
      <th>crack-frequency</th>
      <th>heroin-use</th>
      <th>heroin-frequency</th>
      <th>hallucinogen-use</th>
      <th>hallucinogen-frequency</th>
      <th>inhalant-use</th>
      <th>inhalant-frequency</th>
      <th>pain-releiver-use</th>
      <th>pain-releiver-frequency</th>
      <th>oxycontin-use</th>
      <th>oxycontin-frequency</th>
      <th>tranquilizer-use</th>
      <th>tranquilizer-frequency</th>
      <th>stimulant-use</th>
      <th>stimulant-frequency</th>
      <th>meth-use</th>
      <th>meth-frequency</th>
      <th>sedative-use</th>
      <th>sedative-frequency</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>12</td>
      <td>2798</td>
      <td>3.9</td>
      <td>3.0</td>
      <td>1.1</td>
      <td>4.0</td>
      <td>0.1</td>
      <td>5.0</td>
      <td>0.0</td>
      <td>-</td>
      <td>0.1</td>
      <td>35.5</td>
      <td>0.2</td>
      <td>52.0</td>
      <td>1.6</td>
      <td>19.0</td>
      <td>2.0</td>
      <td>36.0</td>
      <td>0.1</td>
      <td>24.5</td>
      <td>0.2</td>
      <td>52.0</td>
      <td>0.2</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>-</td>
      <td>0.2</td>
      <td>13.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
df1.isnull().sum()
```




    age                        0
    n                          0
    alcohol-use                0
    alcohol-frequency          0
    marijuana-use              0
    marijuana-frequency        0
    cocaine-use                0
    cocaine-frequency          0
    crack-use                  0
    crack-frequency            0
    heroin-use                 0
    heroin-frequency           0
    hallucinogen-use           0
    hallucinogen-frequency     0
    inhalant-use               0
    inhalant-frequency         0
    pain-releiver-use          0
    pain-releiver-frequency    0
    oxycontin-use              0
    oxycontin-frequency        0
    tranquilizer-use           0
    tranquilizer-frequency     0
    stimulant-use              0
    stimulant-frequency        0
    meth-use                   0
    meth-frequency             0
    sedative-use               0
    sedative-frequency         0
    dtype: int64




```python
df1.info()

```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 17 entries, 0 to 16
    Data columns (total 28 columns):
    age                        17 non-null object
    n                          17 non-null int64
    alcohol-use                17 non-null float64
    alcohol-frequency          17 non-null float64
    marijuana-use              17 non-null float64
    marijuana-frequency        17 non-null float64
    cocaine-use                17 non-null float64
    cocaine-frequency          17 non-null object
    crack-use                  17 non-null float64
    crack-frequency            17 non-null object
    heroin-use                 17 non-null float64
    heroin-frequency           17 non-null object
    hallucinogen-use           17 non-null float64
    hallucinogen-frequency     17 non-null float64
    inhalant-use               17 non-null float64
    inhalant-frequency         17 non-null object
    pain-releiver-use          17 non-null float64
    pain-releiver-frequency    17 non-null float64
    oxycontin-use              17 non-null float64
    oxycontin-frequency        17 non-null object
    tranquilizer-use           17 non-null float64
    tranquilizer-frequency     17 non-null float64
    stimulant-use              17 non-null float64
    stimulant-frequency        17 non-null float64
    meth-use                   17 non-null float64
    meth-frequency             17 non-null object
    sedative-use               17 non-null float64
    sedative-frequency         17 non-null float64
    dtypes: float64(20), int64(1), object(7)
    memory usage: 3.8+ KB
    


```python
for col in df1.columns:
    print str(col), df1[col].unique()

# this reveals that '-' is used to indicate NaN values
```

    age ['12' '13' '14' '15' '16' '17' '18' '19' '20' '21' '22-23' '24-25' '26-29'
     '30-34' '35-49' '50-64' '65+']
    n [2798 2757 2792 2956 3058 3038 2469 2223 2271 2354 4707 4591 2628 2864 7391
     3923 2448]
    alcohol-use [  3.9   8.5  18.1  29.2  40.1  49.3  58.7  64.6  69.7  83.2  84.2  83.1
      80.7  77.5  75.   67.2]
    alcohol-frequency [  3.   6.   5.  10.  13.  24.  36.  48.  52.]
    marijuana-use [  1.1   3.4   8.7  14.5  22.5  28.   33.7  33.4  34.   33.   28.4  24.9
      20.8  16.4  10.4   7.3   1.2]
    marijuana-frequency [  4.  15.  24.  25.  30.  36.  52.  60.  72.  48.]
    cocaine-use [ 0.1  0.5  1.   2.   3.2  4.1  4.9  4.8  4.5  4.   2.1  1.5  0.9  0. ]
    cocaine-frequency ['5.0' '1.0' '5.5' '4.0' '7.0' '8.0' '6.0' '15.0' '36.0' '-']
    crack-use [ 0.   0.1  0.4  0.5  0.6]
    crack-frequency ['-' '3.0' '9.5' '1.0' '21.0' '10.0' '2.0' '5.0' '17.0' '6.0' '15.0' '48.0'
     '62.0']
    heroin-use [ 0.1  0.   0.2  0.4  0.5  0.9  0.6  1.1  0.7]
    heroin-frequency ['35.5' '-' '2.0' '1.0' '66.5' '64.0' '46.0' '180.0' '45.0' '30.0' '57.5'
     '88.0' '50.0' '66.0' '280.0' '41.0' '120.0']
    hallucinogen-use [ 0.2  0.6  1.6  2.1  3.4  4.8  7.   8.6  7.4  6.3  5.2  4.5  3.2  1.8  0.3
      0.1]
    hallucinogen-frequency [ 52.   6.   3.   4.   2.  44.]
    inhalant-use [ 1.6  2.5  2.6  3.   2.   1.8  1.4  1.5  1.   0.8  0.6  0.4  0.3  0.2  0. ]
    inhalant-frequency ['19.0' '12.0' '5.0' '5.5' '3.0' '4.0' '2.0' '3.5' '10.0' '13.5' '-']
    pain-releiver-use [  2.    2.4   3.9   5.5   6.2   8.5   9.2   9.4  10.    9.    8.3   5.9
       4.2   2.5   0.6]
    pain-releiver-frequency [ 36.  14.  12.  10.   7.   9.  15.  13.  22.  24.]
    oxycontin-use [ 0.1  0.4  0.8  1.1  1.4  1.7  1.5  1.3  1.2  0.9  0.3  0. ]
    oxycontin-frequency ['24.5' '41.0' '4.5' '3.0' '4.0' '6.0' '7.0' '7.5' '12.0' '13.5' '17.5'
     '20.0' '46.0' '5.0' '-']
    tranquilizer-use [ 0.2  0.3  0.9  2.   2.4  3.5  4.9  4.2  5.4  3.9  4.4  4.3  3.6  1.9  1.4]
    tranquilizer-frequency [ 52.   25.5   5.    4.5  11.    7.   12.   10.    8.    6. ]
    stimulant-use [ 0.2  0.3  0.8  1.5  1.8  2.8  3.   3.3  4.   4.1  3.6  2.6  2.3  1.4  0.6
      0. ]
    stimulant-frequency [   2.     4.    12.     6.     9.5    9.     8.    10.     7.    24.   364. ]
    meth-use [ 0.   0.1  0.3  0.6  0.5  0.4  0.9  0.7  0.2]
    meth-frequency ['-' '5.0' '24.0' '10.5' '36.0' '48.0' '12.0' '105.0' '2.0' '46.0' '21.0'
     '30.0' '54.0' '104.0']
    sedative-use [ 0.2  0.1  0.4  0.5  0.3  0. ]
    sedative-frequency [  13.    19.    16.5   30.     3.     6.5   10.     6.     4.     9.    52.
       17.5  104.    15. ]
    


```python
# Float values or return NaN function
def f(x):
    try:
        return np.float(x)
    except:
        return np.nan

df1['cocaine-frequency'] = df1['cocaine-frequency'].apply(f)
df1['crack-frequency'] = df1['crack-frequency'].apply(f)
df1['heroin-frequency'] = df1['heroin-frequency'].apply(f)
df1['inhalant-frequency'] = df1['inhalant-frequency'].apply(f)
df1['oxycontin-frequency'] = df1['oxycontin-frequency'].apply(f)
df1['meth-frequency'] = df1['meth-frequency'].apply(f)

df1.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 17 entries, 0 to 16
    Data columns (total 28 columns):
    age                        17 non-null object
    n                          17 non-null int64
    alcohol-use                17 non-null float64
    alcohol-frequency          17 non-null float64
    marijuana-use              17 non-null float64
    marijuana-frequency        17 non-null float64
    cocaine-use                17 non-null float64
    cocaine-frequency          16 non-null float64
    crack-use                  17 non-null float64
    crack-frequency            14 non-null float64
    heroin-use                 17 non-null float64
    heroin-frequency           16 non-null float64
    hallucinogen-use           17 non-null float64
    hallucinogen-frequency     17 non-null float64
    inhalant-use               17 non-null float64
    inhalant-frequency         16 non-null float64
    pain-releiver-use          17 non-null float64
    pain-releiver-frequency    17 non-null float64
    oxycontin-use              17 non-null float64
    oxycontin-frequency        16 non-null float64
    tranquilizer-use           17 non-null float64
    tranquilizer-frequency     17 non-null float64
    stimulant-use              17 non-null float64
    stimulant-frequency        17 non-null float64
    meth-use                   17 non-null float64
    meth-frequency             15 non-null float64
    sedative-use               17 non-null float64
    sedative-frequency         17 non-null float64
    dtypes: float64(26), int64(1), object(1)
    memory usage: 3.8+ KB
    


```python
# clean use columns with 0, we cant possibly have 0 use with frequency being available
# age, cocaine freq, crack freq, heroin freq, inhalant freq, oxycontin freq,meth freq needs to be cleaned, as they are of object type and should be int/float instead


df1.fillna(value=0.,inplace=True)

df1.replace(to_replace=0., value=0.0001,inplace=True)

df1
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
      <th>age</th>
      <th>n</th>
      <th>alcohol-use</th>
      <th>alcohol-frequency</th>
      <th>marijuana-use</th>
      <th>marijuana-frequency</th>
      <th>cocaine-use</th>
      <th>cocaine-frequency</th>
      <th>crack-use</th>
      <th>crack-frequency</th>
      <th>heroin-use</th>
      <th>heroin-frequency</th>
      <th>hallucinogen-use</th>
      <th>hallucinogen-frequency</th>
      <th>inhalant-use</th>
      <th>inhalant-frequency</th>
      <th>pain-releiver-use</th>
      <th>pain-releiver-frequency</th>
      <th>oxycontin-use</th>
      <th>oxycontin-frequency</th>
      <th>tranquilizer-use</th>
      <th>tranquilizer-frequency</th>
      <th>stimulant-use</th>
      <th>stimulant-frequency</th>
      <th>meth-use</th>
      <th>meth-frequency</th>
      <th>sedative-use</th>
      <th>sedative-frequency</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>12</td>
      <td>2798</td>
      <td>3.9</td>
      <td>3.0</td>
      <td>1.1</td>
      <td>4.0</td>
      <td>0.1000</td>
      <td>5.0000</td>
      <td>0.0001</td>
      <td>0.0001</td>
      <td>0.1000</td>
      <td>35.5000</td>
      <td>0.2</td>
      <td>52.0</td>
      <td>1.6000</td>
      <td>19.0000</td>
      <td>2.0</td>
      <td>36.0</td>
      <td>0.1000</td>
      <td>24.5000</td>
      <td>0.2</td>
      <td>52.0</td>
      <td>0.2000</td>
      <td>2.0</td>
      <td>0.0001</td>
      <td>0.0001</td>
      <td>0.2000</td>
      <td>13.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>13</td>
      <td>2757</td>
      <td>8.5</td>
      <td>6.0</td>
      <td>3.4</td>
      <td>15.0</td>
      <td>0.1000</td>
      <td>1.0000</td>
      <td>0.0001</td>
      <td>3.0000</td>
      <td>0.0001</td>
      <td>0.0001</td>
      <td>0.6</td>
      <td>6.0</td>
      <td>2.5000</td>
      <td>12.0000</td>
      <td>2.4</td>
      <td>14.0</td>
      <td>0.1000</td>
      <td>41.0000</td>
      <td>0.3</td>
      <td>25.5</td>
      <td>0.3000</td>
      <td>4.0</td>
      <td>0.1000</td>
      <td>5.0000</td>
      <td>0.1000</td>
      <td>19.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>14</td>
      <td>2792</td>
      <td>18.1</td>
      <td>5.0</td>
      <td>8.7</td>
      <td>24.0</td>
      <td>0.1000</td>
      <td>5.5000</td>
      <td>0.0001</td>
      <td>0.0001</td>
      <td>0.1000</td>
      <td>2.0000</td>
      <td>1.6</td>
      <td>3.0</td>
      <td>2.6000</td>
      <td>5.0000</td>
      <td>3.9</td>
      <td>12.0</td>
      <td>0.4000</td>
      <td>4.5000</td>
      <td>0.9</td>
      <td>5.0</td>
      <td>0.8000</td>
      <td>12.0</td>
      <td>0.1000</td>
      <td>24.0000</td>
      <td>0.2000</td>
      <td>16.5</td>
    </tr>
    <tr>
      <th>3</th>
      <td>15</td>
      <td>2956</td>
      <td>29.2</td>
      <td>6.0</td>
      <td>14.5</td>
      <td>25.0</td>
      <td>0.5000</td>
      <td>4.0000</td>
      <td>0.1000</td>
      <td>9.5000</td>
      <td>0.2000</td>
      <td>1.0000</td>
      <td>2.1</td>
      <td>4.0</td>
      <td>2.5000</td>
      <td>5.5000</td>
      <td>5.5</td>
      <td>10.0</td>
      <td>0.8000</td>
      <td>3.0000</td>
      <td>2.0</td>
      <td>4.5</td>
      <td>1.5000</td>
      <td>6.0</td>
      <td>0.3000</td>
      <td>10.5000</td>
      <td>0.4000</td>
      <td>30.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>16</td>
      <td>3058</td>
      <td>40.1</td>
      <td>10.0</td>
      <td>22.5</td>
      <td>30.0</td>
      <td>1.0000</td>
      <td>7.0000</td>
      <td>0.0001</td>
      <td>1.0000</td>
      <td>0.1000</td>
      <td>66.5000</td>
      <td>3.4</td>
      <td>3.0</td>
      <td>3.0000</td>
      <td>3.0000</td>
      <td>6.2</td>
      <td>7.0</td>
      <td>1.1000</td>
      <td>4.0000</td>
      <td>2.4</td>
      <td>11.0</td>
      <td>1.8000</td>
      <td>9.5</td>
      <td>0.3000</td>
      <td>36.0000</td>
      <td>0.2000</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>17</td>
      <td>3038</td>
      <td>49.3</td>
      <td>13.0</td>
      <td>28.0</td>
      <td>36.0</td>
      <td>2.0000</td>
      <td>5.0000</td>
      <td>0.1000</td>
      <td>21.0000</td>
      <td>0.1000</td>
      <td>64.0000</td>
      <td>4.8</td>
      <td>3.0</td>
      <td>2.0000</td>
      <td>4.0000</td>
      <td>8.5</td>
      <td>9.0</td>
      <td>1.4000</td>
      <td>6.0000</td>
      <td>3.5</td>
      <td>7.0</td>
      <td>2.8000</td>
      <td>9.0</td>
      <td>0.6000</td>
      <td>48.0000</td>
      <td>0.5000</td>
      <td>6.5</td>
    </tr>
    <tr>
      <th>6</th>
      <td>18</td>
      <td>2469</td>
      <td>58.7</td>
      <td>24.0</td>
      <td>33.7</td>
      <td>52.0</td>
      <td>3.2000</td>
      <td>5.0000</td>
      <td>0.4000</td>
      <td>10.0000</td>
      <td>0.4000</td>
      <td>46.0000</td>
      <td>7.0</td>
      <td>4.0</td>
      <td>1.8000</td>
      <td>4.0000</td>
      <td>9.2</td>
      <td>12.0</td>
      <td>1.7000</td>
      <td>7.0000</td>
      <td>4.9</td>
      <td>12.0</td>
      <td>3.0000</td>
      <td>8.0</td>
      <td>0.5000</td>
      <td>12.0000</td>
      <td>0.4000</td>
      <td>10.0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>19</td>
      <td>2223</td>
      <td>64.6</td>
      <td>36.0</td>
      <td>33.4</td>
      <td>60.0</td>
      <td>4.1000</td>
      <td>5.5000</td>
      <td>0.5000</td>
      <td>2.0000</td>
      <td>0.5000</td>
      <td>180.0000</td>
      <td>8.6</td>
      <td>3.0</td>
      <td>1.4000</td>
      <td>3.0000</td>
      <td>9.4</td>
      <td>12.0</td>
      <td>1.5000</td>
      <td>7.5000</td>
      <td>4.2</td>
      <td>4.5</td>
      <td>3.3000</td>
      <td>6.0</td>
      <td>0.4000</td>
      <td>105.0000</td>
      <td>0.3000</td>
      <td>6.0</td>
    </tr>
    <tr>
      <th>8</th>
      <td>20</td>
      <td>2271</td>
      <td>69.7</td>
      <td>48.0</td>
      <td>34.0</td>
      <td>60.0</td>
      <td>4.9000</td>
      <td>8.0000</td>
      <td>0.6000</td>
      <td>5.0000</td>
      <td>0.9000</td>
      <td>45.0000</td>
      <td>7.4</td>
      <td>2.0</td>
      <td>1.5000</td>
      <td>4.0000</td>
      <td>10.0</td>
      <td>10.0</td>
      <td>1.7000</td>
      <td>12.0000</td>
      <td>5.4</td>
      <td>10.0</td>
      <td>4.0000</td>
      <td>12.0</td>
      <td>0.9000</td>
      <td>12.0000</td>
      <td>0.5000</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>9</th>
      <td>21</td>
      <td>2354</td>
      <td>83.2</td>
      <td>52.0</td>
      <td>33.0</td>
      <td>52.0</td>
      <td>4.8000</td>
      <td>5.0000</td>
      <td>0.5000</td>
      <td>17.0000</td>
      <td>0.6000</td>
      <td>30.0000</td>
      <td>6.3</td>
      <td>4.0</td>
      <td>1.4000</td>
      <td>2.0000</td>
      <td>9.0</td>
      <td>15.0</td>
      <td>1.3000</td>
      <td>13.5000</td>
      <td>3.9</td>
      <td>7.0</td>
      <td>4.1000</td>
      <td>10.0</td>
      <td>0.6000</td>
      <td>2.0000</td>
      <td>0.3000</td>
      <td>9.0</td>
    </tr>
    <tr>
      <th>10</th>
      <td>22-23</td>
      <td>4707</td>
      <td>84.2</td>
      <td>52.0</td>
      <td>28.4</td>
      <td>52.0</td>
      <td>4.5000</td>
      <td>5.0000</td>
      <td>0.5000</td>
      <td>5.0000</td>
      <td>1.1000</td>
      <td>57.5000</td>
      <td>5.2</td>
      <td>3.0</td>
      <td>1.0000</td>
      <td>4.0000</td>
      <td>10.0</td>
      <td>15.0</td>
      <td>1.7000</td>
      <td>17.5000</td>
      <td>4.4</td>
      <td>12.0</td>
      <td>3.6000</td>
      <td>10.0</td>
      <td>0.6000</td>
      <td>46.0000</td>
      <td>0.2000</td>
      <td>52.0</td>
    </tr>
    <tr>
      <th>11</th>
      <td>24-25</td>
      <td>4591</td>
      <td>83.1</td>
      <td>52.0</td>
      <td>24.9</td>
      <td>60.0</td>
      <td>4.0000</td>
      <td>6.0000</td>
      <td>0.5000</td>
      <td>6.0000</td>
      <td>0.7000</td>
      <td>88.0000</td>
      <td>4.5</td>
      <td>2.0</td>
      <td>0.8000</td>
      <td>2.0000</td>
      <td>9.0</td>
      <td>15.0</td>
      <td>1.3000</td>
      <td>20.0000</td>
      <td>4.3</td>
      <td>10.0</td>
      <td>2.6000</td>
      <td>10.0</td>
      <td>0.7000</td>
      <td>21.0000</td>
      <td>0.2000</td>
      <td>17.5</td>
    </tr>
    <tr>
      <th>12</th>
      <td>26-29</td>
      <td>2628</td>
      <td>80.7</td>
      <td>52.0</td>
      <td>20.8</td>
      <td>52.0</td>
      <td>3.2000</td>
      <td>5.0000</td>
      <td>0.4000</td>
      <td>6.0000</td>
      <td>0.6000</td>
      <td>50.0000</td>
      <td>3.2</td>
      <td>3.0</td>
      <td>0.6000</td>
      <td>4.0000</td>
      <td>8.3</td>
      <td>13.0</td>
      <td>1.2000</td>
      <td>13.5000</td>
      <td>4.2</td>
      <td>10.0</td>
      <td>2.3000</td>
      <td>7.0</td>
      <td>0.6000</td>
      <td>30.0000</td>
      <td>0.4000</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>13</th>
      <td>30-34</td>
      <td>2864</td>
      <td>77.5</td>
      <td>52.0</td>
      <td>16.4</td>
      <td>72.0</td>
      <td>2.1000</td>
      <td>8.0000</td>
      <td>0.5000</td>
      <td>15.0000</td>
      <td>0.4000</td>
      <td>66.0000</td>
      <td>1.8</td>
      <td>2.0</td>
      <td>0.4000</td>
      <td>3.5000</td>
      <td>5.9</td>
      <td>22.0</td>
      <td>0.9000</td>
      <td>46.0000</td>
      <td>3.6</td>
      <td>8.0</td>
      <td>1.4000</td>
      <td>12.0</td>
      <td>0.4000</td>
      <td>54.0000</td>
      <td>0.4000</td>
      <td>10.0</td>
    </tr>
    <tr>
      <th>14</th>
      <td>35-49</td>
      <td>7391</td>
      <td>75.0</td>
      <td>52.0</td>
      <td>10.4</td>
      <td>48.0</td>
      <td>1.5000</td>
      <td>15.0000</td>
      <td>0.5000</td>
      <td>48.0000</td>
      <td>0.1000</td>
      <td>280.0000</td>
      <td>0.6</td>
      <td>3.0</td>
      <td>0.3000</td>
      <td>10.0000</td>
      <td>4.2</td>
      <td>12.0</td>
      <td>0.3000</td>
      <td>12.0000</td>
      <td>1.9</td>
      <td>6.0</td>
      <td>0.6000</td>
      <td>24.0</td>
      <td>0.2000</td>
      <td>104.0000</td>
      <td>0.3000</td>
      <td>10.0</td>
    </tr>
    <tr>
      <th>15</th>
      <td>50-64</td>
      <td>3923</td>
      <td>67.2</td>
      <td>52.0</td>
      <td>7.3</td>
      <td>52.0</td>
      <td>0.9000</td>
      <td>36.0000</td>
      <td>0.4000</td>
      <td>62.0000</td>
      <td>0.1000</td>
      <td>41.0000</td>
      <td>0.3</td>
      <td>44.0</td>
      <td>0.2000</td>
      <td>13.5000</td>
      <td>2.5</td>
      <td>12.0</td>
      <td>0.4000</td>
      <td>5.0000</td>
      <td>1.4</td>
      <td>10.0</td>
      <td>0.3000</td>
      <td>24.0</td>
      <td>0.2000</td>
      <td>30.0000</td>
      <td>0.2000</td>
      <td>104.0</td>
    </tr>
    <tr>
      <th>16</th>
      <td>65+</td>
      <td>2448</td>
      <td>49.3</td>
      <td>52.0</td>
      <td>1.2</td>
      <td>36.0</td>
      <td>0.0001</td>
      <td>0.0001</td>
      <td>0.0001</td>
      <td>0.0001</td>
      <td>0.0001</td>
      <td>120.0000</td>
      <td>0.1</td>
      <td>2.0</td>
      <td>0.0001</td>
      <td>0.0001</td>
      <td>0.6</td>
      <td>24.0</td>
      <td>0.0001</td>
      <td>0.0001</td>
      <td>0.2</td>
      <td>5.0</td>
      <td>0.0001</td>
      <td>364.0</td>
      <td>0.0001</td>
      <td>0.0001</td>
      <td>0.0001</td>
      <td>15.0</td>
    </tr>
  </tbody>
</table>
</div>



### 7.2 Do a high-level, initial overview of the data


```python
pd.options.display.max_columns = 999
df1.describe(include='all')

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
      <th>age</th>
      <th>n</th>
      <th>alcohol-use</th>
      <th>alcohol-frequency</th>
      <th>marijuana-use</th>
      <th>marijuana-frequency</th>
      <th>cocaine-use</th>
      <th>cocaine-frequency</th>
      <th>crack-use</th>
      <th>crack-frequency</th>
      <th>heroin-use</th>
      <th>heroin-frequency</th>
      <th>hallucinogen-use</th>
      <th>hallucinogen-frequency</th>
      <th>inhalant-use</th>
      <th>inhalant-frequency</th>
      <th>pain-releiver-use</th>
      <th>pain-releiver-frequency</th>
      <th>oxycontin-use</th>
      <th>oxycontin-frequency</th>
      <th>tranquilizer-use</th>
      <th>tranquilizer-frequency</th>
      <th>stimulant-use</th>
      <th>stimulant-frequency</th>
      <th>meth-use</th>
      <th>meth-frequency</th>
      <th>sedative-use</th>
      <th>sedative-frequency</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>17</td>
      <td>17.000000</td>
      <td>17.000000</td>
      <td>17.000000</td>
      <td>17.000000</td>
      <td>17.000000</td>
      <td>17.000000</td>
      <td>17.000000</td>
      <td>17.000000</td>
      <td>17.000000</td>
      <td>17.000000</td>
      <td>17.000000</td>
      <td>17.000000</td>
      <td>17.000000</td>
      <td>17.000000</td>
      <td>17.000000</td>
      <td>17.000000</td>
      <td>17.000000</td>
      <td>17.000000</td>
      <td>17.000000</td>
      <td>17.000000</td>
      <td>17.000000</td>
      <td>17.000000</td>
      <td>17.000000</td>
      <td>17.000000</td>
      <td>17.000000</td>
      <td>17.000000</td>
      <td>17.000000</td>
    </tr>
    <tr>
      <th>unique</th>
      <td>17</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>top</th>
      <td>22-23</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>freq</th>
      <td>1</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>NaN</td>
      <td>3251.058824</td>
      <td>55.429412</td>
      <td>33.352941</td>
      <td>18.923529</td>
      <td>42.941176</td>
      <td>2.176476</td>
      <td>7.411771</td>
      <td>0.294147</td>
      <td>12.382371</td>
      <td>0.352953</td>
      <td>68.970594</td>
      <td>3.394118</td>
      <td>8.411765</td>
      <td>1.388241</td>
      <td>5.794124</td>
      <td>6.270588</td>
      <td>14.705882</td>
      <td>0.935300</td>
      <td>13.941182</td>
      <td>2.805882</td>
      <td>11.735294</td>
      <td>1.917653</td>
      <td>31.147059</td>
      <td>0.382365</td>
      <td>31.735306</td>
      <td>0.282359</td>
      <td>19.382353</td>
    </tr>
    <tr>
      <th>std</th>
      <td>NaN</td>
      <td>1297.890426</td>
      <td>26.878866</td>
      <td>21.318833</td>
      <td>11.959752</td>
      <td>18.362566</td>
      <td>1.816765</td>
      <td>8.014113</td>
      <td>0.235733</td>
      <td>17.361519</td>
      <td>0.333749</td>
      <td>70.153279</td>
      <td>2.792506</td>
      <td>15.000245</td>
      <td>0.927273</td>
      <td>4.937283</td>
      <td>3.166379</td>
      <td>6.935098</td>
      <td>0.608206</td>
      <td>12.902128</td>
      <td>1.753379</td>
      <td>11.485205</td>
      <td>1.407665</td>
      <td>85.973790</td>
      <td>0.262744</td>
      <td>32.206411</td>
      <td>0.137988</td>
      <td>24.833527</td>
    </tr>
    <tr>
      <th>min</th>
      <td>NaN</td>
      <td>2223.000000</td>
      <td>3.900000</td>
      <td>3.000000</td>
      <td>1.100000</td>
      <td>4.000000</td>
      <td>0.000100</td>
      <td>0.000100</td>
      <td>0.000100</td>
      <td>0.000100</td>
      <td>0.000100</td>
      <td>0.000100</td>
      <td>0.100000</td>
      <td>2.000000</td>
      <td>0.000100</td>
      <td>0.000100</td>
      <td>0.600000</td>
      <td>7.000000</td>
      <td>0.000100</td>
      <td>0.000100</td>
      <td>0.200000</td>
      <td>4.500000</td>
      <td>0.000100</td>
      <td>2.000000</td>
      <td>0.000100</td>
      <td>0.000100</td>
      <td>0.000100</td>
      <td>3.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>NaN</td>
      <td>2469.000000</td>
      <td>40.100000</td>
      <td>10.000000</td>
      <td>8.700000</td>
      <td>30.000000</td>
      <td>0.500000</td>
      <td>5.000000</td>
      <td>0.000100</td>
      <td>2.000000</td>
      <td>0.100000</td>
      <td>35.500000</td>
      <td>0.600000</td>
      <td>3.000000</td>
      <td>0.600000</td>
      <td>3.000000</td>
      <td>3.900000</td>
      <td>12.000000</td>
      <td>0.400000</td>
      <td>5.000000</td>
      <td>1.400000</td>
      <td>6.000000</td>
      <td>0.600000</td>
      <td>7.000000</td>
      <td>0.200000</td>
      <td>10.500000</td>
      <td>0.200000</td>
      <td>6.500000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>NaN</td>
      <td>2798.000000</td>
      <td>64.600000</td>
      <td>48.000000</td>
      <td>20.800000</td>
      <td>52.000000</td>
      <td>2.000000</td>
      <td>5.000000</td>
      <td>0.400000</td>
      <td>6.000000</td>
      <td>0.200000</td>
      <td>50.000000</td>
      <td>3.200000</td>
      <td>3.000000</td>
      <td>1.400000</td>
      <td>4.000000</td>
      <td>6.200000</td>
      <td>12.000000</td>
      <td>1.100000</td>
      <td>12.000000</td>
      <td>3.500000</td>
      <td>10.000000</td>
      <td>1.800000</td>
      <td>10.000000</td>
      <td>0.400000</td>
      <td>24.000000</td>
      <td>0.300000</td>
      <td>10.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>NaN</td>
      <td>3058.000000</td>
      <td>77.500000</td>
      <td>52.000000</td>
      <td>28.400000</td>
      <td>52.000000</td>
      <td>4.000000</td>
      <td>7.000000</td>
      <td>0.500000</td>
      <td>15.000000</td>
      <td>0.600000</td>
      <td>66.500000</td>
      <td>5.200000</td>
      <td>4.000000</td>
      <td>2.000000</td>
      <td>5.500000</td>
      <td>9.000000</td>
      <td>15.000000</td>
      <td>1.400000</td>
      <td>17.500000</td>
      <td>4.200000</td>
      <td>11.000000</td>
      <td>3.000000</td>
      <td>12.000000</td>
      <td>0.600000</td>
      <td>46.000000</td>
      <td>0.400000</td>
      <td>17.500000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>NaN</td>
      <td>7391.000000</td>
      <td>84.200000</td>
      <td>52.000000</td>
      <td>34.000000</td>
      <td>72.000000</td>
      <td>4.900000</td>
      <td>36.000000</td>
      <td>0.600000</td>
      <td>62.000000</td>
      <td>1.100000</td>
      <td>280.000000</td>
      <td>8.600000</td>
      <td>52.000000</td>
      <td>3.000000</td>
      <td>19.000000</td>
      <td>10.000000</td>
      <td>36.000000</td>
      <td>1.700000</td>
      <td>46.000000</td>
      <td>5.400000</td>
      <td>52.000000</td>
      <td>4.100000</td>
      <td>364.000000</td>
      <td>0.900000</td>
      <td>105.000000</td>
      <td>0.500000</td>
      <td>104.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
df2 = df1.iloc[:,[2,4,6,8,10,12,14,16,18,20,22,24,26]] # Split df1 into use columns

df3 = df1.iloc[:,[3,5,7,9,11,13,15,17,19,21,23,25,27]] # Split df1 into freq columns


matplotlib.pyplot.figure(figsize=(10,10))
sns.heatmap(df2.corr(), cmap="PiYG", center=0)

# Heatmap showing use correlations
```




    <matplotlib.axes._subplots.AxesSubplot at 0x4d99e7b8>




![png](output_59_1.png)



```python
matplotlib.pyplot.figure(figsize=(10,10))
sns.heatmap(df3.corr(), cmap="PiYG", center=0)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x4dde2208>




![png](output_60_1.png)


## <span style="color:blue">Observations:</span>
**1. Drug use correlations:**  
It is observed that most drug use are correlated with one another, except for inhalant use.

**2. Drug frequency of use correlations:**  
Drug frequencies do not seem to be correlated.


```python
# Plotting drug use across age
f, ax = plt.subplots(figsize=(15, 15))

x = df1['age']

for col in df2.columns:
    y = df2[col]
    plt.plot(x, y)

f.suptitle('Drug use across age',fontsize = 20)
f.legend()
plt.xlabel('age')
plt.ylabel('useage')
```




    Text(0,0.5,u'useage')




![png](output_62_1.png)


## <span style="color:blue">Observations:</span>
The predominant drugs that are used across all age groups are the following:
1. Alchohol
2. Marijuana
3. Cocaine


```python
# Plotting drug use frequency across age
f, ax = plt.subplots(figsize=(15, 15))

x = df1['age']

for col in df3.columns:
    y = df3[col]
    plt.plot(x, y)

f.suptitle('Drug frequency across age',fontsize = 20)
f.legend()
plt.xlabel('age')
plt.ylabel('frequency')
```




    Text(0,0.5,u'frequency')




![png](output_64_1.png)


## <span style="color:blue">Observations:</span>
Frequency of drug use does not follow a pattern but interestingly there are spikes in certain ages where frequency of a drug use is high.

### 7.3 Create a testable hypothesis about this data

<span style="color:blue">** Question and deliverables**<span>

Question: Are the means of drug use across age groups different from one another; specifically is the mean of drug use amongst adolescents (age 21 and below) different from mean of drug use amongst non-adolescents (age >21)


```python
# Compile mean of drug use among columns for every row
df2['mean_use']=df2.mean(axis=1)
df2['minor_label'] = 0 # initialise label column

df2.loc[[0,1,2,3,4,5,6,7,8,9],['minor_label']]=1 # minors are labelled group 1

mean_ado = df2.loc[[0,1,2,3,4,5,6,7,8,9],'mean_use'].sum()/10

mean_nonado = df2.loc[[10,11,12,13,14,15,16],'mean_use'].sum()/7

print mean_nonado

df2

```

    C:\ProgramData\Anaconda2\lib\site-packages\ipykernel_launcher.py:2: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-view-versus-copy
      
    C:\ProgramData\Anaconda2\lib\site-packages\ipykernel_launcher.py:3: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-view-versus-copy
      This is separate from the ipykernel package so we can avoid doing imports until
    

    8.22967912088
    




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
      <th>alcohol-use</th>
      <th>marijuana-use</th>
      <th>cocaine-use</th>
      <th>crack-use</th>
      <th>heroin-use</th>
      <th>hallucinogen-use</th>
      <th>inhalant-use</th>
      <th>pain-releiver-use</th>
      <th>oxycontin-use</th>
      <th>tranquilizer-use</th>
      <th>stimulant-use</th>
      <th>meth-use</th>
      <th>sedative-use</th>
      <th>mean_use</th>
      <th>minor_label</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>3.9</td>
      <td>1.1</td>
      <td>0.1000</td>
      <td>0.0001</td>
      <td>0.1000</td>
      <td>0.2</td>
      <td>1.6000</td>
      <td>2.0</td>
      <td>0.1000</td>
      <td>0.2</td>
      <td>0.2000</td>
      <td>0.0001</td>
      <td>0.2000</td>
      <td>0.746169</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>8.5</td>
      <td>3.4</td>
      <td>0.1000</td>
      <td>0.0001</td>
      <td>0.0001</td>
      <td>0.6</td>
      <td>2.5000</td>
      <td>2.4</td>
      <td>0.1000</td>
      <td>0.3</td>
      <td>0.3000</td>
      <td>0.1000</td>
      <td>0.1000</td>
      <td>1.415400</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>18.1</td>
      <td>8.7</td>
      <td>0.1000</td>
      <td>0.0001</td>
      <td>0.1000</td>
      <td>1.6</td>
      <td>2.6000</td>
      <td>3.9</td>
      <td>0.4000</td>
      <td>0.9</td>
      <td>0.8000</td>
      <td>0.1000</td>
      <td>0.2000</td>
      <td>2.884623</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>29.2</td>
      <td>14.5</td>
      <td>0.5000</td>
      <td>0.1000</td>
      <td>0.2000</td>
      <td>2.1</td>
      <td>2.5000</td>
      <td>5.5</td>
      <td>0.8000</td>
      <td>2.0</td>
      <td>1.5000</td>
      <td>0.3000</td>
      <td>0.4000</td>
      <td>4.584615</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>40.1</td>
      <td>22.5</td>
      <td>1.0000</td>
      <td>0.0001</td>
      <td>0.1000</td>
      <td>3.4</td>
      <td>3.0000</td>
      <td>6.2</td>
      <td>1.1000</td>
      <td>2.4</td>
      <td>1.8000</td>
      <td>0.3000</td>
      <td>0.2000</td>
      <td>6.315392</td>
      <td>1</td>
    </tr>
    <tr>
      <th>5</th>
      <td>49.3</td>
      <td>28.0</td>
      <td>2.0000</td>
      <td>0.1000</td>
      <td>0.1000</td>
      <td>4.8</td>
      <td>2.0000</td>
      <td>8.5</td>
      <td>1.4000</td>
      <td>3.5</td>
      <td>2.8000</td>
      <td>0.6000</td>
      <td>0.5000</td>
      <td>7.969231</td>
      <td>1</td>
    </tr>
    <tr>
      <th>6</th>
      <td>58.7</td>
      <td>33.7</td>
      <td>3.2000</td>
      <td>0.4000</td>
      <td>0.4000</td>
      <td>7.0</td>
      <td>1.8000</td>
      <td>9.2</td>
      <td>1.7000</td>
      <td>4.9</td>
      <td>3.0000</td>
      <td>0.5000</td>
      <td>0.4000</td>
      <td>9.607692</td>
      <td>1</td>
    </tr>
    <tr>
      <th>7</th>
      <td>64.6</td>
      <td>33.4</td>
      <td>4.1000</td>
      <td>0.5000</td>
      <td>0.5000</td>
      <td>8.6</td>
      <td>1.4000</td>
      <td>9.4</td>
      <td>1.5000</td>
      <td>4.2</td>
      <td>3.3000</td>
      <td>0.4000</td>
      <td>0.3000</td>
      <td>10.169231</td>
      <td>1</td>
    </tr>
    <tr>
      <th>8</th>
      <td>69.7</td>
      <td>34.0</td>
      <td>4.9000</td>
      <td>0.6000</td>
      <td>0.9000</td>
      <td>7.4</td>
      <td>1.5000</td>
      <td>10.0</td>
      <td>1.7000</td>
      <td>5.4</td>
      <td>4.0000</td>
      <td>0.9000</td>
      <td>0.5000</td>
      <td>10.884615</td>
      <td>1</td>
    </tr>
    <tr>
      <th>9</th>
      <td>83.2</td>
      <td>33.0</td>
      <td>4.8000</td>
      <td>0.5000</td>
      <td>0.6000</td>
      <td>6.3</td>
      <td>1.4000</td>
      <td>9.0</td>
      <td>1.3000</td>
      <td>3.9</td>
      <td>4.1000</td>
      <td>0.6000</td>
      <td>0.3000</td>
      <td>11.461538</td>
      <td>1</td>
    </tr>
    <tr>
      <th>10</th>
      <td>84.2</td>
      <td>28.4</td>
      <td>4.5000</td>
      <td>0.5000</td>
      <td>1.1000</td>
      <td>5.2</td>
      <td>1.0000</td>
      <td>10.0</td>
      <td>1.7000</td>
      <td>4.4</td>
      <td>3.6000</td>
      <td>0.6000</td>
      <td>0.2000</td>
      <td>11.184615</td>
      <td>0</td>
    </tr>
    <tr>
      <th>11</th>
      <td>83.1</td>
      <td>24.9</td>
      <td>4.0000</td>
      <td>0.5000</td>
      <td>0.7000</td>
      <td>4.5</td>
      <td>0.8000</td>
      <td>9.0</td>
      <td>1.3000</td>
      <td>4.3</td>
      <td>2.6000</td>
      <td>0.7000</td>
      <td>0.2000</td>
      <td>10.507692</td>
      <td>0</td>
    </tr>
    <tr>
      <th>12</th>
      <td>80.7</td>
      <td>20.8</td>
      <td>3.2000</td>
      <td>0.4000</td>
      <td>0.6000</td>
      <td>3.2</td>
      <td>0.6000</td>
      <td>8.3</td>
      <td>1.2000</td>
      <td>4.2</td>
      <td>2.3000</td>
      <td>0.6000</td>
      <td>0.4000</td>
      <td>9.730769</td>
      <td>0</td>
    </tr>
    <tr>
      <th>13</th>
      <td>77.5</td>
      <td>16.4</td>
      <td>2.1000</td>
      <td>0.5000</td>
      <td>0.4000</td>
      <td>1.8</td>
      <td>0.4000</td>
      <td>5.9</td>
      <td>0.9000</td>
      <td>3.6</td>
      <td>1.4000</td>
      <td>0.4000</td>
      <td>0.4000</td>
      <td>8.592308</td>
      <td>0</td>
    </tr>
    <tr>
      <th>14</th>
      <td>75.0</td>
      <td>10.4</td>
      <td>1.5000</td>
      <td>0.5000</td>
      <td>0.1000</td>
      <td>0.6</td>
      <td>0.3000</td>
      <td>4.2</td>
      <td>0.3000</td>
      <td>1.9</td>
      <td>0.6000</td>
      <td>0.2000</td>
      <td>0.3000</td>
      <td>7.376923</td>
      <td>0</td>
    </tr>
    <tr>
      <th>15</th>
      <td>67.2</td>
      <td>7.3</td>
      <td>0.9000</td>
      <td>0.4000</td>
      <td>0.1000</td>
      <td>0.3</td>
      <td>0.2000</td>
      <td>2.5</td>
      <td>0.4000</td>
      <td>1.4</td>
      <td>0.3000</td>
      <td>0.2000</td>
      <td>0.2000</td>
      <td>6.261538</td>
      <td>0</td>
    </tr>
    <tr>
      <th>16</th>
      <td>49.3</td>
      <td>1.2</td>
      <td>0.0001</td>
      <td>0.0001</td>
      <td>0.0001</td>
      <td>0.1</td>
      <td>0.0001</td>
      <td>0.6</td>
      <td>0.0001</td>
      <td>0.2</td>
      <td>0.0001</td>
      <td>0.0001</td>
      <td>0.0001</td>
      <td>3.953908</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
# perform t test
# Let null hypothesis be: The mean overall drug use for adolescents is equals to the mean overall drug use for non-adolescents

list1 = df2['mean_use'].tolist()
list2 = list1[-7:]
list3 = list1[0:10]

stats.ttest_ind(list3, list2)
```




    Ttest_indResult(statistic=-0.94213218959786749, pvalue=0.36105217344934259)




```python
# Visualisation of values using box plot
plt.subplots(figsize=(8,8))
ax = sns.boxplot(x='minor_label', y='mean_use', data=df2)
ax.set(xticklabels=['Non-minor (>21 y.o.)', 'Minor (<=21 y.o.)'], title='Box Plot of overall drug use')
```




    [[Text(0,0,u'Non-minor (>21 y.o.)'), Text(0,0,u'Minor (<=21 y.o.)')],
     Text(0.5,1,u'Box Plot of overall drug use')]




![png](output_70_1.png)



<img src="https://i.imgur.com/wLPdKgZ.png" style="float: left; margin: 25px 15px 0px 0px; height: 25px">
## 8. **Report**

## Problem:  
Is there a method to segment age groups?  
What aggregate parameters can we compare amongst the groups?  

## Approach:  
From the column labels, age 21 and below are represented in single columns while age 22 and above are represented in age ranges. It is fair to seggregate the ages into 2 separate groups, the minors(21 and below) and non-minors(22 and above). In the U.S., youths aged 21 and below may be still categorised as minors in the eyes of the law which indicates a good way to split the dataset into 2 groups.  

Some data preparation will be done to prepare the dataset from analysis:  
1. label the rows accordingly to minors (label=1) / non-minors (label=0)
2. decide on parameter to investigate. In this case, mean overall drug use will be compared between the 2 groups as it is indicative of whether minors are more prone to drug use.
3. t test to be done to compare between mean drug use between 2 groups.  

## Results:  
p-value of 0.361 indicates that we do not reject null hypothesis. We may infer that the mean drug use between minors and non minors have no statistically significant difference.  

Box plot visually shows that both group means are quite close to each other.


<img src="https://i.imgur.com/wLPdKgZ.png" style="float: left; margin: 25px 15px 0px 0px; height: 25px">

## 9. Introduction to dealing with outliers

---

Outliers are an interesting problem in statistics, in that there is not an agreed upon best way to define them. Subjectivity in selecting and analyzing data is a problem that will recur throughout the course.


```python
datasat['Rate']
```




    0     82
    1     81
    2     79
    3     77
    4     72
    5     71
    6     71
    7     69
    8     69
    9     68
    10    67
    11    65
    12    65
    13    63
    14    60
    15    57
    16    56
    17    55
    18    54
    19    53
    20    53
    21    52
    22    51
    23    51
    24    34
    25    33
    26    31
    27    26
    28    23
    29    18
    30    17
    31    13
    32    13
    33    12
    34    12
    35    11
    36    11
    37     9
    38     9
    39     9
    40     8
    41     8
    42     8
    43     7
    44     6
    45     6
    46     5
    47     5
    48     4
    49     4
    50     4
    Name: Rate, dtype: int64



Rate does not seem to have outliers since it is a percentage range and most values fall between 1 to 100.

**Definition of outlier: **  
An outlier is defined to be a value that is 1.5x inter-quartile range away from the first or third quartile.


```python
sns.boxplot(data=datasat)
# box plot shows that only ver_diff has outliers
```




    <matplotlib.axes._subplots.AxesSubplot at 0x380064e0>




![png](output_75_1.png)



```python
# Function to calculate +/- 1.5 IQR range for outliers , ie, values that fall beyond this range will be considered outliers
def iqr(series):
    q1 = datasat[series].quantile(0.25)
    q3 = datasat[series].quantile(0.75)
    iqr = q3 - q1
    out_range = [q1-1.5*iqr, q3+1.5*iqr]
    return out_range

```


```python
print 'outlier range for \'Rate\' is :', iqr('Rate')
print 'outlier range for \'Verbal\' is :', iqr('Verbal')
print 'outlier range for \'Math\' is :', iqr('Math')
print 'outlier range for \'ver_diff\' is :', iqr('ver_diff')
```

    outlier range for 'Rate' is : [-73.5, 146.5]
    outlier range for 'Verbal' is : [409.5, 653.5]
    outlier range for 'Math' is : [421.25, 639.25]
    outlier range for 'ver_diff' is : [-21.75, 20.25]
    

## <span style="color:blue">Observations:</span>
Since the 3 main columns (rate, verbal and math) fall within the outlier range of values, there are no outliers in these columns.  

For the ver_diff column, which is a difference in verbal and math scores, outliers may arise if the difference in the 2 columns is large. Thus, it will not be useful to remove the outliers in this column since this measures a difference, which may be a feature we might be interested in.


<img src="https://i.imgur.com/wLPdKgZ.png" style="float: left; margin: 25px 15px 0px 0px; height: 25px">

## 10. Percentile scoring and spearman rank correlation

---

### 10.1 Calculate the spearman correlation of sat `Verbal` and `Math`


```python
# Spearman correlation
datasat.corr(method='spearman').loc[['Math'],['Verbal']]
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
      <th>Verbal</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Math</th>
      <td>0.909413</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Pearson correlation
datasat.corr(method='pearson').loc[['Math'],['Verbal']]
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
      <th>Verbal</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Math</th>
      <td>0.899909</td>
    </tr>
  </tbody>
</table>
</div>



## <span style="color:blue">Observations:</span>
Q1: Spearman correlation between verbal and math has a slightly higher value than pearson correlation.  

Q2: The Spearman correlation coefficient is defined as the Pearson correlation coefficient between the ranked variables.  

Calculation process: 
First, we rank the variables in each column that we are calculating, in order of magnitude. Then we compute the difference in ranks between the 2 variables, d and square it.

We then calculate the following:
$$\sum d_i^{2}$$

And substitute appropriate values into:  
![](https://wikimedia.org/api/rest_v1/media/math/render/svg/ee773221f85a9ab9ed0e13813d9d1ccafa5dda4e)

### 10.2 Percentile scoring

Look up percentile scoring of data. In other words, the conversion of numeric data to their equivalent percentile scores.

1. Convert `Rate` to percentiles in the sat scores as a new column.
2. Show the percentile of California in `Rate`.
3. How is percentile related to the spearman rank correlation?


```python
datasat['rate_percent'] = datasat['Rate'].map(lambda x: stats.percentileofscore(datasat['Rate'],x))

datasat.head(5)
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
      <th>State</th>
      <th>Rate</th>
      <th>Verbal</th>
      <th>Math</th>
      <th>ver_diff</th>
      <th>rate_percent</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>CT</td>
      <td>82</td>
      <td>509</td>
      <td>510</td>
      <td>-1</td>
      <td>100.000000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>NJ</td>
      <td>81</td>
      <td>499</td>
      <td>513</td>
      <td>-14</td>
      <td>98.039216</td>
    </tr>
    <tr>
      <th>2</th>
      <td>MA</td>
      <td>79</td>
      <td>511</td>
      <td>515</td>
      <td>-4</td>
      <td>96.078431</td>
    </tr>
    <tr>
      <th>3</th>
      <td>NY</td>
      <td>77</td>
      <td>495</td>
      <td>505</td>
      <td>-10</td>
      <td>94.117647</td>
    </tr>
    <tr>
      <th>4</th>
      <td>NH</td>
      <td>72</td>
      <td>520</td>
      <td>516</td>
      <td>4</td>
      <td>92.156863</td>
    </tr>
  </tbody>
</table>
</div>



## <span style="color:blue">Observations:</span>  
The percentile shows the ranking of the variable, ie, the higher the percentile, the higher ranked the variable is.

### 10.3 Percentiles and outliers

1. Why might percentile scoring be useful for dealing with outliers?
2. Plot the distribution of a variable of your choice from the drug use dataset.
3. Plot the same variable but percentile scored.
4. Describe the effect, visually, of coverting raw scores to percentile.

## <span style="color:blue">Observations:</span>  
Outliers will be autoassigned the 100th percentile or close to 0th percentile of an array due to its divergence from the range of most values in a variable.


```python
# Dist plot of drug: stimulant frequency column
sns.distplot(df1['stimulant-frequency'],bins=30,kde=True)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x5191bb00>




![png](output_88_1.png)



```python
# Dist plot of stimulant freq percentiles
df1['stim_percent'] = df1['stimulant-frequency'].map(lambda x: stats.percentileofscore(df1['stimulant-frequency'],x))
sns.distplot(df1['stim_percent'],bins=30,kde=True)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x4d99af60>




![png](output_89_1.png)


## <span style="color:blue">Observations:</span>  
By plotting percentiles rather than the variable itself, it smoothens the distribution and tends it looking somewhat smillar to a symmetric distribuion centred around 50th percentile.
