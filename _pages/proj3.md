---
layout: single
permalink: proj3/
title: &title "Project 3"
author_profile: true
---

<img src="https://i.imgur.com/iYGFYFv.png" style="float: left; margin: 15px; height: 80px">

### Web Scraping and Analysis
---

<img src="/assets/images/proj3.jpg" width="100%">

For this project, we will explore web scraping to obtain data from websites. More specifically, we will be looking at job search websites to obtain job listings together with job descriptions and salary information. We will next use the information scraped to predict salary or classify jobs according to its description. Classifying jobs according to job description involves analysing textual information. We can make use of scikit learn's natural language processing packages to help us in our analysis. 

Packages used:
1. BeautifulSoup
2. pandas
3. urllib3
4. re
5. pickle
6. matplotlib
7. seaborn
8. scikit-learn
9. imblearn

<img src="https://i.imgur.com/wLPdKgZ.png" style="float: left; margin: 25px 15px 0px 0px; height: 25px">

### 1. Getting the links from the job site for scraping using BeautifulSoup and urllib3 


```python
url = 'https://jobscentral.com.sg/jobsearch?q=data%20science&pg='
urllist = []
for i in range(1,15):
    a = url + str(i)
    urllist.append(a)
```


```python
links = []

for link in urllist:
    http = urllib3.PoolManager()
    response = http.request('GET', link)
    soup = BeautifulSoup(response.data, "html.parser")
    
    a = soup.find_all('a',attrs={"class":"job-title"}, href=True)

    for val in a:
        linkstr = 'https://jobscentral.com.sg' + val['href']
        if linkstr not in links:
            links.append(linkstr)

### 1.2 Append results into pandas dataframe
```
<iframe src="https://jerylcodes.github.io/proj3/table1/" height="400" width="600" overflow="auto"></iframe>

<img src="https://i.imgur.com/wLPdKgZ.png" style="float: left; margin: 25px 15px 0px 0px; height: 25px">

## 2. Analysing factors that affect salary
### 2.1 Generate meaningful features from text using term frequency–inverse document frequency (TFIDF)

Tf-idf stands for term frequency-inverse document frequency, and the tf-idf weight is a weight often used in information retrieval and text mining. This weight is a statistical measure used to evaluate how important a word is to a document in a collection or corpus. The importance increases proportionally to the number of times a word appears in the document but is offset by the frequency of the word in the corpus. Variations of the tf-idf weighting scheme are often used by search engines as a central tool in scoring and ranking a document's relevance given a user query.  

Source: [http://www.tfidf.com/](http://www.tfidf.com/)


```python
from sklearn.feature_extraction.text import TfidfVectorizer

vect = TfidfVectorizer(stop_words='english', ngram_range=(2,3),max_features=2000)
textvec = vect.fit_transform(X['full_description'])
```

__Example of features after tf-idf__

<iframe src="https://jerylcodes.github.io/proj3/table2/" height="200" width="600" overflow="auto"></iframe> 

### 2.2 Principal Component Analysis (PCA)  

tf-idf often produces large amounts of features. We may make use of dimensionality reduction techniques like Principal Component Analysis (PCA) to reduce the number of features needed for prediction or classification.


```python
from sklearn.decomposition import PCA
pca = PCA(n_components=15)
pca.fit(X1)

Explained variance shows how much variance in the dataset is accounted for in your principal components.


```python
print(pca.explained_variance_ratio_)  
print(sum(pca.explained_variance_ratio_))
```

    [0.17547903 0.1269256  0.08849329 0.05611374 0.03669752 0.02350112
     0.01906956 0.01444719 0.01286519 0.01149481 0.01019614 0.0093229
     0.00823705 0.00807786 0.00669269]
    0.607613709219762
    

### 2.3 Linear regression to predict salary (with lasso regularisation)


```python
lasso = linear_model.Lasso()
lasso.fit(X_train, y_train)
```




    Lasso(alpha=1.0, copy_X=True, fit_intercept=True, max_iter=1000,
       normalize=False, positive=False, precompute=False, random_state=None,
       selection='cyclic', tol=0.0001, warm_start=False)




```python
cross_val_score(lasso, X_train, y_train)
```




    array([0.33293736, 0.35060433, 0.27800117])




```python
y_pred = lasso.predict(X_test)

# The coefficients
print('Coefficients: \n', lasso.coef_)
# The mean squared error
print("Mean squared error: %.2f"
      % mean_squared_error(y_test, y_pred))
```

    Coefficients: 
     [ 1240.71748822  -462.59215024  2215.14691749  -225.41913544
     -2109.89030805  -327.22110887  -976.31850488   608.75956347
      -173.8406491   -525.20852812   625.66194878 -2623.4846474
     -3067.27596946 -1788.03523016  3521.19561575]
    Mean squared error: 2702897.93
    

<img src="https://i.imgur.com/wLPdKgZ.png" style="float: left; margin: 25px 15px 0px 0px; height: 25px">

## 3. Classifying data science and non-data science jobs
### 3.1 Generate value counts of job categories


```python
all_jobsraw.is_category.value_counts()
```




    engineer      818
    analyst       688
    leadership    624
    dont_care     375
    scientist     361
    intern         54
    database        1
    Name: is_category, dtype: int64



Binarise job categories into data science or non-data science.


```python
all_jobsraw['is_datasci'] = all_jobsraw['is_category'].map(lambda x: 1 if x == 'scientist' else 0)

```

### 3.2 Generate tf-idf matrix from job description


```python
df_text1 = pd.DataFrame(data=df_text.todense(), columns=vect.get_feature_names())
df_text1.head(2)
```

<iframe src="https://jerylcodes.github.io/proj3/table3/" height="200" width="600" overflow="auto"></iframe> 

### 3.3 Build a logistic regression and random forest classifier to classify the dataset
__Logistic regression__


```python
from sklearn.metrics import classification_report
target_names = ['non-science', 'science']
print(classification_report(y_test, y_pred, target_names=target_names))
```

                 precision    recall  f1-score   support
    
    non-science       0.96      0.94      0.95       770
        science       0.60      0.71      0.65       107
    
    avg / total       0.92      0.91      0.91       877
    
    

__Random forest classifier__


```python
target_names = ['non-science', 'science']
print(classification_report(y_test, y_pred, target_names=target_names))
```

                 precision    recall  f1-score   support
    
    non-science       0.95      0.94      0.95       770
        science       0.60      0.65      0.62       107
    
    avg / total       0.91      0.90      0.91       877
    
    
