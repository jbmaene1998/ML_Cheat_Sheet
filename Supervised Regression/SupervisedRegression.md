# Supervised Regression

## Imports we often need

<br>

```python
%matplotlib inline
import  numpy  as  np
import  pandas  as  pd
import  matplotlib  as  mpl
import  matplotlib.pyplot  as  plt
import  seaborn  as  sns
plt.style.use('seaborn-darkgrid')
```
<br>

## **Code**

<br>
   
|Code| Description|
|----|-------------|
|.sample(frac=1, random_state=123)| This is a test|
|train_test_split()| This is a test|
|.predict(X_train_stan) / .predict(X_test_stan)| This is a test|
|.score(X_train_stan, y_train) / .score(X_test_stan, y_test)| This is a test|
|mean_absolute_error(Any list, Any predicted_list)| This is a test|
|mean_squared_error(Any list, Any predicted_list)| This is a test|
|transform(Any list)| This is a test|
|LinearRegression(fit_intercept=Boolean bool)| This is a test|

<br>

## **Code snippets**

### Train or test-split
<br>

```python
#Import the function
from sklearn.model_selection import train_test_split

#Split of feaures and outcomes
X = df_shuffle.drop(['column_name'],1)
y = df_shuffle['column_name']

#Perform train/test-split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=int value, random_state=123)
```
<br>

```python
from sklearn.preprocessing import StandardScaler

num_feat = X_train.select_dtypes(include=[Datatype, Datatype]).columns

scaler = StandardScaler()
scaler.fit(X_train[num_feat])

X_train_stan = X_train.copy()
X_test_stan = X_test.copy()

X_train_stan[num_feat] = scaler.transform(X_train[num_feat])
X_test_stan[num_feat] = scaler.transform(X_test[num_feat])
```

<br>

```python
from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(degree=int value)
X_train_poly = poly.fit_transform(X_train_stan)
X_test_poly = poly.transform(X_test_stan)
```

<br>

```python
from sklearn.pipeline import Pipeline

from sklearn.model_selection import cross_val_score

avg_scores = [None] * 5

for i in np.arange(1,6):
    
    reg_poly = Pipeline([('poly', PolynomialFeatures(degree=i)),
                  ('linear', LinearRegression(fit_intercept=False))])
    
    scores = cross_val_score(reg_poly, X_train_stan, y_train, scoring = 'r2', cv=5)
    
    avg_scores[i-1] = scores.mean()
    
    print("Order "+str(i)+": avg R^2 = "+str( avg_scores[i-1]))
```


## **Checklist data cleaning**

<br>

- [ ] Loading packages and dataset.
- [ ] Check if the dataset has been cleaned.
- [ ] Shuffle dataset.
- [ ] Train-split and test-split.
- [ ] Standardize
- [ ] Train linear regression model\
- [ ] Predict outcome from trained model
- [ ] Calculate the coefficient of determination
- [ ] Calculate the Mean Absolute Error (MAE)
- [ ] Calculate the Mean Square Error (MSE)
- [ ] Is the model overfitted, underfitted or neither?
- [ ] Design polynomial features, don't forget to also transform the test data.
- [ ] Fit linear regression
- [ ] Find the optimal order for the polynomial using cross validation

