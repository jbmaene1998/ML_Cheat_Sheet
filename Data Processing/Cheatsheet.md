# Cheat sheet
## Data preparation

### Imports we often need

<br>

```python
%matplotlib inline
import  numpy  as  np
import  pandas  as  pd
import  matplotlib  as  mpl
import  matplotlib.pyplot  as  plt
import  seaborn  as  sns
plt.style.use('seaborn-darkgrid')

from sklearn.impute import SimpleImputer
from scipy.stats import skew
```
<br>

### **Code**

<br>

|Code| Description|
|----|-------------|
|.read_csv| Read the CSV file.|
|.show()|Used in Python's Matplotlib library to display the plot that has been created.|
|.shape| Get the shape of a dataset. |
|.info()| Get name, non-null count and data type for each column.|
|.head| Display the first n amount of columns, n=5 by default.|
|.astype(str column-name)| Convert column to specific or custom datatype.|
|.describe() / .describe(include='all')| Get a summary of the count, mean, standard deviation, minimum, 25th percentile, 50th percentile (median), 75th percentile, and maximum for each numeric column in a dataset. When using include='all' non-numeric will be included in the calculations.|
|.hist() / .hist( figsize = (int width, int heigth))| Show a frequency distribution of the data. The parameter figsze is used to determine the size of the diagram.|
|.countplot(y=df['column_name'])| Show a frequency count of each unique category of column.|
|.set()| Sets the default style to Seaborn's theme.|
|.copy()| Create a copy of the current object.|
|.drop()| Drop a column from a dataset.|
|.pairplot(DataFrame df, y_vars=['column_name'],x=vars[int value1: value2])| Used to create a matrix of scatter plots that show the relationships between multiple variables in a dataset. y_vars=['column_name'] is the main variable and x=vars[int value1: value2] will be scatterd based y_vars=['column_name'].|
|.corr()| Used to calculate the correlation between columns in a DataFrame.|
|.drop_duplicates()| Remove duplicates from a DataFrame.|
|.capitalize()| Capitalize the first letter of a word.|
|.isnull| Returns if there are Null values in the DataFrame|
|.sum| Count the values of the DataFrame|
|.reshape()| Changing the layout or structure of the data, while keeping the content of the data intact|
|.fit(Any df)| Method used to train a machine learning model on a dataset.|

<br>
<br>

### **Code snippets**

<br>

#### how to replace missing values:


```python
df_imputed = df.copy()

# import Imputer 
from sklearn.impute import SimpleImputer

# Create an imputer object that looks for 'Nan' values, then replaces them with a descriptive statistic value of the feature by columns (axis=0)
stat_imputer = SimpleImputer(missing_values=np.nan, strategy='statistical_method')

# Train the imputor on the dataset
stat_imputer = stat_imputer.fit(np.array(df['column_name']).reshape(int value1, int value2) )

# Apply the imputer to the dataset (This imputer can also be used on future datasets)
df_imputed['column_name'] = stat_imputer.transform(np.array(df['column_name']).reshape(int value1, int value2) )
```

<br>

#### Create dummy variables (categorical features):
```python
## I make sure all three categorical features are classified as 'object' to be able to check if they are categorical
df_imputed['column_name']= df_imputed['column_name'].astype('object') 

for col in df_imputed:
    if df_imputed[col].dtype ==  'DataType':
        dummies = pd.get_dummies(df_imputed[col], dummy_na=Boolean boolean, prefix=col)  #create dummies
        df_imputed = pd.concat([df_imputed, dummies],axis=1)                             # add dummies to dataset
        df_imputed.drop(columns=[col], inplace=Boolean boolean)                                     # delete original feature
```


<br>

#### **Checklist data cleaning**

<br>


- [ ] Check for duplicates. You can use .info() and check if the Non-Null Count has been chenged.
- [ ] Remove obeservations that you don't need.
- [ ] Remove features you don't need. You can use .drop() for this.
- [ ] Resolve typos in the features. For example you can use .capitalize().
- [ ] Group categorizable classes with the same meaning. You can use .astype(str column_name).
- [ ] Check for outliners, check if the values are impossible or there was a typing error.
- [ ] Check for missing values.
- [ ] Determine how to resolve missing values
- [ ] Add dummy values in neccesairy
