# Cheat sheet
## Data preparation

### Imports we often need

```python
%matplotlib inline
import  numpy  as  np
import  pandas  as  pd
import  matplotlib  as  mpl
import  matplotlib.pyplot  as  plt
import  seaborn  as  sns
plt.style.use('seaborn-darkgrid')
```

### Code

|Code| Description|
|----|-------------|
|.read_csv| Read the CSV file.|
|.show()|Used in Python's Matplotlib library to display the plot that has been created.|
|.shape| Get the shape of a dataset. |
|.info()| Get name, non-null count and data type for each column.|
|.head| Display the first n amount of columns, n=5 by default.|
|.astype(str column-name)| Convert column to specific or custom datatype.|
|.describe() / .describe(include='all')| Get a summary of the count, mean, standard deviation, minimum, 25th percentile, 50th percentile (median), 75th percentile, and maximum for each numeric column in a dataset. When using include='all' non-mumeric will be included in the calculations.|
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

- [ ] Check for duplicates. You can use .info() and check if the Non-Null Count has been chenged.
- [ ] Remove obeservations that you don't need.
- [ ] Remove features you don't need. You can use .drop() for this.
- [ ] Resolve typos in the features. For example you can use .capitalize().
- [ ] Group categorizable classes with the same meaning. You can use .astype(str column_name).
- [ ] Check for outliners, check if the values are impossible or there was a typing error.
- [ ] Check for missing values.
- [ ] Determine how to resolve missing values



