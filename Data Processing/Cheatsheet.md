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
   <div class="relative flex w-[calc(100%-50px)] flex-col gap-1 md:gap-3 lg:w-[calc(100%-115px)]">
      <div class="flex flex-grow flex-col gap-3">
         <div class="min-h-[20px] flex flex-col items-start gap-4 whitespace-pre-wrap">
            <div class="markdown prose w-full break-words dark:prose-invert light">
               <table>
                  <thead>
                     <tr>
                        <th>Code</th>
                        <th>Description</th>
                        <th>Example</th>
                     </tr>
                  </thead>
                  <tbody>
                     <tr>
                        <td>.read_csv</td>
                        <td>Read the CSV file.</td>
                        <td>df = pd.read_csv('data.csv')</td>
                     </tr>
                     <tr>
                        <td>.show()</td>
                        <td>Used in Python's Matplotlib library to display the plot that has been created.</td>
                        <td>plt.show()</td>
                     </tr>
                     <tr>
                        <td>.shape</td>
                        <td>Get the shape of a dataset.</td>
                        <td>df.shape # returns the number of rows and columns in the dataset</td>
                     </tr>
                     <tr>
                        <td>.info()</td>
                        <td>Get name, non-null count and data type for each column.</td>
                        <td>df.info() # prints information about the columns in the dataset, including the column name, number of non-null values, and data type</td>
                     </tr>
                     <tr>
                        <td>.head()</td>
                        <td>Display the first n amount of columns, n=5 by default.</td>
                        <td>df.head(10) # displays the first 10 rows of the dataset</td>
                     </tr>
                     <tr>
                        <td>.astype(str column-name)</td>
                        <td>Convert column to specific or custom datatype.</td>
                        <td>df['column_name'] = df['column_name'].astype(str) # changes the data type of a column to string</td>
                     </tr>
                     <tr>
                        <td>.describe() / .describe(include='all')</td>
                        <td>Get a summary of the count, mean, standard deviation, minimum, 25th percentile, 50th percentile (median), 75th percentile, and maximum for each numeric column in a dataset. When using include='all' non-numeric will be included in the calculations.</td>
                        <td>df.describe() # calculates summary statistics for numeric columns in the dataset</td>
                     </tr>
                     <tr>
                        <td>.hist() / .hist( figsize = (int width, int heigth))</td>
                        <td>Show a frequency distribution of the data. The parameter figsze is used to determine the size of the diagram.</td>
                        <td>df['column_name'].hist() # displays a histogram of the values in a column</td>
                     </tr>
                     <tr>
                        <td>.countplot(y=df['column_name'])</td>
                        <td>Show a frequency count of each unique category of column.</td>
                        <td>sns.countplot(y=df['column_name']) # displays a bar chart of the number of occurrences of each unique category in a column</td>
                     </tr>
                     <tr>
                        <td>.set()</td>
                        <td>Sets the default style to Seaborn's theme.</td>
                        <td>sns.set() # sets the default style to Seaborn's theme</td>
                     </tr>
                     <tr>
                        <td>.copy()</td>
                        <td>Create a copy of the current object.</td>
                        <td>df_copy = df.copy() # creates a copy of a dataframe</td>
                     </tr>
                     <tr>
                        <td>.drop()</td>
                        <td>Drop a column from a dataset.</td>
                        <td>df.drop('column_name', axis=1, inplace=True) # removes a column from a dataframe</td>
                     </tr>
                     <tr>
                        <td>.pairplot(DataFrame df, y_vars=['column_name'],x=vars[int value1: value2])</td>
                        <td>Used to create a matrix of scatter plots that show the relationships between multiple variables in a dataset. y_vars=['column_name'] is the main variable and x=vars[int value1: value2] will be scatterd based y_vars=['column_name'].</td>
                        <td>sns.pairplot(df, y_vars=['column_name'], x_vars=['column_name_1', 'column_name_2']) # creates a scatter plot matrix for the relationships between multiple variables in a dataset</td>
                     </tr>
                     <tr>
                        <td>.corr()</td>
                        <td>Used to calculate the correlation between columns in a DataFrame.</td>
                        <td>df.corr() # calculates the correlation between numeric columns in a dataframe</td>
                     </tr>
                     <tr>
                        <td>.drop_duplicates()</td>
                        <td>Remove duplicates from a DataFrame.</td>
                        <td>df.drop_duplicates() # removes duplicate rows from a dataframe</td>
                     </tr>
                     <tr>
                        <td>.capitalize()</td>
                        <td>Capitalize the first letter of a word.</td>
                        <td>str.capitalize('hello world') # capitalizes the first letter of a string</td>
                     </tr>
                     <tr>
                        <td>.isnull()</td>
                        <td>Returns if there are Null values in the DataFrame</td>
                        <td>df.isnull() # returns a boolean dataframe indicating which values in a dataframe are null</td>
                     </tr>
                     <tr>
                        <td>.sum()</td>
                        <td>Count the values of the DataFrame</td>
                        <td>df.sum() # calculates the sum of values in a dataframe</td>
                     </tr>
                     <tr>
                        <td>.reshape()</td>
                        <td>Changing the layout or structure of the data, while keeping the content of the data intact</td>
                        <td></td>
                     </tr>
                        <tr>
                        <td>.fit(Any df)</td>
                        <td>Method used to train a machine learning model on a dataset.</td>
                        <td></td>
                     </tr>
                  </tbody>
               </table>
            </div>
         </div>
      </div>
      <div class="flex justify-between">
         <div class="text-gray-400 flex self-end lg:self-center justify-center mt-2 gap-3 md:gap-4 lg:gap-1 lg:absolute lg:top-0 lg:translate-x-full lg:right-0 lg:mt-0 lg:pl-2 visible">
            <button class="p-1 rounded-md hover:bg-gray-100 hover:text-gray-700 dark:text-gray-400 dark:hover:bg-gray-700 dark:hover:text-gray-200 disabled:dark:hover:text-gray-400">
               <svg stroke="currentColor" fill="none" stroke-width="2" viewBox="0 0 24 24" stroke-linecap="round" stroke-linejoin="round" class="h-4 w-4" height="1em" width="1em" xmlns="http://www.w3.org/2000/svg">
                  <path d="M14 9V5a3 3 0 0 0-3-3l-4 9v11h11.28a2 2 0 0 0 2-1.7l1.38-9a2 2 0 0 0-2-2.3zM7 22H4a2 2 0 0 1-2-2v-7a2 2 0 0 1 2-2h3"></path>
               </svg>
            </button>
            <button class="p-1 rounded-md hover:bg-gray-100 hover:text-gray-700 dark:text-gray-400 dark:hover:bg-gray-700 dark:hover:text-gray-200 disabled:dark:hover:text-gray-400">
               <svg stroke="currentColor" fill="none" stroke-width="2" viewBox="0 0 24 24" stroke-linecap="round" stroke-linejoin="round" class="h-4 w-4" height="1em" width="1em" xmlns="http://www.w3.org/2000/svg">
                  <path d="M10 15v4a3 3 0 0 0 3 3l4-9V2H5.72a2 2 0 0 0-2 1.7l-1.38 9a2 2 0 0 0 2 2.3zm7-13h2.67A2.31 2.31 0 0 1 22 4v7a2.31 2.31 0 0 1-2.33 2H17"></path>
               </svg>
            </button>
         </div>
      </div>
   </div>
</div>

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

#### Replace NaN values with undefined:
```python
categorical = ['column_name1','column_name2', 'column_name3']


for col in categorical:
    df_imputed[col] = df_imputed[col].astype('category')
    df_imputed[col] = df_imputed[col].cat.add_categories('Undefined')
    df_imputed[col] = df_imputed[col].fillna('Undefined')
    df_imputed[col] = df_imputed[col].astype('object')
```


<br>

#### Merge multiple categories in 1 category
```python
df_sparse['column_name'][df_sparse['column_name'] == 'old_category'] = 'new_category'
df_sparse['column_name'][df_sparse['column_name'] == 'old_category'] = 'new_category'
df_sparse['column_name'][df_sparse['column_name'] == 'old_category'] = 'new_category'
```
<br>

<br>

#### Merge multiple categories in 1 category based on a percentage
```python
threshold_percent = int value

series = pd.value_counts(df_sparse['column_name'])
mask = (series / series.sum() * 100).lt(threshold_percent)
df_sparse['column_name']= np.where(df_sparse['column_name'].isin(series[mask].index),'new_category', df_sparse['column_name'])
df_sparse['column_name'].value_counts()
```
<br>

#### **Checklist data cleaning**

<br>


- [ ] Check for duplicates. You can use .info() and check if the Non-Null Count has been changed.
- [ ] Remove obeservations that you don't need.
- [ ] Remove features you don't need. You can use .drop() for this.
- [ ] Resolve typos in the features. For example you can use .capitalize().
- [ ] Group categorizable classes with the same meaning. You can use .astype(str column_name).
- [ ] Check for outliners, check if the values are impossible or there was a typing error.
- [ ] Check for missing values.
- [ ] Determine how to resolve missing values
- [ ] Add dummy values if necessary
