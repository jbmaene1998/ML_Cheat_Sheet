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
<div class="text-base gap-4 md:gap-6 m-auto md:max-w-2xl lg:max-w-2xl xl:max-w-3xl p-4 md:py-6 flex lg:px-0">
   <div class="w-[30px] flex flex-col relative items-end">
      <div class="relative h-[30px] w-[30px] p-1 rounded-sm text-white flex items-center justify-center" style="background-color: rgb(16, 163, 127);">
         <svg width="41" height="41" viewBox="0 0 41 41" fill="none" xmlns="http://www.w3.org/2000/svg" stroke-width="1.5" class="h-6 w-6">
            <path d="M37.5324 16.8707C37.9808 15.5241 38.1363 14.0974 37.9886 12.6859C37.8409 11.2744 37.3934 9.91076 36.676 8.68622C35.6126 6.83404 33.9882 5.3676 32.0373 4.4985C30.0864 3.62941 27.9098 3.40259 25.8215 3.85078C24.8796 2.7893 23.7219 1.94125 22.4257 1.36341C21.1295 0.785575 19.7249 0.491269 18.3058 0.500197C16.1708 0.495044 14.0893 1.16803 12.3614 2.42214C10.6335 3.67624 9.34853 5.44666 8.6917 7.47815C7.30085 7.76286 5.98686 8.3414 4.8377 9.17505C3.68854 10.0087 2.73073 11.0782 2.02839 12.312C0.956464 14.1591 0.498905 16.2988 0.721698 18.4228C0.944492 20.5467 1.83612 22.5449 3.268 24.1293C2.81966 25.4759 2.66413 26.9026 2.81182 28.3141C2.95951 29.7256 3.40701 31.0892 4.12437 32.3138C5.18791 34.1659 6.8123 35.6322 8.76321 36.5013C10.7141 37.3704 12.8907 37.5973 14.9789 37.1492C15.9208 38.2107 17.0786 39.0587 18.3747 39.6366C19.6709 40.2144 21.0755 40.5087 22.4946 40.4998C24.6307 40.5054 26.7133 39.8321 28.4418 38.5772C30.1704 37.3223 31.4556 35.5506 32.1119 33.5179C33.5027 33.2332 34.8167 32.6547 35.9659 31.821C37.115 30.9874 38.0728 29.9178 38.7752 28.684C39.8458 26.8371 40.3023 24.6979 40.0789 22.5748C39.8556 20.4517 38.9639 18.4544 37.5324 16.8707ZM22.4978 37.8849C20.7443 37.8874 19.0459 37.2733 17.6994 36.1501C17.7601 36.117 17.8666 36.0586 17.936 36.0161L25.9004 31.4156C26.1003 31.3019 26.2663 31.137 26.3813 30.9378C26.4964 30.7386 26.5563 30.5124 26.5549 30.2825V19.0542L29.9213 20.998C29.9389 21.0068 29.9541 21.0198 29.9656 21.0359C29.977 21.052 29.9842 21.0707 29.9867 21.0902V30.3889C29.9842 32.375 29.1946 34.2791 27.7909 35.6841C26.3872 37.0892 24.4838 37.8806 22.4978 37.8849ZM6.39227 31.0064C5.51397 29.4888 5.19742 27.7107 5.49804 25.9832C5.55718 26.0187 5.66048 26.0818 5.73461 26.1244L13.699 30.7248C13.8975 30.8408 14.1233 30.902 14.3532 30.902C14.583 30.902 14.8088 30.8408 15.0073 30.7248L24.731 25.1103V28.9979C24.7321 29.0177 24.7283 29.0376 24.7199 29.0556C24.7115 29.0736 24.6988 29.0893 24.6829 29.1012L16.6317 33.7497C14.9096 34.7416 12.8643 35.0097 10.9447 34.4954C9.02506 33.9811 7.38785 32.7263 6.39227 31.0064ZM4.29707 13.6194C5.17156 12.0998 6.55279 10.9364 8.19885 10.3327C8.19885 10.4013 8.19491 10.5228 8.19491 10.6071V19.808C8.19351 20.0378 8.25334 20.2638 8.36823 20.4629C8.48312 20.6619 8.64893 20.8267 8.84863 20.9404L18.5723 26.5542L15.206 28.4979C15.1894 28.5089 15.1703 28.5155 15.1505 28.5173C15.1307 28.5191 15.1107 28.516 15.0924 28.5082L7.04046 23.8557C5.32135 22.8601 4.06716 21.2235 3.55289 19.3046C3.03862 17.3858 3.30624 15.3413 4.29707 13.6194ZM31.955 20.0556L22.2312 14.4411L25.5976 12.4981C25.6142 12.4872 25.6333 12.4805 25.6531 12.4787C25.6729 12.4769 25.6928 12.4801 25.7111 12.4879L33.7631 17.1364C34.9967 17.849 36.0017 18.8982 36.6606 20.1613C37.3194 21.4244 37.6047 22.849 37.4832 24.2684C37.3617 25.6878 36.8382 27.0432 35.9743 28.1759C35.1103 29.3086 33.9415 30.1717 32.6047 30.6641C32.6047 30.5947 32.6047 30.4733 32.6047 30.3889V21.188C32.6066 20.9586 32.5474 20.7328 32.4332 20.5338C32.319 20.3348 32.154 20.1698 31.955 20.0556ZM35.3055 15.0128C35.2464 14.9765 35.1431 14.9142 35.069 14.8717L27.1045 10.2712C26.906 10.1554 26.6803 10.0943 26.4504 10.0943C26.2206 10.0943 25.9948 10.1554 25.7963 10.2712L16.0726 15.8858V11.9982C16.0715 11.9783 16.0753 11.9585 16.0837 11.9405C16.0921 11.9225 16.1048 11.9068 16.1207 11.8949L24.1719 7.25025C25.4053 6.53903 26.8158 6.19376 28.2383 6.25482C29.6608 6.31589 31.0364 6.78077 32.2044 7.59508C33.3723 8.40939 34.2842 9.53945 34.8334 10.8531C35.3826 12.1667 35.5464 13.6095 35.3055 15.0128ZM14.2424 21.9419L10.8752 19.9981C10.8576 19.9893 10.8423 19.9763 10.8309 19.9602C10.8195 19.9441 10.8122 19.9254 10.8098 19.9058V10.6071C10.8107 9.18295 11.2173 7.78848 11.9819 6.58696C12.7466 5.38544 13.8377 4.42659 15.1275 3.82264C16.4173 3.21869 17.8524 2.99464 19.2649 3.1767C20.6775 3.35876 22.0089 3.93941 23.1034 4.85067C23.0427 4.88379 22.937 4.94215 22.8668 4.98473L14.9024 9.58517C14.7025 9.69878 14.5366 9.86356 14.4215 10.0626C14.3065 10.2616 14.2466 10.4877 14.2479 10.7175L14.2424 21.9419ZM16.071 17.9991L20.4018 15.4978L24.7325 17.9975V22.9985L20.4018 25.4983L16.071 22.9985V17.9991Z" fill="currentColor"></path>
         </svg>
      </div>
      <div class="text-xs flex items-center justify-center gap-1 invisible absolute left-0 top-2 -ml-4 -translate-x-full group-hover:visible">
         <button class="dark:text-white disabled:text-gray-300 dark:disabled:text-gray-400">
            <svg stroke="currentColor" fill="none" stroke-width="1.5" viewBox="0 0 24 24" stroke-linecap="round" stroke-linejoin="round" class="h-3 w-3" height="1em" width="1em" xmlns="http://www.w3.org/2000/svg">
               <polyline points="15 18 9 12 15 6"></polyline>
            </svg>
         </button>
         <span class="flex-grow flex-shrink-0">2 / 2</span>
         <button disabled="" class="dark:text-white disabled:text-gray-300 dark:disabled:text-gray-400">
            <svg stroke="currentColor" fill="none" stroke-width="1.5" viewBox="0 0 24 24" stroke-linecap="round" stroke-linejoin="round" class="h-3 w-3" height="1em" width="1em" xmlns="http://www.w3.org/2000/svg">
               <polyline points="9 18 15 12 9 6"></polyline>
            </svg>
         </button>
      </div>
   </div>
   <div class="relative flex w-[calc(100%-50px)] flex-col gap-1 md:gap-3 lg:w-[calc(100%-115px)]">
      <div class="flex flex-grow flex-col gap-3">
         <div class="min-h-[20px] flex flex-col items-start gap-4 whitespace-pre-wrap">
            <div class="markdown prose w-full break-words dark:prose-invert light">
               <p>Sure, I apologize for the confusion. Here are some updated examples that should be easier to understand:</p>
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
