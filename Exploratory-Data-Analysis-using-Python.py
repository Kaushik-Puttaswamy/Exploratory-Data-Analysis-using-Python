#!/usr/bin/env python
# coding: utf-8

# In[282]:


## Install and import the Python libraries 
# Importing the libraries needed for the project 
''' 
Python libraries are used by installing and importing them. 
These libraries are required for operations such as reading, manipulating, and preparing data, as well as visualising it. 
They also support code testing, database access, and warning handling.
''' 
# Library for testing
import unittest

# Library for accessing database
from sqlalchemy import create_engine

# Libraries for reading and manipulating data 
import pandas as pd  # pandas is used for data manipulation and analysis
import numpy as np   # numpy is essential for numerical computing
from sklearn.metrics import r2_score # r2_score function from scikit-learn for calculating R-squared
from scipy.stats import linregress  # linregress function from SciPy for linear regression analysis

# Libraries for data visualization 
import matplotlib.pyplot as plt #plotting library for creating static, interactive, and animated visualizations
import seaborn as sns # seaborn is a high-level interface built on top of matplotlib, providing additional functionalities and making it easier to create attractive statistical graphics
from bokeh.models import Label#for interactive data visualization that targets modern web browsers. Label is a Bokeh model used to add text annotations to plots
from bokeh.plotting import figure, show# bokeh.plotting provides a high-level interface for creating visualizations, and show is used to display Bokeh plots in the output
from bokeh.models import ColumnDataSource# ColumnDataSource is a Bokeh data structure that allows for efficient and convenient data handling for Bokeh plots
from bokeh.io import output_notebook#for data exploration, analysis, and presentation in a collaborative and interactive environment
# Library for warning: 'warnings' module
import warnings  # Import the warnings module
warnings.filterwarnings('ignore')  # Set a filter to ignore warning messages
# Enable inline plotting for matplotlib in Jupyter Notebook
get_ipython().run_line_magic('matplotlib', 'inline')


# In[283]:


# Set the file paths for the datasets
train_data_file = r"C:\Users\HP\OneDrive\Desktop\dataset\train.csv"
ideal_data_file = r"C:\Users\HP\OneDrive\Desktop\dataset\ideal.csv"
test_data_file = r"C:\Users\HP\OneDrive\Desktop\dataset\test.csv"

# Load training data
train_data = pd.read_csv(train_data_file)

# Load ideal functions
ideal_data = pd.read_csv(ideal_data_file)

# Load test data
test_data = pd.read_csv(test_data_file)

# Remove duplicate X values from test data and calculate the mean of corresponding Y values
test_data = test_data.groupby('x').mean().reset_index()

# Specify the absolute path to the database file
database_path = r"C:\Users\HP\OneDrive\Desktop\mydatabase.db"

# Create a SQLite database using SQLAlchemy with the modified path
engine = create_engine(f'sqlite:///{database_path}', echo=True)

# Create tables in the database
with engine.begin() as connection:
    train_data.to_sql('train_data', con=connection, if_exists='replace', index=False)
    ideal_data.to_sql('ideal_data', con=connection, if_exists='replace', index=False)
    test_data.to_sql('test_data', con=connection, if_exists='replace', index=False)

# Confirm successful database creation
print("The database was successfully created, and the data was loaded!")


# In[284]:


#Display the created table names in the database
def display_table_names():
    """
    Connects to an SQLite database engine and displays the table names in the database.

    Returns:
    None
    """

    # Create a SQLite database engine
    engine = create_engine(r'sqlite:///C:\Users\HP\OneDrive\Desktop\mydatabase.db', echo=True)


    # Get the table names from the database
    table_names = engine.table_names()

    # Display the table names
    for table_name in table_names:
        print(table_name)


# Call the function to display table names
display_table_names()


# In[151]:


#display the contents of the ideal_data table from a SQLite database
def ideal_data_table():
    """
    Fetches and displays the contents of the ideal_data table from a SQLite database.
    """
    # Create a SQLite database engine
    engine = create_engine('sqlite:///C:\\Users\\HP\\OneDrive\\Desktop\\mydatabase.db', echo=True)

    # Fetch and display the contents of the ideal_data table
    query = "SELECT * FROM ideal_data"
    ideal_data = pd.read_sql_query(query, engine)
    print("Contents of ideal_data table:")
    print(ideal_data)

ideal_data_table()


# In[152]:


#displays the contents of the train_data table from a SQLite database
def train_data_table():
    """
    Fetches and displays the contents of the train_data table from a SQLite database.
    """
    # Create a SQLite database engine
    engine = create_engine('sqlite:///C:\\Users\\HP\\OneDrive\\Desktop\\mydatabase.db', echo=True)

    # Fetch and display the contents of the train_data table
    query = "SELECT * FROM train_data"
    train_data = pd.read_sql_query(query, engine)
    print("Contents of train_data table:")
    print(train_data)

train_data_table()


# In[153]:


#displays the contents of the test_data table from a SQLite database
def test_data_table():
    """
    Fetches and displays the contents of the test_data table from a SQLite database.
    """
    # Create a SQLite database engine
    engine = create_engine('sqlite:///C:\\Users\\HP\\OneDrive\\Desktop\\mydatabase.db', echo=True)

    # Fetch and display the contents of the test_data table
    query = "SELECT * FROM test_data"
    test_data = pd.read_sql_query(query, engine)
    print("Contents of test_data table:")
    print(test_data)

test_data_table()


# In[154]:


# Find number of rows and columns in idea data
print("Number of rows in ideal_data set:", len(ideal_data))
print("Number of columns in ideal_data set:", len(ideal_data.columns))


# In[155]:


# check first five rows in ideal data 
first_five_rows = ideal_data.head(5)
print(first_five_rows)


# In[156]:


# check last five rows in ideal data 
last_five_rows = ideal_data.tail(5)
print(last_five_rows)


# In[157]:


ideal_data.info() 


# In[158]:


# Describe ideal data frame transpose 
ideal_data.describe().T


# In[159]:


#define a function to visualize ideal data using box plot
def visualize_ideal_data_boxplot(data):
    """
    Visualize multiple variables of an ideal dataset using a boxplot.

    Parameters:
        data (DataFrame): The ideal dataset.

    Returns:
        None
    """
    plt.figure(figsize=(15, 10))
    sns.boxplot(data=data, palette="Set1")
    plt.xlabel("Variables")  # x-axis label 
    plt.ylabel("Values")  # y-axis label
    plt.title("Boxplot of Ideal Dataset")  #title
    plt.show()

# Call the function to visualize the boxplot of the ideal_data dataset
visualize_ideal_data_boxplot(ideal_data)


# In[57]:


#display the correlation heatmap of ideal dataset
plt.figure(figsize=(50, 20))  # Set the size of the figure for optimum view

# Calculate the correlation matrix
correlation_matrix = ideal_data.corr()

# Create the heatmap
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f")

# Add a title
plt.title("Correlation Heatmap of Ideal Dataset")

# Display the plot
plt.show()


# In[36]:


# Find number of rows and columns in train data
print("Number of rows in train_data set:", len(train_data))
print("Number of columns in train_data set:", len(train_data.columns))


# In[37]:


# check first five rows in train data
train_data.head(5)


# In[45]:


# check last five rows in train data
train_data.tail(5)


# In[38]:


# Info of each Train dataset 
train_data.info()


# In[39]:


# Describe train data 
train_data.describe().T


# In[40]:


#define a function to plot individual sactter plot with regression line for train data variable
def plot_scatter_with_regression(train_data, x_column, y_column, title):
    """
    Plot a scatter plot with a regression line.

    Parameters:
        train_data (pd.DataFrame): DataFrame containing the train data.
        x_column (str): Name of the x-axis column.
        y_column (str): Name of the y-axis column.
        title (str): Title of the plot.

    Returns:
        None
    """
    fig, ax = plt.subplots()
    sns.scatterplot(x=x_column, y=y_column, data=train_data, color='blue', ax=ax)
    sns.regplot(x=x_column, y=y_column, data=train_data, scatter=False, line_kws={'color': 'red'}, ax=ax)
    ax.legend(labels=['Train Data Points', 'Regression Line'], loc='upper right')
    plt.title(title)
    plt.show()

# Plot scatter plots with regression lines
plot_scatter_with_regression(train_data, 'x', 'y1', 'Scatter Plot for y1 with regression line')
plot_scatter_with_regression(train_data, 'x', 'y2', 'Scatter Plot for y2 with regression line')
plot_scatter_with_regression(train_data, 'x', 'y3', 'Scatter Plot for y3 with regression line')
plot_scatter_with_regression(train_data, 'x', 'y4', 'Scatter Plot for y4 with regression line')


# In[41]:


# Boxplot of train dataset
''' 
Visualize more than two variables 
of a train dataset 
'''
fig, ax = plt.subplots(figsize=(12, 6))
sns.boxplot(data=train_data, palette="Paired", ax=ax)
plt.xlabel("Variables")
plt.ylabel("Values")
plt.title("Boxplot of Train Dataset")
plt.xticks(rotation=45)
plt.show()


# In[42]:


# Single graph for complete train dataset considering a pair of variables at a time
sns.pairplot(train_data) 
plt.suptitle("Pairwise Relationships in Train Dataset", y=1.02)
plt.show()


# In[43]:


# Find number of rows and columns in test data after removing 10 duplicate rows
print("Number of rows in test_data set:", len(test_data))
print("Number of columns in test_data set:", len(test_data.columns))


# In[44]:


# check first five rows in test data
test_data.head()


# In[46]:


# check last five rows in test data
test_data.tail()


# In[47]:


# Info of each test dataset 
test_data.info()


# In[48]:


# Describe test data 
test_data.describe().T


# In[49]:


#define a function to plot box plot for test data
def hist_box(test_data, col):
    """
    Generate a combination of histogram and boxplot for a given column in the test dataset.

    Parameters:
        test_df (pd.DataFrame): DataFrame containing the test data.
        col (str): Name of the column to visualize.

    Returns:
        None
    """
    f, (ax_box, ax_hist) = plt.subplots(2, sharex=True, gridspec_kw={'height_ratios': (0.10, 0.70)},
                                         figsize=(10, 8))

    # Adding a boxplot in the test dataset
    sns.boxplot(test_data[col], ax=ax_box, showmeans=True)
    ax_box.set(xlabel='')

    # Adding a histogram in the test dataset
    sns.histplot(test_data[col], ax=ax_hist, kde=True)
    ax_hist.set(xlabel=col)

    plt.tight_layout()
    plt.show()


# Visualize hist_box for 'x' variable in test dataset
hist_box(test_data, 'x')


# In[50]:


# Visualize hist_box for 'y' variable in test dataset
hist_box(test_data, 'y')


# In[53]:


# Boxplot of test dataset for both x and y variable
plt.figure(figsize=(10, 6))
sns.boxplot(data=test_data, palette="Set1")
plt.title('Boxplot of the Test Dataset for both x and y variable (duplicate removed)')
plt.xticks(rotation=45)
plt.show()


# In[52]:


# Single graph for test dataset considering a pair of variables at a time
sns.pairplot(test_data) 
plt.suptitle("Pairwise Relationships in Test Dataset", y=1.02)
plt.show()


# In[285]:


#Calculating leat square deviation between ideal data and train data 
#Selecting best fit four ideal function for individual train data which shows sum of minimum least sqaure deviation using OOP concept

class DeviationAnalysis:
    """
    A class for calculating and analyzing the sum of least square deviations between train_data and ideal_data.
    """
    def __init__(self, train_data, ideal_data, y_index):
        """
        Initializes DeviationAnalysis object.

        Parameters:
            train_data (pd.DataFrame): The training data DataFrame.
            ideal_data (pd.DataFrame): The ideal data DataFrame.
            y_index (int): The index of the column to analyze.
        """
        self.train_data = train_data
        self.ideal_data = ideal_data
        self.y_index = y_index
        self.column_name = f"y{y_index}"

    def calculate_lsd(self):
        """
        Calculates the sum of least square deviations for each column.

        Returns:
            list: A list containing the sum of least square deviations for each column.
        """
        lsd_sum = []
        for i in range(1, 51):
            column_name = f"y{i}"
            lsd = ((self.train_data[self.column_name] - self.ideal_data[column_name]) ** 2).sum()
            lsd_sum.append(lsd)
        return lsd_sum

    def plot_graph(self, lsd_sum):
        """
        Plots a bar graph showing the sum of least square deviations for each column.

        Parameters:
            lsd_sum (list): A list containing the sum of least square deviations for each column.

        Returns:
            tuple: A tuple containing the minimum LSD value and its corresponding column index.
        """
        min_lsd_index = np.argmin(lsd_sum)
        min_lsd_value = lsd_sum[min_lsd_index]

        plt.figure(figsize=(12, 6))

        x = np.arange(1, 51)
        colors = ['red' if i == min_lsd_index else 'blue' for i in range(len(lsd_sum))]

        plt.bar(x, lsd_sum, color=colors)
        plt.xlabel('Column')
        plt.ylabel('Sum of Least Square Deviation (Log Scale)')
        plt.title(f'Sum of Least Square Deviation for Y1 to Y50 (Log Scale) - {self.column_name} Train Data')
        plt.xticks(x, [f"y{i}" for i in range(1, 51)], rotation='vertical')
        plt.yscale('log')

        min_lsd_label = f"Min LSD: {min_lsd_value:.2f}"
        plt.text(0.02, 0.98, min_lsd_label, transform=plt.gca().transAxes,
                 ha='left', va='top', color='red', bbox=dict(facecolor='white', edgecolor='black'))

        plt.tight_layout()
        plt.show()

        return min_lsd_value, min_lsd_index

    def run(self):
        """
        Runs the deviation analysis and plotting process.
        """
        try:
            lsd_sum = self.calculate_lsd()
        except Exception as e:
            print("An error occurred while calculating LSD:", e)
        else:
            try:
                min_lsd_value, min_lsd_index = self.plot_graph(lsd_sum)
            except Exception as e:
                print("An error occurred while plotting the graph:", e)
            else:
                min_lsd_column = f"y{min_lsd_index + 1}"
                print(f"The minimum sum of least square deviation is {min_lsd_value}")
                print(f"The ideal function for the train data {self.column_name} is: {min_lsd_column}")
        finally:
            print('This is always executed at the end of the run method')

class LeastSquareDeviation(DeviationAnalysis):
    """
    A class for calculating and analyzing the sum of least square deviations between train_data and ideal_data.
    Inherits from DeviationAnalysis.
    """
    def __init__(self, train_data, ideal_data, y_index):
        """
        Initializes LeastSquareDeviation object.

        Parameters:
            train_data (pd.DataFrame): The training data DataFrame.
            ideal_data (pd.DataFrame): The ideal data DataFrame.
            y_index (int): The index of the column to analyze.
        """
        super().__init__(train_data, ideal_data, y_index)

# Least Square Deviation for Multiple Data Columns
for y_index in range(1, 5):
    lsd = LeastSquareDeviation(train_data, ideal_data, y_index)
    lsd.run()


# In[251]:


#Unit testing deviation analysis code to check the correctness of least square method
class DeviationAnalysis:
    def __init__(self, train_data, ideal_data, y_index):
        """
        Initializes the DeviationAnalysis class with the given data and index.

        Parameters:
            train_data (dict): A dictionary containing train data for different columns.
            ideal_data (dict): A dictionary containing ideal data for different columns.
            y_index (int): The index of the column to analyze (1 to n).

        Returns:
            None
        """
        self.train_data = train_data
        self.ideal_data = ideal_data
        self.y_index = y_index
        self.column_name = f"y{y_index}"

    def calculate_lsd(self):
        """
        Calculates the Least Squared Deviation (LSD) for the specified column index.

        The LSD measures the difference between the train data and the ideal data for the specified column.

        Parameters:
            None

        Returns:
            list: A list containing the LSD values for each column in the range from 1 to 4 (inclusive).
        """
        lsd_sum = []
        for i in range(1, 5):  # Update the range based on the available columns in ideal_data
            column_name = f"y{i}"
            train_values = np.array(self.train_data[column_name])
            ideal_values = np.array(self.ideal_data[column_name])
            lsd = np.sum((train_values - ideal_values) ** 2)
            lsd_sum.append(lsd)
        return lsd_sum

class DeviationAnalysis_Test(unittest.TestCase):
    def test_calculate_lsd(self):
        """
        Test case for the calculate_lsd method of DeviationAnalysis.

        Compares the calculated LSD values with the expected values based on the number of columns.

        Parameters:
            None

        Returns:
            None
        """
        train_data = {
            'y1': [2, 3, 4],
            'y2': [5, 6, 7],
            'y3': [8, 9, 10],
            'y4': [11, 12, 13],
        }
        ideal_data = {
            'y1': [3, 4, 5],
            'y2': [6, 7, 8],
            'y3': [9, 10, 11],
            'y4': [12, 13, 14],
        }
        y_index = 1
        deviation_analysis = DeviationAnalysis(train_data, ideal_data, y_index)
        lsd_sum = deviation_analysis.calculate_lsd()

        expected_lsd_sum = [3, 3, 3, 3]  #  expected values based on the number of columns

        self.assertEqual(lsd_sum, expected_lsd_sum)

if __name__ == '__main__':
    unittest.main(argv=[''], exit=False)


# In[286]:


#Plotting a scatter plot for train data with the corresponding selected regression line of ideal data

def plot_regression_scatter(x, y, ideal_func_data, title, x_label, y_label):
    """
    Plot a scatter plot for train data with corresponding regression lines of ideal data.

    Parameters:
        x (numpy.ndarray): Array containing the input features (x values) for the scatter plot.
        y (numpy.ndarray): Array containing the corresponding target values (y values) for the scatter plot.
        ideal_func_data (numpy.ndarray): Array containing the y values of the ideal function used for the ideal regression line.
        title (str): Title of the plot.
        x_label (str): Label for the x-axis.
        y_label (str): Label for the y-axis.

    Returns:
        None. Displays the plot using Bokeh's show() function.
    """

    # Create a new figure
    p = figure(title=title, x_axis_label=x_label, y_axis_label=y_label)

    # Scatter plot
    p.circle(x, y, legend_label='Train Data', color='blue')

    # Regression line for train data
    train_coeffs = np.polyfit(x, y, 1)
    train_line = np.polyval(train_coeffs, x)
    p.line(x, train_line, legend_label='Train Data Regression Line', color='blue')

    # Regression line for selected ideal function column
    ideal_func_coeffs = np.polyfit(x, ideal_func_data, 1)
    ideal_func_line = np.polyval(ideal_func_coeffs, x)
    p.line(x, ideal_func_line, legend_label='Ideal Function Regression Line', color='red')

    # Add legend
    p.legend.location = 'top_left'

    # Add title
    p.add_layout(Label(text=title, text_font_size='16pt', text_font_style='bold', render_mode='css', y=550))

    # Show the plot
    show(p)
    
# Plot scatter plot with regression lines
plot_regression_scatter(train_data['x'], train_data['y1'], ideal_data['y50'], 'y1 Scatter Plot with y1 train and y50 ideal regression line', 'X', 'Y1')
plot_regression_scatter(train_data['x'], train_data['y2'], ideal_data['y8'], 'y2 Scatter Plot with y2 train and y8 ideal regression line', 'X', 'Y2')
plot_regression_scatter(train_data['x'], train_data['y3'], ideal_data['y34'], 'y3 Scatter Plot with y3 train and y34 ideal regression line', 'X', 'Y3')
plot_regression_scatter(train_data['x'], train_data['y4'], ideal_data['y9'], 'y4 Scatter Plot with y4 train and y9 ideal regression line', 'X', 'Y4')


# In[162]:


#Evaluation metric MSE method to confirm the selected 4 ideal function from least square method
#including user defined exception handling 

class EmptyMappingDictionaryError(Exception):
    """Exception raised when the mapping dictionary is empty."""

    def __init__(self, message="Empty mapping dictionary."):
        self.message = message
        super().__init__(self.message)

def mse(train_data, ideal_data):
    '''
    Calculate the Mean Squared Error (MSE) between the train data and ideal data.

    Args:
        train_data (ndarray): Array containing the predicted values (train data).
        ideal_data (ndarray): Array containing the true labels (ideal data).

    Returns:
        float: The calculated MSE value.
    '''
    return np.mean((train_data - ideal_data) ** 2)

def lowest_mse_label(train_feature, ideal_data):
    '''
    Calculate the lowest Mean Squared Error (MSE) and associated label for a given train feature.

    Args:
        train_feature (str): Name of the train feature.
        ideal_data (DataFrame): DataFrame containing the ideal data.

    Returns:
        tuple: A tuple containing the label with the lowest MSE and its corresponding MSE value.
    '''
    error_list = []
    for i in ideal_data.columns[1:]:
        prediction = ideal_data[i].values
        label = train_data[train_feature].values
        error = mse(label, prediction)
        label_error_tuple = i, error
        error_list.append(label_error_tuple)
    error_list_sorted = sorted(error_list, key=lambda x: x[1])
    return error_list_sorted[0]

def best_label_mapping(train_data, ideal_data):
    '''
    Find the best label mapping between train data and ideal data based on the lowest MSE values.

    Args:
        train_data (DataFrame): DataFrame containing the train data.
        ideal_data (DataFrame): DataFrame containing the ideal data.

    Returns:
        dict: A dictionary mapping each train feature to the label in ideal data with the lowest MSE value.
    '''
    best_label_list = []
    for train_col in train_data.columns[1:]:
        try:
            best_label = lowest_mse_label(train_col, ideal_data)
            best_label_list.append(best_label)
        except Exception as e:
            print(f"An error occurred while processing '{train_col}': {str(e)}")
    col_list = train_data.columns[1:]
    return dict(zip(col_list, best_label_list))

# Calculate the label mapping
try:
    mapping_dict = best_label_mapping(train_data, ideal_data)

    # Print the mapping dictionary
    print(mapping_dict)

    # Raise a user-defined exception if the mapping dictionary is empty
    if len(mapping_dict) == 0:
        raise EmptyMappingDictionaryError()

except EmptyMappingDictionaryError as EMDE:
    print(f"An error occurred: {str(EMDE)}")
except Exception as EX:
    print(f"An error occurred during label mapping: {str(EX)}")


# In[287]:


#Calculate the R-squared value of each selected four ideal functions with the test dataset based on matching x values

# Connect to the SQLite database
conn = engine.connect()

# Query the ideal data for columns y50, y8, y34, y9
ideal_query = "SELECT x, y50, y8, y34, y9 FROM ideal_data"
ideal_results = conn.execute(ideal_query).fetchall()

# Convert the ideal data into a pandas DataFrame
ideal_df = pd.DataFrame(ideal_results, columns=['x', 'y50', 'y8', 'y34', 'y9'])

# Query the test data for columns x, y
test_query = "SELECT x, y FROM test_data"
test_results = conn.execute(test_query).fetchall()

# Convert the test data into a pandas DataFrame
test_df = pd.DataFrame(test_results, columns=['x', 'y'])

# Initialize a dictionary to store the R-squared values
r_squared_values = {}

# Iterate over the columns of the ideal data
for col in ['y50', 'y8', 'y34', 'y9']:
    # Merge the test and ideal data based on matching x values
    merged_df = pd.merge(test_df, ideal_df[['x', col]], on='x', how='inner')
    
    # Calculate the R-squared value
    r_squared = r2_score(merged_df['y'], merged_df[col])
    
    # Store the R-squared value in the dictionary
    r_squared_values[col] = r_squared

# Print the R-squared values
for col, r_squared in r_squared_values.items():
    print(f"R-square value between {col} ideal function and Y test data points: {r_squared}")

# Close the database connection
conn.close()


# In[288]:


#Calculating absolute deviation between selected ideal data and train data to know individual largest deviation and plotting the results via bar graph
def max_abs_deviation(train_data, ideal_data):
    """
    Calculate the absolute maximum deviation between train_data and ideal_data.

    Parameters:
        train_data (numpy.ndarray): Array containing train data values.
        ideal_data (numpy.ndarray): Array containing ideal function values.

    Returns:
        float: The absolute maximum deviation between train_data and ideal_data.
    """
    max_abs_dev = np.max(np.abs(train_data - ideal_data))
    return max_abs_dev

def plot_bar_chart(ax, deviations, max_deviation_index, x_labels, title):
    """
    Plot a bar chart with deviation values.

    Parameters:
        ax (matplotlib.axes.Axes): Axes object to plot the chart on.
        deviations (numpy.ndarray): Array containing deviation values.
        max_deviation_index (int): Index of the maximum deviation value.
        x_labels (list): List of labels for the x-axis.
        title (str): Title of the plot.
    """
    ax.bar(range(max_deviation_index + 1), deviations[:max_deviation_index + 1], color='blue')
    ax.bar(max_deviation_index, deviations[max_deviation_index], color='red')
    for i, deviation in enumerate(deviations[:max_deviation_index + 1]):
        ax.text(i, deviation, f'{deviation:.4f}', ha='center', va='bottom')
    ax.set_xlabel('X-Data Points')
    ax.set_ylabel('Deviation')
    ax.set_title(title)
    ax.set_xticks(range(max_deviation_index + 1))
    ax.set_xticklabels(x_labels[:max_deviation_index + 1], rotation=90)
    ax.margins(x=0.01)  # Adjust the x-axis margins for better visibility
    
# Maximum deviation allowed for 'y1' variable
y1_train_data = np.array(list(train_data['y1'].values))
y1_ideal_data = np.array(list(ideal_data['y50'].values))
deviations_y1 = np.abs(y1_train_data - y1_ideal_data) * np.sqrt(2)
max_deviation_index_y1 = np.argmax(deviations_y1)
x_labels_y1 = list(train_data['x'].values)  # assuming 'x' is the corresponding variable for 'y1'
max_dev_1 = max_abs_deviation(y1_train_data, y1_ideal_data) * np.sqrt(2)
print('Maximum deviation allowed for y1 train variable and selected ideal function y50 is', max_dev_1)

# Set up the figure and axis for y1 train variable and ideal function y50
fig1, ax1 = plt.subplots(figsize=(18, 10))
plot_bar_chart(ax1, deviations_y1, max_deviation_index_y1, x_labels_y1, 'Deviation between y1 train variable and ideal function y50')
plt.show()

# Maximum deviation allowed for 'y2' variable
y2_train_data = np.array(list(train_data['y2'].values))
y2_ideal_data = np.array(list(ideal_data['y8'].values))
deviations_y2 = np.abs(y2_train_data - y2_ideal_data) * np.sqrt(2)
max_deviation_index_y2 = np.argmax(deviations_y2)
x_labels_y2 = list(train_data['x'].values)  # assuming 'x' is the corresponding variable for 'y2'
max_dev_2 = max_abs_deviation(y2_train_data, y2_ideal_data) * np.sqrt(2)
print('Maximum deviation allowed for y2 train variable and selected ideal function y8 is', max_dev_2)

# Set up the figure and axis for y2 train variable and ideal function y8
fig2, ax2 = plt.subplots(figsize=(18, 10))
plot_bar_chart(ax2, deviations_y2, max_deviation_index_y2, x_labels_y2, 'Deviation between y2 train variable and ideal function y8')
plt.show()

# Maximum deviation allowed for 'y3' variable
y3_train_data = np.array(list(train_data['y3'].values))
y3_ideal_data = np.array(list(ideal_data['y34'].values))
deviations_y3 = np.abs(y3_train_data - y3_ideal_data) * np.sqrt(2)
max_deviation_index_y3 = np.argmax(deviations_y3)
x_labels_y3 = list(train_data['x'].values)  # assuming 'x' is the corresponding variable for 'y3'
max_dev_3 = max_abs_deviation(y3_train_data, y3_ideal_data) * np.sqrt(2)
print('Maximum deviation allowed for y3 train variable and selected ideal function y34 is', max_dev_3)

# Set up the figure and axis for y3 train variable and ideal function y34
fig3, ax3 = plt.subplots(figsize=(25, 10))
plot_bar_chart(ax3, deviations_y3, max_deviation_index_y3, x_labels_y3, 'Deviation between y3 train variable and ideal function y34')
plt.show()

# Maximum deviation allowed for 'y4' variable
y4_train_data = np.array(list(train_data['y4'].values))
y4_ideal_data = np.array(list(ideal_data['y9'].values))
deviations_y4 = np.abs(y4_train_data - y4_ideal_data) * np.sqrt(2)
max_deviation_index_y4 = np.argmax(deviations_y4)
x_labels_y4 = list(train_data['x'].values)  # assuming 'x' is the corresponding variable for 'y4'
max_dev_4 = max_abs_deviation(y4_train_data, y4_ideal_data) * np.sqrt(2)
print('Maximum deviation allowed for y4 train variable and selected ideal function y9 is', max_dev_4)

# Set up the figure and axis for y4 train variable and ideal function y9
fig4, ax4 = plt.subplots(figsize=(18, 10))
plot_bar_chart(ax4, deviations_y4, max_deviation_index_y4, x_labels_y4, 'Deviation between y4 train variable and ideal function y9')
plt.show()


# In[289]:


#Calculate absolute deviations between predicted values of selected ideal function and test data 'y' values

def calculate_and_store_abs_deviations(database_path):
    """
    Calculate absolute deviations between predicted values of selected ideal function and test data 'y' values,
    and store the updated test data back to the database.

    Parameters:
        database_path (str): The absolute path to the SQLite database file.

    Returns:
        None
    """

    # Calculate regression values for each 'y' variable of selected ideal function and store them in a dictionary
    regression_results = {}
    y_variables = ['y50', 'y8', 'y34', 'y9']

    for y_var in y_variables:
        regression = linregress(ideal_data.index, ideal_data[y_var])
        regression_results[y_var] = regression

        print(f"\nRegression values for '{y_var}':")
        print("Slope:", regression.slope)
        print("Intercept:", regression.intercept)
        print("R-value:", regression.rvalue)
        print("P-value:", regression.pvalue)
        print("Standard Error:", regression.stderr)

    # Calculate predicted values for each 'y' variable of selected ideal function and store them in a dictionary
    predicted_values = {}

    for y_var in y_variables:
        slope = regression_results[y_var].slope
        intercept = regression_results[y_var].intercept
        predicted_values[y_var] = slope * test_data['x'] + intercept

    # Access predicted values for each 'y' variable of selected ideal function:
    y50_predicted = predicted_values['y50']
    y8_predicted = predicted_values['y8']
    y34_predicted = predicted_values['y34']
    y9_predicted = predicted_values['y9']

    # Map the predicted values to the test data
    test_data['y50_predicted'] = y50_predicted
    test_data['y8_predicted'] = y8_predicted
    test_data['y34_predicted'] = y34_predicted
    test_data['y9_predicted'] = y9_predicted

    # Calculate the absolute deviation between mapped predicted values and test data 'y' values
    test_data['y50_abs_deviation'] = np.abs(test_data['y'] - test_data['y50_predicted'])
    test_data['y8_abs_deviation'] = np.abs(test_data['y'] - test_data['y8_predicted'])
    test_data['y34_abs_deviation'] = np.abs(test_data['y'] - test_data['y34_predicted'])
    test_data['y9_abs_deviation'] = np.abs(test_data['y'] - test_data['y9_predicted'])

    # Print the test data with mapped predicted values and absolute deviations
    print(test_data)

    # Save the updated test data with absolute deviations to the database
    conn = sqlite3.connect(database_path)
    test_data.to_sql('test_data', con=conn, if_exists='replace', index=False)
    conn.close()

    # Confirm successful update
    print("Mapped predicted values and absolute deviations calculated and stored in the database!")

# Call the function to calculate and store absolute deviations
calculate_and_store_abs_deviations(database_path)


# In[290]:


#Filtering and Plotting Scatter plot of mapped values of x-y test data with corresponding selected ideal function and respective abso-lute deviation 

def filter_and_plot_data(test_data, max_dev_1, max_dev_2, max_dev_3, max_dev_4):
    """
    Filter the test data based on the condition, plot the mapped values, and display relevant information.

    Parameters:
        test_data (pd.DataFrame): DataFrame containing test data with 'x', 'y', 'y50_predicted', 'y50_abs_deviation',
                                  'y8_predicted', 'y8_abs_deviation', 'y34_predicted', 'y34_abs_deviation',
                                  'y9_predicted', and 'y9_abs_deviation' columns.
        max_dev_1 (float): Maximum deviation allowed for the 'y50_abs_deviation' variable.
        max_dev_2 (float): Maximum deviation allowed for the 'y8_abs_deviation' variable.
        max_dev_3 (float): Maximum deviation allowed for the 'y34_abs_deviation' variable.
        max_dev_4 (float): Maximum deviation allowed for the 'y9_abs_deviation' variable.

    Returns:
        None
    """
    # Filter the test data based on the conditions
    filtered_data_1 = test_data[test_data['y50_abs_deviation'] <= max_dev_1]
    filtered_data_2 = test_data[test_data['y8_abs_deviation'] <= max_dev_2]
    filtered_data_3 = test_data[test_data['y34_abs_deviation'] <= max_dev_3]
    filtered_data_4 = test_data[test_data['y9_abs_deviation'] <= max_dev_4]

    # Create Bokeh figures
    p1 = figure(title="Scatter Plot of Mapped Values of x-y test data and corresponding Y50 and y50_abs_deviation", x_axis_label='x', y_axis_label='y')
    p2 = figure(title="Scatter Plot of Mapped Values of x-y test data and corresponding Y8 and y8_abs_deviation", x_axis_label='x', y_axis_label='y')
    p3 = figure(title="Scatter Plot of Mapped Values of x-y test data and corresponding Y34 and y34_abs_deviation", x_axis_label='x', y_axis_label='y')
    p4 = figure(title="Scatter Plot of Mapped Values of x-y test data and corresponding Y9 and y9_abs_deviation", x_axis_label='x', y_axis_label='y')

    # Extract the mapped values for each condition
    mapped_x_points1 = filtered_data_1['x']
    y_values1 = filtered_data_1['y']
    y50_values = filtered_data_1['y50_predicted']
    y50_abs_deviation = filtered_data_1['y50_abs_deviation']

    mapped_x_points2 = filtered_data_2['x']
    y_values2 = filtered_data_2['y']
    y8_values = filtered_data_2['y8_predicted']
    y8_abs_deviation = filtered_data_2['y8_abs_deviation']

    mapped_x_points3 = filtered_data_3['x']
    y_values3 = filtered_data_3['y']
    y34_values = filtered_data_3['y34_predicted']
    y34_abs_deviation = filtered_data_3['y34_abs_deviation']

    mapped_x_points4 = filtered_data_4['x']
    y_values4 = filtered_data_4['y']
    y9_values = filtered_data_4['y9_predicted']
    y9_abs_deviation = filtered_data_4['y9_abs_deviation']

    # Create DataFrames for the mapped values
    mapped_values1 = pd.DataFrame({'x': mapped_x_points1, 'y': y_values1, 'y50': y50_values, 'y50_abs_deviation': y50_abs_deviation})
    mapped_values2 = pd.DataFrame({'x': mapped_x_points2, 'y': y_values2, 'y8': y8_values, 'y8_abs_deviation': y8_abs_deviation})
    mapped_values3 = pd.DataFrame({'x': mapped_x_points3, 'y': y_values3, 'y34': y34_values, 'y34_abs_deviation': y34_abs_deviation})
    mapped_values4 = pd.DataFrame({'x': mapped_x_points4, 'y': y_values4, 'y9': y9_values, 'y9_abs_deviation': y9_abs_deviation})

    # Print the mapped values for each condition
    print("Mapped Values of x-y test data and corresponding Y50 and y50_abs_deviation:")
    print(mapped_values1[['x', 'y', 'y50', 'y50_abs_deviation']])
    print(f"Total mapped test data points within {max_dev_1} Maximum deviation allowed for y1 train variable and selected ideal function y50: {len(mapped_x_points1)}")

    print("\nMapped Values of x-y test data and corresponding Y8 and y8_abs_deviation:")
    print(mapped_values2[['x', 'y', 'y8', 'y8_abs_deviation']])
    print(f"Total mapped test data points within {max_dev_2} Maximum deviation allowed for y2 train variable and selected ideal function y8: {len(mapped_x_points2)}")

    print("\nMapped Values of x-y test data and corresponding Y34 and y34_abs_deviation:")
    print(mapped_values3[['x', 'y', 'y34', 'y34_abs_deviation']])
    print(f"Total mapped test data points within {max_dev_3} Maximum deviation allowed for y3 train variable and selected ideal function y34: {len(mapped_x_points3)}")

    print("\nMapped Values of x-y test data and corresponding Y9 and y9_abs_deviation:")
    print(mapped_values4[['x', 'y', 'y9', 'y9_abs_deviation']])
    print(f"Total mapped test data points within {max_dev_4} Maximum deviation allowed for y4 train variable and selected ideal function y9: {len(mapped_x_points4)}")

    # Plot scatter glyphs with different shapes for each condition
    p1.circle(mapped_values1['x'], mapped_values1['y'], legend_label='y', color='blue', size=8, alpha=0.7)
    p1.square(mapped_values1['x'], mapped_values1['y50'], legend_label='y50', color='green', size=8, alpha=0.7)
    p1.triangle(mapped_values1['x'], mapped_values1['y50_abs_deviation'], legend_label='y50_abs_deviation', color='red', size=8, alpha=0.7)
    p1.legend.location = "top_left"

    p2.circle(mapped_values2['x'], mapped_values2['y'], legend_label='y', color='blue', size=8, alpha=0.7)
    p2.square(mapped_values2['x'], mapped_values2['y8'], legend_label='y8', color='green', size=8, alpha=0.7)
    p2.triangle(mapped_values2['x'], mapped_values2['y8_abs_deviation'], legend_label='y8_abs_deviation', color='red', size=8, alpha=0.7)
    p2.legend.location = "top_left"

    p3.circle(mapped_values3['x'], mapped_values3['y'], legend_label='y', color='blue', size=8, alpha=0.7)
    p3.square(mapped_values3['x'], mapped_values3['y34'], legend_label='y34', color='green', size=8, alpha=0.7)
    p3.triangle(mapped_values3['x'], mapped_values3['y34_abs_deviation'], legend_label='y34_abs_deviation', color='red', size=8, alpha=0.7)
    p3.legend.location = "top_left"

    p4.circle(mapped_values4['x'], mapped_values4['y'], legend_label='y', color='blue', size=8, alpha=0.7)
    p4.square(mapped_values4['x'], mapped_values4['y9'], legend_label='y9', color='green', size=8, alpha=0.7)
    p4.triangle(mapped_values4['x'], mapped_values4['y9_abs_deviation'], legend_label='y9_abs_deviation', color='red', size=8, alpha=0.7)
    p4.legend.location = "top_left"

    # Show the plots
    output_notebook()
    show(p1)
    show(p2)
    show(p3)
    show(p4)

# Call the function with test data and maximum deviations
filter_and_plot_data(test_data, max_dev_1, max_dev_2, max_dev_3, max_dev_4)


# In[291]:


#Visual representation proportion of the total test case uniquely mapped to all ideal function along with the un-mapped points
def analyze_mapped_data(test_data, max_dev_1, max_dev_2, max_dev_3, max_dev_4):
    """
    Analyzes the mapped test data based on given conditions for y50, y8, y34, and y9 absolute deviations.

    Parameters:
        test_data (pd.DataFrame): DataFrame containing the test data with columns 'x', 'y50_abs_deviation',
                                  'y8_abs_deviation', 'y34_abs_deviation', and 'y9_abs_deviation'.
        max_dev_1 (float): Maximum absolute deviation allowed for y50.
        max_dev_2 (float): Maximum absolute deviation allowed for y8.
        max_dev_3 (float): Maximum absolute deviation allowed for y34.
        max_dev_4 (float): Maximum absolute deviation allowed for y9.

    Returns:
        None: The function prints the analysis results and displays a pie chart.

    """
    # Filter the test data based on the condition for y50_abs_deviation
    filtered_data_50 = test_data[test_data['y50_abs_deviation'] <= max_dev_1]
    mapped_x_points_50 = filtered_data_50['x']

    # Filter the test data based on the condition for y8_abs_deviation
    filtered_data_8 = test_data[test_data['y8_abs_deviation'] <= max_dev_2]
    mapped_x_points_8 = filtered_data_8['x']

    # Filter the test data based on the condition for y34_abs_deviation
    filtered_data_34 = test_data[test_data['y34_abs_deviation'] <= max_dev_3]
    mapped_x_points_34 = filtered_data_34['x']

    # Filter the test data based on the condition for y9_abs_deviation
    filtered_data_9 = test_data[test_data['y9_abs_deviation'] <= max_dev_4]
    mapped_x_points_9 = filtered_data_9['x']

    # Create a set of all mapped x points
    mapped_x_points = set(mapped_x_points_50).union(set(mapped_x_points_8), set(mapped_x_points_34), set(mapped_x_points_9))

    # Count the number of x values not mapped for any y abs deviation
    unmapped_x_points = test_data[~test_data['x'].isin(mapped_x_points)]
    unmapped_count = len(unmapped_x_points)

    # Concatenate all the mapped x points
    mapped_x_points_all = pd.concat([mapped_x_points_50, mapped_x_points_8, mapped_x_points_34, mapped_x_points_9])

    # Count the number of x values mapped to multiple y abs deviations
    duplicated_count = mapped_x_points_all.duplicated().sum()

    # Count the number of x values uniquely mapped to y abs deviations
    unique_mapped_count = mapped_x_points_all.nunique()

    # Display the counts
    print(f"Total test data points not mapped for any ideal function (y50, y8, y34, y9): {unmapped_count}")
    print(f"Total test data points mapped to multiple selected ideal functions (y50, y8, y34, y9): {duplicated_count}")
    print(f"Total test data points uniquely mapped to all ideal functions (y50, y8, y34, y9): {unique_mapped_count}")

    # Create data for the pie chart
    labels = ['Unmapped', 'Mapped']
    sizes = [unmapped_count, unique_mapped_count]

    # Plot the pie chart
    plt.pie(sizes, labels=labels, autopct='%1.1f%%')
    plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
    plt.title('Proportion of Test Data Points mapped out of 90 data points (after removing duplicates)')
    plt.show()

# Call the function with test data
analyze_mapped_data(test_data, max_dev_1, max_dev_2, max_dev_3, max_dev_4)


# In[292]:


#Plot the predicted values, upper limits, lower limits, and mapped test data points for selcted best fit ideal function

def plot_predicted_values(ideal_data, max_dev, mapped_x_points, y_values, y_column):
    """
    Plot the predicted values, upper limits, lower limits, and mapped test data points for the given 'y_column'.

    Parameters:
        ideal_data (DataFrame): The DataFrame containing the ideal data with columns 'index', 'x', and the 'y_column'.
        max_dev (float): Maximum deviation for 'y_column' predictions.
        mapped_x_points (list): List of x-coordinates for the mapped test data points.
        y_values (list): List of corresponding y-coordinates for the mapped test data points.
        y_column (str): Name of the 'y' column to be used for plotting.

    Returns:
        None (displays the plot using Bokeh's show() function).
    """

    # Calculate predicted values for the given 'y_column'
    y_regression = linregress(ideal_data.index, ideal_data[y_column])
    slope_y = y_regression.slope
    intercept_y = y_regression.intercept
    y_predicted = slope_y * ideal_data.index + intercept_y

    # Add upper and lower limits for the predicted values of 'y_column'
    y_predicted_upper = y_predicted + max_dev
    y_predicted_lower = y_predicted - max_dev

    # Create a DataFrame to store the results
    results_df = pd.DataFrame({'Data Point': ideal_data.index,
                               'x': ideal_data['x'],
                               y_column: ideal_data[y_column],
                               y_column + ' Predicted': y_predicted,
                               'Upper Limit': y_predicted_upper,
                               'Lower Limit': y_predicted_lower})

    # Print the predicted values, upper limits, lower limits, and corresponding x and y values for 'y_column'
    print(results_df)

    # Convert the DataFrame to a ColumnDataSource
    source = ColumnDataSource(results_df)

    # Create the plot
    p = figure(title=f'Mapped test data points within the Upper and Lower Limits of Predicted Values of {y_column}',
               x_axis_label='x',
               y_axis_label=f'{y_column} Predicted',
               plot_width=800,
               plot_height=500)

    # Plot the predicted line
    p.line(x='x', y=y_column + ' Predicted', line_color='red', legend_label=f'{y_column} Predicted line', source=source)

    # Plot the upper and lower limits
    p.line(x='x', y='Upper Limit', line_color='green', line_dash='dashed', legend_label='Upper Limit', source=source)
    p.line(x='x', y='Lower Limit', line_color='green', line_dash='dashed', legend_label='Lower Limit', source=source)

    # Plot the mapped test data points
    p.scatter(mapped_x_points, y_values, color='blue', legend_label='Mapped Test Data Points')

    # Show the plot
    show(p)

plot_predicted_values(ideal_data, max_dev_1, mapped_x_points1, y_values1, 'y50')
plot_predicted_values(ideal_data, max_dev_2, mapped_x_points2, y_values2, 'y8')
plot_predicted_values(ideal_data, max_dev_3, mapped_x_points3, y_values3, 'y34')
plot_predicted_values(ideal_data, max_dev_4, mapped_x_points4, y_values4, 'y9')


# In[295]:


#Filter the test data based on conditions and save the results in the specified format as a table named 'test_data' in the database

def filter_and_save_test_data(engine, max_dev_1, max_dev_2, max_dev_3, max_dev_4):
    """
    This function filters the test data based on specified conditions and saves the results in the specified format as a table named 'test_data' in the database
.

    Parameters:
        engine (sqlalchemy.engine.Engine): The SQLAlchemy engine to connect to the database.
        max_dev_1 (float): The maximum deviation for 'y50_abs_deviation'.
        max_dev_2 (float): The maximum deviation for 'y8_abs_deviation'.
        max_dev_3 (float): The maximum deviation for 'y34_abs_deviation'.
        max_dev_4 (float): The maximum deviation for 'y9_abs_deviation'.

    Returns:
        None
    """
    # Query the test_data table and load the results into a DataFrame
    test_data_query = "SELECT * FROM test_data"
    test_data_df = pd.read_sql_query(test_data_query, engine)

    # Filter the test data based on the condition
    filtered_data1 = test_data_df[test_data_df['y50_abs_deviation'] <= max_dev_1]
    filtered_data2 = test_data_df[test_data_df['y8_abs_deviation'] <= max_dev_2]
    filtered_data3 = test_data_df[test_data_df['y34_abs_deviation'] <= max_dev_3]
    filtered_data4 = test_data_df[test_data_df['y9_abs_deviation'] <= max_dev_4]

    # Extract the mapped values of x points and corresponding y50_abs_deviation
    mapped_x_points1 = filtered_data1['x']
    y_values1 = filtered_data1['y']
    y50_values = filtered_data1['y50_predicted']
    y50_abs_deviation = filtered_data1['y50_abs_deviation']

    # Create a new DataFrame for the mapped values
    mapped_values1 = pd.DataFrame({'x': mapped_x_points1, 'y': y_values1, 'Delta Y': y50_abs_deviation, 'No. of ideal func': 'y50'})

    # Extract the mapped values of x points and corresponding y8_abs_deviation
    mapped_x_points2 = filtered_data2['x']
    y_values2 = filtered_data2['y']
    y8_values = filtered_data2['y8_predicted']
    y8_abs_deviation = filtered_data2['y8_abs_deviation']

    # Create a new DataFrame for the mapped values
    mapped_values2 = pd.DataFrame({'x': mapped_x_points2, 'y': y_values2, 'Delta Y': y8_abs_deviation, 'No. of ideal func': 'y8'})

    # Extract the mapped values of x points and corresponding y34_abs_deviation
    mapped_x_points3 = filtered_data3['x']
    y_values3 = filtered_data3['y']
    y34_values = filtered_data3['y34_predicted']
    y34_abs_deviation = filtered_data3['y34_abs_deviation']

    # Create a new DataFrame for the mapped values
    mapped_values3 = pd.DataFrame({'x': mapped_x_points3, 'y': y_values3, 'Delta Y': y34_abs_deviation, 'No. of ideal func': 'y34'})

    # Extract the mapped values of x points and corresponding y9_abs_deviation
    mapped_x_points4 = filtered_data4['x']
    y_values4 = filtered_data4['y']
    y9_values = filtered_data4['y9_predicted']
    y9_abs_deviation = filtered_data4['y9_abs_deviation']

    # Create a new DataFrame for the mapped values
    mapped_values4 = pd.DataFrame({'x': mapped_x_points4, 'y': y_values4, 'Delta Y': y9_abs_deviation, 'No. of ideal func': 'y9'})

    # Concatenate all the mapped values dataframes
    all_mapped_values = pd.concat([mapped_values1, mapped_values2, mapped_values3, mapped_values4])

    # Drop the existing test_data table from the database
    engine.execute("DROP TABLE IF EXISTS test_data")

    # Save the new DataFrame as the test_data table in the database
    all_mapped_values.to_sql('test_data', engine, index=False)

    # Display the new test_data table
    new_test_data_query = "SELECT * FROM test_data"
    new_test_data_df = pd.read_sql_query(new_test_data_query, engine)
    print("New test_data table:")
    print(new_test_data_df)

# Call the function with arguments for max_dev_1, max_dev_2, max_dev_3, and max_dev_4
filter_and_save_test_data(engine, max_dev_1, max_dev_2, max_dev_3, max_dev_4)


# In[301]:


#Query the database to display all the contents of ideal_data, train_data and test_data table

def query_table(table_name):
    """
    Query the database for a given table and load the results into a DataFrame.

    Parameters:
        table_name (str): The name of the table to query.

    Returns:
        pandas.DataFrame: DataFrame containing the results from the queried table.
    """
    query = f"SELECT * FROM {table_name}"
    return pd.read_sql_query(query, engine)

# Query and display the ideal_data DataFrame
ideal_data_df = query_table("ideal_data")
print("ideal_data DataFrame:")
print(ideal_data_df)

# Query and display the train_data DataFrame
train_data_df = query_table("train_data")
print("\ntrain_data DataFrame:")
print(train_data_df)

# Query and display the test_data DataFrame
test_data_df = query_table("test_data")
print("\ntest_data DataFrame:")
print(test_data_df)


# In[ ]:




