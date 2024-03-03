import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

#---------part1---------------
# Load the dataset into a Pandas DataFrame
filePath = "C:/Users/LENOVO/Downloads/cars.csv"
df = pd.read_csv(filePath)

# Print the dataset
print("\nFull dataset:")
print(df)
print("______________________________________________________________")

#---------part2---------------
# Print the number of missing values in each feature
print("\nFeatures with missing values and the number of missing values:")
print(df.isnull().sum())
print("______________________________________________________________")

#---------part3---------------
# Check the data types of each feature
print("\nData types of each feature:")
print(df.dtypes)

df['horsepower'] = pd.to_numeric(df['horsepower'], errors='coerce')

# Identify rows with missing values before imputation
missing_before = df[df.isnull().any(axis=1)]

# Create a copy of the DataFrame before imputation
df_before_imputation = df.copy()

# Exclude the origin since it is non-numeric column to calculate the mean
numeric_columns = df.select_dtypes(include=['number']).columns

# Print mean values before imputation
print("\nMean values before imputation:")
print(df[numeric_columns].mean())

# Fill missing values with mean
df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].mean())

# Print mean values after imputation
print("\nMean values after imputation:")
print(df[numeric_columns].mean())

# Identify rows with missing values after imputation
missing_after = df[df.isnull().any(axis=1)]

# Print the rows where updates occurred
print("\nRows where updates occurred:")
print(df_before_imputation.loc[missing_before.index])
print("\nUpdated rows:")
print(df.loc[missing_after.index])
print("______________________________________________________________")

#-----part4--------
# Calculate quartiles (Q1, Q2, Q3) for each country
quartiles = df.groupby('origin')['mpg'].describe(percentiles=[.25, .5, .75])

# print the quartile info
print("\nQuartiles for each country:")
print(quartiles)

# Create a box plot
plt.figure(figsize=(10, 6))
sns.boxplot(x='origin', y='mpg', data=df, color='skyblue')

# Calculate and display the median for each
medians = df.groupby('origin')['mpg'].median()
for i, median in enumerate(medians):
    plt.text(i, median, f'{median:.2f}', horizontalalignment='center', verticalalignment='bottom', fontdict={'size': 12, 'weight': 'bold'}, color='black')

plt.title('Fuel Economy Comparison by Country')
plt.xlabel('Country of Origin')
plt.ylabel('Miles Per Gallon (mpg)')
plt.show()
print("______________________________________________________________")

#------part5&6----------
# Create subplots for each feature
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Plot histograms for acceleration, horsepower, and mpg
sns.histplot(df['acceleration'], kde=True, color='blue', ax=axes[0])
axes[0].set_title('Acceleration Distribution')

sns.histplot(df['horsepower'], kde=True, color='royalblue', ax=axes[1])
axes[1].set_title('Horsepower Distribution')

sns.histplot(df['mpg'], kde=True, color='navy', ax=axes[2])
axes[2].set_title('MPG Distribution')

plt.tight_layout()

plt.show()

# Calculate mean, and median to support the answer in Q5
mean_value_mpg = df['mpg'].mean()
median_value_mpg = df['mpg'].median()
# Display the results
print(f"Mean for {'mpg'}: {mean_value_mpg}")
print(f"Median for {'mpg'}: {median_value_mpg}")

mean_value_acceleration= df['acceleration'].mean()
median_value_acceleration = df['acceleration'].median()
# Display the results
print(f"Mean for {'acceleration'}: {mean_value_acceleration}")
print(f"Median for {'acceleration'}: {median_value_acceleration}")

mean_value_horsepower= df['horsepower'].mean()
median_value_horsepower = df['horsepower'].median()
# Display the results
print(f"Mean for {'horsepower'}: {mean_value_horsepower}")
print(f"Median for {'horsepower'}: {median_value_horsepower}")
print("______________________________________________________________")

#-------part7---------
correlation_coefficient = df['horsepower'].corr(df['mpg'])
print("Correlation Coefficient:", correlation_coefficient)

# Scatter plot
plt.scatter(df['horsepower'], df['mpg'], color='navy', alpha=0.6)
plt.title('Scatter Plot of Horsepower vs. MPG')
plt.xlabel('Horsepower')
plt.ylabel('MPG')
plt.grid(True)
plt.show()
print("______________________________________________________________")

#-------part8--------
# Add a column of 1's for the intercept term
X = np.c_[np.ones(df.shape[0]), df['horsepower'].values]

# Target
y = df['mpg'].values

# Calculate the closed-form solution
theta = np.linalg.inv(X.T @ X) @ X.T @ y

plt.scatter(df['horsepower'], df['mpg'], color='navy', alpha=0.6, label='Data Points')

# Plot the learned line
plt.plot(df['horsepower'], X @ theta, color='red', label='Linear Regression')

plt.title('Linear Regression: Horsepower vs. MPG')
plt.xlabel('Horsepower')
plt.ylabel('MPG')
plt.legend()
plt.grid(True)
plt.show()

#-----------part9-------------

# Add columns for x and x^2 to the matrix
X_quad = np.c_[np.ones(df.shape[0]), df['horsepower'].values, df['horsepower'].values**2]

# Target
y_quad = df['mpg'].values

# Calculate the closed-form solution for the quadratic function
theta_quad = np.linalg.inv(X_quad.T @ X_quad) @ X_quad.T @ y_quad

plt.scatter(df['horsepower'], df['mpg'], color='navy', alpha=0.6, label='Data Points')

# Generating points for the quadratic equation
x_values = np.linspace(df['horsepower'].min(), df['horsepower'].max(), 100)
y_values_quad = theta_quad[0] + theta_quad[1] * x_values + theta_quad[2] * x_values**2

plt.plot(x_values, y_values_quad, color='red', label='Quadratic Regression')

plt.title('Quadratic Regression: Horsepower vs. MPG')
plt.xlabel('Horsepower')
plt.ylabel('MPG')
plt.legend()
plt.grid(True)
plt.show()

#-----------part10-------------

# Feature matrix
X = np.c_[np.ones(df.shape[0]), df['horsepower'].values]

# Target
y = df['mpg'].values

# Initialize parameters
theta = np.zeros(X.shape[1])

# The parameters
alpha = 0.000025
num_iterations = 1000000

# Cost function
def compute_cost(X, y, theta):
    m = len(y)
    predictions = X.dot(theta)
    cost = (1/(2*m)) * np.sum(np.square(predictions - y))
    return cost

# Gradient descent
def gradient_descent(X, y, theta, alpha, num_iterations):
    m = len(y)
    cost_history = []

    for _ in range(num_iterations):
        predictions = X.dot(theta)
        errors = predictions - y
        gradient = (1/m) * X.T.dot(errors)
        theta = theta - alpha * gradient
        cost_history.append(compute_cost(X, y, theta))

    return theta, cost_history

# Run the gradient descent algo
theta_grad_desc, cost_history = gradient_descent(X, y, theta, alpha, num_iterations)

plt.scatter(df['horsepower'], df['mpg'], color='navy', alpha=0.6, label='Data Points')

# Plot the linear regression line
x_values = np.linspace(df['horsepower'].min(), df['horsepower'].max(), 100)
y_values_grad_desc = theta_grad_desc[0] + theta_grad_desc[1] * x_values
plt.plot(x_values, y_values_grad_desc, color='red', label='Linear Regression (Gradient Descent)')

plt.title('Linear Regression: Horsepower vs. MPG (By Gradient Descent)')
plt.xlabel('Horsepower')
plt.ylabel('MPG')
plt.legend()
plt.grid(True)
plt.show()