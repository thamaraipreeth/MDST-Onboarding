# %% [markdown]
# # Checkpoint 1 
# ## (Do not remove any comments that start with"# @@@".) 

# %% [markdown]
# Reminder: 
# 
# - You are being evaluated for completion and effort in this checkpoint. 
# - Avoid manual labor / hard coding as much as possible, everything we've taught you so far are meant to simplify and automate your process.
# - Please do not remove any comment that starts with: "# @@@". 

# %% [markdown]
# We will be working with the same `states_edu.csv` that you should already be familiar with from the tutorial.
# 
# We investigated Grade 8 reading score in the tutorial. For this checkpoint, you are asked to investigate another test. Here's an overview:
# 
# * Choose a specific response variable to focus on
# >Grade 4 Math, Grade 4 Reading, Grade 8 Math
# * Pick or create features to use
# >Will all the features be useful in predicting test score? Are some more important than others? Should you standardize, bin, or scale the data?
# * Explore the data as it relates to that test
# >Create at least 2 visualizations (graphs), each with a caption describing the graph and what it tells us about the data
# * Create training and testing data
# >Do you want to train on all the data? Only data from the last 10 years? Only Michigan data?
# * Train a ML model to predict outcome 
# >Define what you want to predict, and pick a model in sklearn to use (see sklearn <a href="https://scikit-learn.org/stable/modules/linear_model.html">regressors</a>).
# 
# 
# Include comments throughout your code! Every cleanup and preprocessing task should be documented.
# 

# %% [markdown]
# 

# %% [markdown]
# <h2> Data Cleanup </h2>
# 
# Import `numpy`, `pandas`, and `matplotlib`.
# 
# (Feel free to import other libraries!)

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# %% [markdown]
# Load in the "states_edu.csv" dataset and take a look at the head of the data

# %%
df = pd.read_csv("/Users/preethiparthasarathy/Downloads/MDST/MDST-Onboarding/data/states_edu.csv")
type(df)

# %% [markdown]
# You should always familiarize yourself with what each column in the dataframe represents. Read about the states_edu dataset here: https://www.kaggle.com/noriuk/us-education-datasets-unification-project

# %% [markdown]
# Use this space to rename columns, deal with missing data, etc. _(optional)_

# %%
df.shape
df.info()
df.isna().sum()

# %% [markdown]
# <h2>Exploratory Data Analysis (EDA) </h2>

# %% [markdown]
# Chosen one of Grade 4 Reading, Grade 4 Math, or Grade 8 Math to focus on: *Grade 8 Math*

# %% [markdown]
# How many years of data are logged in our dataset? 

# %%
# @@@ 1
df['YEAR'].nunique()

# %% [markdown]
# Let's compare Michigan to Ohio. Which state has the higher average across all years in the test you chose?

# %%
# @@@ 2
df[df['STATE'].isin(['MICHIGAN', 'OHIO'])].groupby('STATE')['AVG_MATH_4_SCORE'].mean()

# %% [markdown]
# Find the average for your chosen test across all states in 2019

# %%
# @@@ 3
df[df['YEAR'] == 2019]['AVG_MATH_8_SCORE'].mean()
print(df[df['YEAR'] == 2019]['AVG_MATH_8_SCORE'].mean())

# %% [markdown]
# For each state, find a maximum value for your chosen test score

# %%
# @@@ 4
max_score = df["AVG_MATH_8_SCORE"].max()
print(max_score)

# %% [markdown]
# *Refer to the `Grouping and Aggregating` section in Tutorial 0 if you are stuck.

# %% [markdown]
# <h2> Feature Engineering </h2>
# 
# After exploring the data, you can choose to modify features that you would use to predict the performance of the students on your chosen response variable. 
# 
# You can also create your own features. For example, perhaps you figured that maybe a state's expenditure per student may affect their overall academic performance so you create a expenditure_per_student feature.
# 
# Use this space to modify or create features.

# %%
# @@@ 5
# Create expenditure per student
df["EXPENDITURE_PER_STUDENT"] = df["TOTAL_EXPENDITURE"] / df["ENROLL"]

df[["TOTAL_EXPENDITURE", "ENROLL", "EXPENDITURE_PER_STUDENT"]].head()

# %% [markdown]
# Feature engineering justification: **I created an expenditure per student feature because total expenditure alone does not account for differences in state enrollment. Expenditure per student provides a more comparable measure of the financial resources available for each student and may be related to Grade 8 math performance.**

# %% [markdown]
# <h2>Visualization</h2>
# 
# Investigate the relationship between your chosen response variable and at least two predictors using visualizations. Write down your observations.
# 
# **Visualization 1**

# %%
# @@@ 6
import matplotlib.pyplot as plt

viz1 = df[["EXPENDITURE_PER_STUDENT", "AVG_MATH_8_SCORE"]].dropna()

plt.scatter(viz1["EXPENDITURE_PER_STUDENT"],
            viz1["AVG_MATH_8_SCORE"],
            alpha=0.6)

plt.xlabel("Expenditure per Student")
plt.ylabel("Average Grade 8 Math Score")
plt.title("Grade 8 Math Score vs. Expenditure per Student")

plt.show()

# %% [markdown]
# **There appears to be a weak positive relationship between expenditure per student and average Grade 8 math scores. States that spend more per student tend to have somewhat higher math scores, although there is considerable variation. This suggests that spending may be related to performance, but it is not the only factor affecting math scores.**

# %% [markdown]
# **Visualization 2**

# %%
# @@@ 7
viz2 = df[["YEAR", "AVG_MATH_8_SCORE"]].dropna()

plt.scatter(viz2["YEAR"],
            viz2["AVG_MATH_8_SCORE"],
            alpha=0.6)

plt.xlabel("Year")
plt.ylabel("Average Grade 8 Math Score")
plt.title("Grade 8 Math Score Over Time")

plt.show()

# %% [markdown]
# **Grade 8 math scores generally increase over time, although scores vary between states within each year. The positive relationship suggests that year may be useful as a predictor of Grade 8 math performance because average scores have changed over the time period represented in the dataset.**

# %% [markdown]
# <h2> Data Creation </h2>
# 
# _Use this space to create train/test data_

# %%
from sklearn.model_selection import train_test_split

# %%
# @@@ 8
# Keep only the variables we are using and remove missing values
model_data = df[
    ["EXPENDITURE_PER_STUDENT", "YEAR", "AVG_MATH_8_SCORE"]
].dropna()

# Predictor variables
X = model_data[["EXPENDITURE_PER_STUDENT", "YEAR"]]
y = model_data["AVG_MATH_8_SCORE"]

# %%
# @@@ 9 
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# %% [markdown]
# <h2> Prediction </h2>

# %% [markdown]
# ML Models [Resource](https://medium.com/@vijaya.beeravalli/comparison-of-machine-learning-classification-models-for-credit-card-default-data-c3cf805c9a5a)

# %%
# @@@ 10
# import your sklearn class here
from sklearn.linear_model import LinearRegression

# %%
# @@@ 11
# create your model here
model = LinearRegression()

# %%
model.fit(X_train, y_train)

# %%
y_pred = model.predict(X_test)

# %% [markdown]
# ## Evaluation

# %% [markdown]
# Choose some metrics to evaluate the performance of your model, some of them are mentioned in the tutorial.

# %%
# @@@ 12
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

print("MAE:", mean_absolute_error(y_test, y_pred))
print("MSE:", mean_squared_error(y_test, y_pred))
print("R2:", r2_score(y_test, y_pred))

# %% [markdown]
# We have copied over the graphs that visualize the model's performance on the training and testing set. 
# 
# Change `col_name` and modify the call to `plt.ylabel()` to isolate how a single predictor affects the model.

# %%
# @@@ 13

col_name = 'YEAR'

f = plt.figure(figsize=(12,6))
plt.scatter(X_train[col_name], y_train, color="red")
plt.scatter(X_train[col_name], model.predict(X_train), color="green")

plt.legend(["True Training", "Predicted Training"])
plt.xlabel(col_name)
plt.ylabel("Average Grade 8 Math Score")
plt.title("Model Behavior on Training Set")

# %%
# @@@ 14

col_name = "YEAR"

f = plt.figure(figsize=(12,6))
plt.scatter(X_test[col_name], y_test, color="blue")
plt.scatter(X_test[col_name], model.predict(X_test), color="black")

plt.legend(["True Testing", "Predicted Testing"])
plt.xlabel(col_name)
plt.ylabel("Average Grade 8 Math Score")
plt.title("Model Behavior on Testing Set")


