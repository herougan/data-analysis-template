from lib2to3.pgen2.pgen import DFAState
import pandas as pd
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
import matplotlib.pyplot as plt
import numpy as np
from statsmodels.formula.api import ols
import seaborn as sns
from statsmodels.stats.outliers_influence import variance_inflation_factor
# from sklearn.linear_model import LinearRegression
# import tensorflow as tf # only 3.11 accepted

'''
    Loads data from stats and stakes file and performs multilinear regression to find
    patterns in the data.
'''

# === Load data ===
stats_file = 'StakeData.csv'
stakes_file = 'VariableData.csv'
stats = pd.read_csv(stats_file)
stakes = pd.read_csv(stakes_file)

df = stats.merge(stakes, on='Name', how='outer') # Case sensitive
df = df.set_index('Name')

# === Data cleaning ===
# Replace '-' with NaN
df = df.replace("-", np.nan)
print("Data Frame, unmodified")
# Forward fill NaN values
df = df.ffill()
print("Forward filling missing values")
print(df)
# Change string to int (I don't why this happens)
df["MG_RTP"] = df["MG_RTP"].astype(float)
df["MG_stdev"] = df["MG_stdev"].astype(float)
df["FS_entry"] = df["FS_entry"].astype(float)

# X = df.loc[:, df.columns != 'Stake'] # For all columns
X = df[["MG_RTP", "MG_stdev", "FS_entry"]]
Y = df[["Stake"]]

# # === Plot relationship === (Uncomment to print images)
sns.pairplot(df,
            corner=True,)
plt.savefig('img/pairplot.png')
sns.pairplot(df,
            kind="kde",
            corner=True,)
plt.savefig('img/pairplotcontour.png')
# Colouring by Categorical Stake
df_stake = df.copy()
df_stake['Stake_Cat'] = pd.cut(df['Stake'], bins=[0,5000,10000,float('Inf')], labels=['Low', 'Med', 'High'])
sns.pairplot(df_stake,
            hue="Stake_Cat",)
plt.savefig('img/pairplotbystake.png')

# Note #
# Challenges faced: I can't select the columns I want for pairplotting. If I do (with {x, y}_var, the plot plots by index instead of by value))

# === Training Test Split ===
# X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3) # Normally, you fit to the training data, then test the results on the test data

# === Fit ===
ols_f = "Stake ~ MG_RTP + MG_stdev + FS_entry" # Use C(Stake_Cat) for example, for categorical variables
ols_data = pd.concat([X, Y], axis = 1)
print(ols_data)
print(ols_data.describe())
print(ols_data.dtypes)
OLS = ols(formula = ols_f, data = ols_data) 


# # === Fit ===
model = OLS.fit()
results = model.summary()
print(results)
residuals = model.resid
coeff = model.params
tvalues = model.tvalues
print(tvalues)

# # === These are the 4 conditions for multi-f(x) regression ===
# # Check linearity
# sns.scatterplot(x = data['Social_Media'], y = data['Sales'],ax=axes[1])
# # Check Normality
# residuals = model.resid
# sns.histplot(residuals, ax=axes[0])
# sm.qqplot(residuals, line='s',ax = axes[1])
# # Check Constant Variance
# fig = sns.scatterplot(x = model.fittedvalues, y = model.resid)
# # Check multicollienarity
# X = data[['Radio','Social_Media']]
# vif = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
# df_vif = pd.DataFrame(vif, index=X.columns, columns = ['VIF'])

# # === Plotting ===
# # Clean, join
# sns.histplot(residuals, ax=axes[0]) # histogram of residuals
# sm.qqplot(residuals, line='s', ax=axes[1]) # qq residue plot
# fig = sns.scatterplot(x = model.fittedvalues, y=model.resid) # fitted curve

# # Pair-plot



# # correlation contribution



# # end

# # Type "exit()"


# #region Future

# # === Variable Selection ===
# # Backward elimination, forward selection, extra-sum-of-squares F-test, regularisation

# # === Other models ===
# # Principal Component Regression
# # Non-linear regression

# # === Unsupervised Learning ===
# # Boosted Trees (e.g. XGBoost)
# # Required: Categorise Stake into "High", "Medium", and "Low"
# # Tree predicts the category of a game given its' attributes
# # also outputs each attributes importance

# #endregion Future