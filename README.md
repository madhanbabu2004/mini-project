# CRICKET STATS ANALYSIS AND PREDICTION
## Introduction
Cricket is a popular sport in many nations throughout the world, including India. Cricket predictions have always caught the interest of fans. Cricket analytic uses data to help players and coaches make better decisions by providing insights into various aspects of the game. 
Data can include player performance. Batsman records, Strick rate, and Average. By identifying patterns and trends in the data, the models. 
Predict players performance based on: Players ranking: The probability of selecting for a team. Team strategies: Saving a particular players for special occasions likely to Hit runs in a special scenario, or the potential run-scoring rate of a batsman.
Predictive analytics uses statistical models and machine learning algorithms to analyze historical.
## SCOPE OF PROJECT:
### Data Collection and Preprocessing

This module is fundamental, as it gathers and prepares the raw data needed for analysis.
Sources of Data: Utilizing publicly available match dataset which includes player statistics, individual stats.
Data Cleaning: Removing duplicates, handling missing values, and correcting inconsistencies using proposed approach.
Data Transformation: Converting raw data into formats suitable for analysis, such as aggregating match statistics or normalizing player performance metrics.

### Data Exploration Analysis

This module focuses on summarizing and visualizing historical data to understand past performances and trends.
Statistical Summaries: Calculating averages, medians, standard deviations, and other basic statistics.
Visualizations: Creating charts and graphs to visualize trends (e.g., player form over time, team performance in different conditions).
Performance Metrics: Developing advanced metrics (e.g., strike rates under pressure, bowler economy rates in different innings).

### Feature Extraction: 

Identifying and creating relevant features from raw data that can improve model accuracy (e.g., recent form, head-to-head stats, weather conditions).
Features;

RUNS	MATCHES	SIXES	FOURS	STRICKRATE

AVERAGE	CENTURIES	INNINGS	NOTOUTS	HALF CENTUARIES
## Programs
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
data = pd.read_excel('cricket.xlsx')

data['RUNS'] = data['RUNS'].apply(lambda x: round(x / 25))
data['AVG'] = data['AVG'].apply(lambda x: round(x / 10))
data['SR'] = data['SR'].apply(lambda x: round(x / 20))
print(data.head())
data['sum_col'] = data['RUNS'] + data['AVG'] + data['SR']
print(data.head())

df_sorted = data.sort_values(by='sum_col', ascending=False)
print(df_sorted.head())

df_sorted.rename(columns={'sum_col': 'RATING'}, inplace=True)
df_sorted.head(20)

df_sorted['Rank'] = df_sorted['RATING'].rank(ascending=False, method='min')
df_rank = df_sorted.sort_values('Rank')
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

SEED = 23

X = df_rank[['MAT','RUNS', 'AVG', 'SR','Rank','Hundred']]
y = df_rank['RATING']
train_X, test_X, train_y, test_y = train_test_split(X, y,
													test_size = 0.25,
													random_state = SEED)

gbc = GradientBoostingClassifier(n_estimators=300,
								learning_rate=0.05,
								random_state=100,
								max_features=5 )
gbc.fit(train_X, train_y)

pred_y = gbc.predict(test_X)

acc = accuracy_score(test_y, pred_y)
print("Gradient Boosting Classifier accuracy is : {:.2f}".format(acc))

from numpy import mean
from numpy import std
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.feature_selection import RFE
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
X=df_rank[['MAT','RUNS', 'AVG', 'SR','Rank','Hundred']]
y = df_rank['RATING']
rfe = RFE(estimator=DecisionTreeClassifier(), n_features_to_select=5)
model = DecisionTreeClassifier()
pipeline = Pipeline(steps=[('s',rfe),('m',model)])
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
n_scores = cross_val_score(pipeline, X, y, scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import RFECV


X = df_rank[['MAT','RUNS', 'AVG', 'SR','Rank']]
y = df_rank['RATING']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf = RandomForestClassifier(random_state=42)


selector = RFECV(estimator=rf, step=1, cv=5, scoring='accuracy')
selector = selector.fit(X_train, y_train)

X_train_selected = selector.transform(X_train)
X_test_selected = selector.transform(X_test)

rf.fit(X_train_selected, y_train)

y_pred = rf.predict(X_test_selected)
accuracy = accuracy_score(y_test, y_pred)


precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')
print("Selected features:", X.columns[selector.support_])
print("Model accuracy after feature selection:", accuracy)
print("Model precision after feature selection:", precision)
print("Model recall after feature selection:", recall)
print("Model F1-score after feature selection:", f1)
```
## Output

![Screenshot 2024-11-12 162912](https://github.com/user-attachments/assets/7e5ae128-2375-464d-a903-8ec4de138e0b)
![Screenshot 2024-11-13 100721](https://github.com/user-attachments/assets/b2b450fa-ed3b-4634-865a-bd8740b50c41)
![Screenshot 2024-10-23 103047](https://github.com/user-attachments/assets/29b19f1e-4711-41b2-8d06-8030d861c554)
![Screenshot 2024-10-23 103101](https://github.com/user-attachments/assets/2da6b334-d894-4759-a506-8a0693f2f1b3)
![Screenshot 2024-10-23 103143](https://github.com/user-attachments/assets/98a32001-991a-4ca0-976d-2fb86ca0bb2b)
![Screenshot 2024-10-23 103425](https://github.com/user-attachments/assets/20f94160-7243-4909-89bf-1ce855b7c782)
![Screenshot 2024-10-23 104601](https://github.com/user-attachments/assets/6ee01544-9599-4c8b-92c9-4a63eedb3cd0)

                         
