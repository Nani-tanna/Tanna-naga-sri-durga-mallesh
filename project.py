#importing required modules
import numpy as np
import pandas as pd
import matplotlib as mt
import seaborn as sns

#loading the dataset
df1 = pd.read_csv("Bangalore_Restaurants.csv")

#exploring the dataset
df1.head()

#summary of the DataFrame
print(df1.info())

#printing the dataframe
print(df1)

#Correlation between two variables
df1.corr

# Display the first few rows of the DataFrame
print(df1.head())

#Visualization of data
k=sns.pairplot(df1)
k

#importing modules for data visulization
import matplotlib.pyplot as plt

# Example: Distribution of ratings
sns.histplot(df1['Delivery_Rating'])
plt.show()

# Example: Scatter plot of price vs. rating
sns.scatterplot(x='Pricing_for_2', y='Delivery_Rating', data=df1)
plt.show()

# Calculate average rating by dining type (Aggregation)
avg_rating_by_cuisine = df1.groupby('Category')['Dining_Rating'].mean()
print(avg_rating_by_cuisine)

# Check for missing values
print(df1.isnull().sum())

#knowing the type of data
df1.dtypes

#Preprocessing Data for AI(Artificial Intelligence)
#importing Scikit-learn module
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Encoding categorical variables
le = LabelEncoder()
df1['Dining_Rating'] = le.fit_transform(df1['Dining_Rating'])

# Scaling numerical features
scaler = StandardScaler()
df1[['Pricing_for_2','Delivery_Rating']] = scaler.fit_transform(df1[['Pricing_for_2', 'Delivery_Rating']])

# Print the preprocessed dataframe
print(df1)

# Predicting Ratings(making Predictions)
#importing scikit-learn module
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Define features and target variable
X = df1[['Pricing_for_2', 'Delivery_Rating']]
y = df1['Dining_Rating']

# Encoding categorical variables
le = LabelEncoder()
y = le.fit_transform(y)

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale numerical features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train a linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse:.2f}")

# Clustering Restaurants(Clustering data)
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Select features for clustering
X_clustering = df1[['Pricing_for_2', 'Dining_Rating']]

# Define and fit the model
kmeans = KMeans(n_clusters=3, random_state=42)
df1['cluster'] = kmeans.fit_predict(X_clustering)

# Plot clusters
plt.scatter(df1['Pricing_for_2'], df1['Dining_Rating'], c=df1['cluster'], cmap='viridis')
plt.xlabel('Pricing_for_2')
plt.ylabel('Dining_Rating')
plt.title('Restaurant Clusters')
plt.show()

# Content-Based Filtering(Recommendation Systems)
from sklearn.neighbors import NearestNeighbors

# Create a feature matrix
features = df1[['Pricing_for_2', 'Delivery_Rating']]

# Scale numerical features
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# Fit Nearest Neighbors model
model = NearestNeighbors(n_neighbors=5, algorithm='ball_tree')
model.fit(features_scaled)

# Find similar restaurants
distances, indices = model.kneighbors([features_scaled[0]])
print("Similar restaurants:")
for i in indices[0]:
    print(df1.iloc[i])

#Sentiment Analysis
#importing textblob
from textblob import TextBlob

# Example of adding sentiment score
def get_sentiment(text):
    analysis = TextBlob(str(text))
    return analysis.sentiment.polarity

df1['sentiment'] = df1['Dining_Rating'].apply(get_sentiment)
print(df1.head())