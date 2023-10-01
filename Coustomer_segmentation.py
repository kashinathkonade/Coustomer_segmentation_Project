import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from urllib.parse import quote
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import pickle
from sqlalchemy import create_engine

# Load the data
CLTV_data = pd.read_csv(r"C:/Users/kashinath konade/Downloads/customer_segmentation.csv", encoding='unicode_escape')

# Database connection details
user = 'root'
pw = 'kashinath@123'
db = 'ml'
engine = create_engine(f'mysql+pymysql://{user}:%s@localhost:3306/{db}' % quote(f'{pw}'))

# Save the data to the database
CLTV_data.to_sql('CLTV', con=engine, if_exists='replace', chunksize=1000, index=False)

# Data Preprocessing
CLTV_data.drop(["InvoiceNo", "StockCode", "Description", "Country"], axis=1, inplace=True)
CLTV_data['InvoiceDate'] = pd.to_datetime(CLTV_data['InvoiceDate'])
CLTV_data.dropna(subset=["CustomerID"], inplace=True)

# Check for outliers
CLTV_data.plot(kind='box', subplots=True, sharey=False, figsize=(15, 8))
plt.show()

# Create a new dataframe with unique CustomerID
CLTV_user = pd.DataFrame(CLTV_data['CustomerID'].unique(), columns=['CustomerID'])

# Calculate Recency
CLTV_max_buy = CLTV_data.groupby("CustomerID").InvoiceDate.max().reset_index()
CLTV_max_buy['Recency'] = (CLTV_max_buy['InvoiceDate'].max() - CLTV_max_buy['InvoiceDate']).dt.days
CLTV_user = pd.merge(CLTV_user, CLTV_max_buy[['CustomerID', 'Recency']], on='CustomerID')

# K-Means clustering for Recency
silhouette_coeff = []
TWSS = []
for k in range(2, 10):
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(CLTV_user[['Recency']])
    score = silhouette_score(CLTV_user[['Recency']], kmeans.labels_)
    TWSS.append(kmeans.inertia_)
    silhouette_coeff.append([k, score])

best_model = KMeans(n_clusters=3)  # Choose the number of clusters based on your analysis
result_c = best_model.fit(CLTV_user[['Recency']])
pickle.dump(result_c, open('CLTV_recency.pkl', 'wb'))
CLTV_user["RecencyCluster"] = best_model.fit_predict(CLTV_user[['Recency']])

# Calculate Frequency
CLTV_frequency = CLTV_data.groupby("CustomerID").InvoiceDate.count().reset_index()
CLTV_frequency.columns = ['CustomerID', 'Frequency']
CLTV_user = pd.merge(CLTV_user, CLTV_frequency, on='CustomerID')

# K-Means clustering for Frequency
silhouette_coeff_f = []
TWSS_f = []
for k in range(2, 10):
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(CLTV_user[['Frequency']])
    score = silhouette_score(CLTV_user[['Frequency']], kmeans.labels_)
    TWSS_f.append(kmeans.inertia_)
    silhouette_coeff_f.append([k, score])

best_model_f = KMeans(n_clusters=3)  # Choose the number of clusters based on your analysis
result_f = best_model_f.fit(CLTV_user[['Frequency']])
pickle.dump(result_f, open('CLTV_frequency.pkl', 'wb'))
CLTV_user["FrequencyCluster"] = best_model_f.fit_predict(CLTV_user[['Frequency']])

# Calculate Monetary value
CLTV_data["Revenue"] = CLTV_data["UnitPrice"] * CLTV_data["Quantity"]
CLTV_monetary = CLTV_data.groupby("CustomerID").Revenue.sum().reset_index()
CLTV_user = pd.merge(CLTV_user, CLTV_monetary, on='CustomerID')

# K-Means clustering for Monetary
silhouette_coeff_m = []
TWSS_m = []
for k in range(2, 10):
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(CLTV_user[['Revenue']])
    score = silhouette_score(CLTV_user[['Revenue']], kmeans.labels_)
    TWSS_m.append(kmeans.inertia_)
    silhouette_coeff_m.append([k, score])

best_model_m = KMeans(n_clusters=3)  # Choose the number of clusters based on your analysis
result_m = best_model_m.fit(CLTV_user[['Revenue']])
pickle.dump(result_m, open('CLTV_monetary.pkl', 'wb'))
CLTV_user['MonetaryCluster'] = best_model_m.fit_predict(CLTV_user[['Revenue']])

# Define a function to order clusters
def order(cluster_field, target, df, ascending):
    df_new = df.groupby(cluster_field)[target].mean().reset_index()
    df_new = df_new.sort_values(by=target, ascending=ascending).reset_index()
    df_new['index'] = df_new.index
    df_final = pd.merge(df, df_new[[cluster_field, 'index']], on=cluster_field)
    df_final = df_final.drop([cluster_field], axis=1)
    df_final = df_final.rename(columns={"index": cluster_field})
    return df_final

# Order clusters for Recency, Frequency, and Monetary
CLTV_user = order('RecencyCluster', 'Recency', CLTV_user, False)
CLTV_user = order('FrequencyCluster', 'Frequency', CLTV_user, True)
CLTV_user = order('MonetaryCluster', 'Revenue', CLTV_user, True)

# Calculate OverallScore
CLTV_user['OverallScore'] = CLTV_user['RecencyCluster'] + CLTV_user['FrequencyCluster'] + CLTV_user['MonetaryCluster']

# Assign segments
CLTV_user["segment"] = "Low-Value"
CLTV_user["segment"] = np.where(CLTV_user["OverallScore"] > 1, "Mid-Value",
                                np.where(CLTV_user["OverallScore"] > 3, "High-Value", CLTV_user["segment"]))

# Save the results to a CSV file
CLTV_user.to_csv('Customer_Segmentation_new.csv', encoding='utf-8', index=False)

import os
print(os.getcwd())  # Print the current working directory
