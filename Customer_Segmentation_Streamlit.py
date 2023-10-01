import streamlit as st
import pandas as pd
import pickle

# Load the K-means models
recency_model = pickle.load(open('CLTV_recency', 'rb'))
frequency_model = pickle.load(open('CLTV_frequecy', 'rb'))
revenue_model = pickle.load(open('CLTV_Revenue', 'rb'))

# Load the customer data DataFrame (CLTV_user)
CLTV_user= pd.read_csv(r"C:/Users/kashinath konade/Desktop/Machine learning (ML)/Coustomer_segmentation_Project/Customer_Segmentation_new.csv")

st.title('Customer Segmentation App')

# Sidebar with dropdown for selecting a Customer ID
selected_customer_id = st.sidebar.selectbox('Select Customer ID', CLTV_user['CustomerID'].unique())

# Display customer information if a Customer ID is selected
if selected_customer_id:
    customer_info = CLTV_user[CLTV_user['CustomerID'] == selected_customer_id]

    if not customer_info.empty:
        recency_cluster = recency_model.predict(customer_info[['Recency']])[0]
        frequency_cluster = frequency_model.predict(customer_info[['Frequency']])[0]
        revenue_cluster = revenue_model.predict(customer_info[['Revenue']])[0]

        st.header('Customer Information')
        st.write(f'Customer ID: {customer_info["CustomerID"].values[0]}')
        st.write(f'Recency Cluster: {recency_cluster}')
        st.write(f'Frequency Cluster: {frequency_cluster}')
        st.write(f'Revenue Cluster: {revenue_cluster}')
    else:
        st.error("Customer not found.")

# You can add more Streamlit elements and customization as needed
