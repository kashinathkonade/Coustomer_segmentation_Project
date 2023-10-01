from flask import Flask, render_template, request
import pandas as pd
import pickle

app = Flask(__name__)

# Load the K-means models
recency_model = pickle.load(open('CLTV_recency', 'rb'))
frequency_model = pickle.load(open('CLTV_frequecy', 'rb'))
revenue_model = pickle.load(open('CLTV_Revenue', 'rb'))

# Load the customer data DataFrame 
CLTV_user= pd.read_csv(r"C:/Users/kashinath konade/Desktop/Machine learning (ML)/Coustomer_segmentation_Project/Customer_Segmentation_new.csv")

@app.route('/')
def index():
    
    # Get unique CustomerIDs for the dropdown
    customer_ids = CLTV_user['CustomerID'].unique()
    return render_template('customer_search.html', customer_ids=customer_ids)
    
@app.route('/customer_info', methods=['GET', 'POST'])
def customer_info():
    if request.method == 'POST':
        customer_id = request.form.get('customer_id')
        customer_info = CLTV_user[CLTV_user['CustomerID'] == int(customer_id)]

        if not customer_info.empty:
            recency_cluster = recency_model.predict(customer_info[['Recency']])[0]
            frequency_cluster = frequency_model.predict(customer_info[['Frequency']])[0]
            revenue_cluster = revenue_model.predict(customer_info[['Revenue']])[0]

            return render_template('customer_info.html', customer_info=customer_info, recency_cluster=recency_cluster,
                                   frequency_cluster=frequency_cluster, revenue_cluster=revenue_cluster)
        else:
            return "Customer not found."

    return render_template('customer_search.html')

if __name__ == '__main__':
    app.run(debug=True)
