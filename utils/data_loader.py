import pandas as pd

def load_data():
    """Loads product and transaction data."""
    products = pd.read_csv('data/all-products.csv')
    transactions = pd.read_csv('data/orders.csv')
    return products, transactions