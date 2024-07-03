from flask import Flask, request, jsonify
import pandas as pd
from models.collaborative_filtering_model import CollaborativeFilteringModel
from models.content_based_filtering import get_content_based_recommendations
from models.collaborative_filtering import get_combined_recommendations
# from data.load_data import load_products, load_transactions
from models.content_based_model import ContentBasedModel
from utils.data_loader import load_data
from utils.payment import create_payment_request

app = Flask(__name__)

# Load data
products, transactions = load_data()


# Initialize models
content_based_model = ContentBasedModel(products)
collaborative_filtering_model = CollaborativeFilteringModel(transactions)

@app.route('/recommend/<string:user_id>', methods=['GET'])
def recommend_by_user(user_id):
    """Recommends products for a given user."""
    try:
        # user_id = int(user_id)  # Chuyển đổi user_id thành số nguyên
        recommendations = collaborative_filtering_model.get_recommendations(user_id)
        return jsonify(recommendations)
    except ValueError:
        return jsonify({"error": "Invalid user ID. Please provide an integer."}), 400

@app.route('/recommend/product/<int:product_id>', methods=['GET'])
def recommend_by_product(product_id):
    """Recommends products similar to a given product."""
    recommendations = content_based_model.get_recommendations(product_id)
    return jsonify(recommendations)

@app.route('/payment', methods=['POST'])
def pay():
    data = request.json
    order_id = data.get('orderId')
    amount = data.get('amount')

    if not order_id or not amount:
        return jsonify({'error': 'Missing orderId or amount'}), 400

    try:
        payment_response = create_payment_request(order_id, amount)
        return jsonify(payment_response)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
