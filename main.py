from flask import Flask, request, jsonify
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from surprise import Dataset, Reader, KNNBasic
from flask import Flask, jsonify

app =  Flask(__name__)

# Đọc dữ liệu từ các tập tin Excel
products = pd.read_csv('all-products.csv')
transactions = pd.read_csv('orders.csv')

# Chuẩn bị dữ liệu cho Content-Based Filtering
tfidf = TfidfVectorizer(stop_words='english')
products['description'] = products['Title'] + ' ' + products['CategoryId'].astype(str)  # Thay đổi thuộc tính để sử dụng
tfidf_matrix = tfidf.fit_transform(products['description'])
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

# Hàm để gợi ý sản phẩm dựa trên nội dung
def get_content_based_recommendations(product_id, cosine_sim=cosine_sim):
    idx = products[products['Id'] == product_id].index[0]  # Sử dụng 'ID' thay vì 'productId'
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:6]
    product_indices = [i[0] for i in sim_scores]
    recommendations = products['Id'].iloc[product_indices].tolist()
    # print(f"Content-based recommendations for product {product_id}: {recommendations}")
    return recommendations

# Chuẩn bị dữ liệu cho Collaborative Filtering
reader = Reader(rating_scale=(1, 1))
dataset = Dataset.load_from_df(transactions[['UserId', 'ProductId', 'Quantity']], reader)  # Thay 'rating' bằng 'Quantity'
trainset = dataset.build_full_trainset()
algo = KNNBasic(sim_options={'user_based': False, 'name': 'msd'})  # Đổi phương pháp tính tương đồng
algo.fit(trainset)

# Hàm để gợi ý sản phẩm kết hợp
def get_combined_recommendations(user_id):
    user_items = transactions[transactions['UserId'] == user_id]['ProductId'].tolist()
    recommendations = set()
    
    # print(f"User's purchased items: {user_items}")

    # Collaborative Filtering
    item_inner_ids = [trainset.to_inner_iid(item) for item in user_items if item in trainset._raw2inner_id_items]
    
    # print(f"User's item inner IDs: {item_inner_ids}")
    
    for inner_id in item_inner_ids:
        neighbors = algo.get_neighbors(inner_id, k=40)
        neighbors_product_ids = [trainset.to_raw_iid(neighbor) for neighbor in neighbors]
        # print(f"Neighbors for item {trainset.to_raw_iid(inner_id)}: {neighbors_product_ids}")
        recommendations.update(neighbors_product_ids)
    
    # Content-Based Filtering
    for item in user_items:
        content_based_recommendations = get_content_based_recommendations(item)
        # print(f"Content-based recommendations for product {item}: {content_based_recommendations}")
        recommendations.update(content_based_recommendations)
    
    recommendations.difference_update(user_items)
    # print(f"Recommendations after filtering for UserId {user_id}: {recommendations}")
    return list(recommendations)[:40]
    
    
@app.route('/recommend/<string:user_id>', methods=['GET'])
def recommend(user_id):
    recommended_products = get_combined_recommendations(user_id)
    # print(f"Recommendations for user {user_id}: {recommended_products}")
    return jsonify(recommended_products)


if __name__ == '__main__':
    # app.run(debug=True)
     app.run(host='0.0.0.0', port=5000)