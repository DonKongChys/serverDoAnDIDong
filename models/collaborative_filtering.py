from models.content_based_filtering import get_content_based_recommendations
from surprise import Dataset, Reader, KNNBasic

def get_combined_recommendations(user_id, products, transactions):
    reader = Reader(rating_scale=(1, 1))
    dataset = Dataset.load_from_df(transactions[['UserId', 'ProductId', 'Quantity']], reader)
    trainset = dataset.build_full_trainset()
    algo = KNNBasic(sim_options={'user_based': False, 'name': 'msd'})
    algo.fit(trainset)
    
    user_items = transactions[transactions['UserId'] == user_id]['ProductId'].tolist()
    recommendations = set()
    
    item_inner_ids = [trainset.to_inner_iid(item) for item in user_items if item in trainset._raw2inner_id_items]
    for inner_id in item_inner_ids:
        neighbors = algo.get_neighbors(inner_id, k=40)
        neighbors_product_ids = [trainset.to_raw_iid(neighbor) for neighbor in neighbors]
        recommendations.update(neighbors_product_ids)
    
    for item in user_items:
        content_based_recommendations = get_content_based_recommendations(item, products)
        recommendations.update(content_based_recommendations)
    
    recommendations.difference_update(user_items)
    return list(recommendations)[:40]
