import pickle
from surprise import Dataset, Reader, KNNBasic

class CollaborativeFilteringModel:
    def __init__(self, transactions):
        self.transactions = transactions
        self.reader = Reader(rating_scale=(1, 5))
        self.dataset = self._build_dataset()
        self.trainset = self.dataset.build_full_trainset()
        self.algo = self._load_or_train_model()

    def _build_dataset(self):
        return Dataset.load_from_df(self.transactions[['UserId', 'ProductId', 'Quantity']], self.reader)

    def _load_or_train_model(self):
        try:
            with open('models/cf_model.pkl', 'rb') as f:
                return pickle.load(f)
        except FileNotFoundError:
            print("Không tìm thấy mô hình đã lưu. Tiến hành huấn luyện...")
            algo = KNNBasic(sim_options={'user_based': False, 'name': 'msd'})
            algo.fit(self.trainset)
            with open('models/cf_model.pkl', 'wb') as f:
                pickle.dump(algo, f)
            return algo

    def get_recommendations(self, user_id, top_n=40):
        user_items = self.transactions[self.transactions['UserId'] == user_id]['ProductId'].tolist()
        recommendations = set()
        item_inner_ids = [self.trainset.to_inner_iid(item) for item in user_items if
                           item in self.trainset._raw2inner_id_items]
        for inner_id in item_inner_ids:
            neighbors = self.algo.get_neighbors(inner_id, k=top_n)
            neighbors_product_ids = [self.trainset.to_raw_iid(neighbor) for neighbor in neighbors]
            recommendations.update(neighbors_product_ids)
        recommendations.difference_update(user_items)
        return list(recommendations)[:top_n]