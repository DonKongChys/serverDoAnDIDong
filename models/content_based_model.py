from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

class ContentBasedModel:
    def __init__(self, products):
        self.products = products
        self.tfidf = TfidfVectorizer(stop_words='english')
        self.tfidf_matrix = self._build_tfidf_matrix()
        self.cosine_sim = linear_kernel(self.tfidf_matrix, self.tfidf_matrix)

    def _build_tfidf_matrix(self):
        self.products['description'] = self.products['Title'] + ' ' + self.products['CategoryId'].astype(str)
        return self.tfidf.fit_transform(self.products['description'])

    def get_recommendations(self, product_id, top_n=100):
        idx = self.products[self.products['Id'] == product_id].index[0]
        sim_scores = list(enumerate(self.cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:top_n + 1]
        product_indices = [i[0] for i in sim_scores]
        return self.products['Id'].iloc[product_indices].tolist()