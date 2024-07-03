import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

def get_content_based_recommendations(product_id, products):
    products['description'] = products['Title'] + ' ' + products['CategoryId'].astype(str)
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(products['description'])
    cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
    
    idx = products[products['Id'] == product_id].index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:6]
    product_indices = [i[0] for i in sim_scores]
    return products['Id'].iloc[product_indices].tolist()
