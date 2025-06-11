from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import joblib
from keras.models import load_model
import pickle
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# === INIT: Load data saat server pertama kali dijalankan ===
supermarket_df = pd.read_csv('data/supermarket_encoded.csv')
transaction_df = pd.read_csv('data/transactions.csv')
product_df = pd.read_csv('data/product.csv')

user_encoder = joblib.load('models/user_encoder.pkl')
product_encoder = joblib.load('models/product_encoder.pkl')

model = load_model('models/supermarket_recommender.keras')

num_items = len(product_encoder.classes_)
all_products = np.arange(num_items)

with open('models/content_based_model.pkl', 'rb') as f:
    products, cosine_sim, product_indices = pickle.load(f)

cb_data = pd.read_csv('data/supermarket_encoded.csv')

@app.route('/')
def index():
    return "<H1>Flask Endpoint Supermarket Recomender : online</H1>"

def get_image_url(name):
    safe_name = name.replace(' ', '_').lower()
    return f"/images/{safe_name}.jpg"

def enrich_recommendations(df):
    records = df.to_dict(orient='records')
    for r in records:
        r['image'] = get_image_url(r['name'])
    return records

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    raw_user_id = data.get('id_user')
    count_items = int(data.get('count_items', 10))

    if int(raw_user_id) > 60:
        return showTopProduct(count_items, user_id=raw_user_id)

    try:
        target_user = user_encoder.transform([raw_user_id])[0]

        supermarket_df['user_id_enc'] = user_encoder.transform(supermarket_df['id_user'])
        supermarket_df['product_id_enc'] = product_encoder.transform(supermarket_df['id_product'])

        rated_products = supermarket_df[supermarket_df['user_id_enc'] == target_user]['product_id_enc'].tolist()
        unrated_products = np.setdiff1d(np.arange(len(product_encoder.classes_)), rated_products)

        if len(unrated_products) == 0:
            return jsonify({
                "user_id": raw_user_id,
                "message": "Semua produk telah dibeli oleh user ini.",
                "recommendations": []
            })

        user_array = np.full(len(unrated_products), target_user)
        pred_ratings = model.predict([user_array, unrated_products]).flatten()

        results = pd.DataFrame({
            'product_id_enc': unrated_products,
            'trusted_score': pred_ratings,
        })
        results['id_product'] = product_encoder.inverse_transform(results['product_id_enc'])

        results = results.merge(
            supermarket_df[['id_product', 'name', 'category', 'harga']].drop_duplicates(),
            on='id_product',
            how='left'
        )

        top_recommendations = results.sort_values('trusted_score', ascending=False).head(count_items)
        enriched_recs = enrich_recommendations(top_recommendations)
        print(f"User {raw_user_id} recommendations: {top_recommendations[['name','trusted_score']].head()}")

        return jsonify({
            'user_id': raw_user_id,
            'message': 'Recommendation',
            'recommendations': enriched_recs
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/predictcb', methods=['POST'])
def predictcb():
    data = request.get_json()
    raw_user_id = data.get('id_user')
    count_items = int(data.get('count_items', 10))

    try:
        user_id_encoded = user_encoder.transform([raw_user_id])[0]
        user_products = cb_data[cb_data['user_id_enc'] == user_id_encoded]['product_id_enc'].unique()

        scores = {}
        for pid in user_products:
            idx = product_indices.get(pid)
            if idx is not None:
                for i, score in enumerate(cosine_sim[idx]):
                    pid_similar = products.iloc[i]['product_id_enc']
                    if pid_similar not in user_products:
                        scores[pid_similar] = scores.get(pid_similar, 0) + score

        if not scores:
            return showTopProduct(count_items)

        recommended_ids = sorted(scores, key=scores.get, reverse=True)[:count_items]
        recommended_products = products.set_index('product_id_enc').loc[recommended_ids].reset_index()
        recommended_products['trusted_score'] = recommended_products['product_id_enc'].map(scores)
        recommended_products = recommended_products.sort_values(by='trusted_score', ascending=False)
        recommended_products['id_product'] = product_encoder.inverse_transform(recommended_products['product_id_enc'])

        recommended_products = recommended_products.merge(
            product_df[['id_product', 'harga']],
            on='id_product',
            how='left'
        )

        enriched_recs = enrich_recommendations(recommended_products)

        return jsonify({
            'user_id': raw_user_id,
            'message': 'Content-Based Recommendation',
            'recommendations': enriched_recs
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 400

def showTopProduct(count_items=10, user_id=None):
    try:
        product_sales = transaction_df.groupby('id_product')['quantity'].sum().reset_index()
        product_sales = product_sales.sort_values(by='quantity', ascending=False)

        top_products = product_sales.merge(
            product_df[['id_product', 'name', 'category', 'harga']],
            on='id_product',
            how='left'
        )

        top_products = top_products.dropna(subset=['harga'])

        top_products = top_products.head(count_items)
        top_products['trusted_score'] = top_products['quantity']

        enriched_recs = enrich_recommendations(top_products)

        return jsonify({
            'user_id': user_id,
            'message': 'Bestseller',
            'recommendations': enriched_recs
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 400
    
# === Run Server ===
if __name__ == '__main__':
    app.run(port=5001)
