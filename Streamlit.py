import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

# Load the data
df = pd.read_csv("./products.csv")

# Subset the data for faster processing (optional)
desc = df.head(300)

# TF-IDF Vectorization
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(desc['product_description'])

# Fit KMeans clustering model
true_k = 10
model = KMeans(n_clusters=true_k, init='k-means++', max_iter=100, n_init=1)
model.fit(tfidf_matrix)

# Get terms from TF-IDF vectorizer
terms = tfidf.get_feature_names_out()

# Function to print top terms in each cluster
def print_cluster(i):
    for ind in order_centroids[i, :10]:
        st.write(terms[ind])

# Function to show recommendations
def show_recommendations(product):
    Y = tfidf.transform([product])
    prediction = model.predict(Y)
    print_cluster(prediction[0])

# Get order of centroids
order_centroids = model.cluster_centers_.argsort()[:, ::-1]

# Custom CSS for styling
st.markdown(
    """
    <style>
    .title {
        font-family: 'Arial', sans-serif;
        font-size: 36px;
        color: #333;
        margin-bottom: 30px;
        text-align: center;
    }
    .input-text {
        width: 100%;
        padding: 15px;
        font-size: 18px;
        border: 2px solid #ccc;
        border-radius: 8px;
        margin-bottom: 30px;
    }
    .button {
        background-color: #4CAF50;
        color: white;
        padding: 15px 30px;
        text-align: center;
        text-decoration: none;
        display: inline-block;
        font-size: 18px;
        border: none;
        border-radius: 8px;
        cursor: pointer;
        transition: background-color 0.3s;
    }
    .button:hover {
        background-color: #45a049;
    }
    .recommendation-title {
        font-family: 'Arial', sans-serif;
        font-size: 24px;
        color: #333;
        margin-bottom: 20px;
    }
    .recommendation {
        font-size: 18px;
        margin-bottom: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Streamlit app
st.title('Product Recommendation System')
st.markdown('<div class="title">Product Recommendation System</div>', unsafe_allow_html=True)

# Sidebar for user input
product_input = st.text_input('Enter a product description:', '')

# Show recommendations when user inputs a product description
if product_input:
    st.markdown('<div class="title">Recommended products:</div>', unsafe_allow_html=True)
    show_recommendations(product_input)
