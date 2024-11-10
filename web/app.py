from flask import Flask, render_template
import pandas as pd
import numpy as np
import skfuzzy as fuzz
import json
import plotly
import plotly.express as px
from sklearn.manifold import TSNE

app = Flask(__name__)

# Load the data from the Excel file
file_path = 'data_dan_inisialisasi.xlsx'
data = pd.read_excel(file_path)

# Function to preprocess the data
def preprocess_data():
    features = data[['Stok Awal', 'Permintaan Puskesmas', 'Pemakaian', 'Sisa Stok']].values
    return features

# Function to extract initial membership values from the data
def get_initial_membership():
    initial_membership = data[['Sangat Tinggi', 'Tinggi', 'Berpotensi Tinggi', 'Sedang', 'Rendah', 'Sangat Rendah']].values.T
    return initial_membership[:4]

# Fuzzy C-Means clustering function
def fuzzy_cmeans_clustering(data, initial_membership, n_clusters=4):
    data = data.T
    cntr, u, _, _, _, _, _ = fuzz.cluster.cmeans(data, n_clusters, 2, error=0.005, maxiter=1000, init=initial_membership)
    cluster_labels = np.argmax(u, axis=0)
    return cluster_labels, cntr, u

# Function to map cluster labels to categories
def map_clusters_to_categories(cluster_labels):
    categories = {0: "Rendah", 1: "Cukup", 2: "Tinggi", 3: "Sangat Tinggi"}
    return [categories[label] for label in cluster_labels]

@app.route('/')
def home():
    features = preprocess_data()
    initial_membership = get_initial_membership()
    
    # Perform clustering
    cluster_labels, centers, membership_values = fuzzy_cmeans_clustering(features, initial_membership)
    category_labels = map_clusters_to_categories(cluster_labels)
    
    # Add clustering results to the data
    data['Tingkat Kebutuhan'] = category_labels
    membership_values_per_obat = membership_values.T.tolist()
    data['Centroid Membership'] = membership_values_per_obat
    
    # Prepare data for visualizations

    # Frequency Bar Chart
    frequency = data['Tingkat Kebutuhan'].value_counts().reset_index()
    frequency.columns = ['Tingkat Kebutuhan', 'Frekuensi']
    fig_bar = px.bar(frequency, x='Tingkat Kebutuhan', y='Frekuensi', title='Frekuensi Kebutuhan Obat')

    # Scatter Plot based on Centroid x and y
    fig_scatter = px.scatter(
        data, x='Stok Awal', y='Permintaan Puskesmas',
        color='Tingkat Kebutuhan', title='Scatter Plot Kebutuhan Obat',
        hover_data=['Pemakaian', 'Sisa Stok']
    )

    # TSNE Visualization
    num_samples = data.shape[0]
    # Set perplexity to a value less than num_samples
    perplexity_value = min(30, num_samples // 3)  # Adjust as necessary

    # Handle case when num_samples is less than 3
    if perplexity_value < 1:
        perplexity_value = 1

    # Compute TSNE
    tsne = TSNE(n_components=2, perplexity=perplexity_value, random_state=42)
    tsne_results = tsne.fit_transform(features)
    data['TSNE1'] = tsne_results[:,0]
    data['TSNE2'] = tsne_results[:,1]

    # Generate TSNE Plot
    fig_tsne = px.scatter(
        data, x='TSNE1', y='TSNE2',
        color='Tingkat Kebutuhan', title='Visualisasi TSNE Kebutuhan Obat',
        hover_data=['Stok Awal', 'Permintaan Puskesmas', 'Pemakaian', 'Sisa Stok']
    )

    # Convert figures to JSON
    bar_graphJSON = json.dumps(fig_bar, cls=plotly.utils.PlotlyJSONEncoder)
    scatter_graphJSON = json.dumps(fig_scatter, cls=plotly.utils.PlotlyJSONEncoder)
    tsne_graphJSON = json.dumps(fig_tsne, cls=plotly.utils.PlotlyJSONEncoder)

    # Prepare data for table
    table_data = data.to_dict(orient='records')

    return render_template(
        'index.html',
        bar_graphJSON=bar_graphJSON,
        scatter_graphJSON=scatter_graphJSON,
        tsne_graphJSON=tsne_graphJSON,
        table_data=table_data
    )

if __name__ == '__main__':
    app.run(debug=True)
