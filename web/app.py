from flask import Flask, render_template
import pandas as pd
import json
import plotly.express as px
import plotly.utils
from sklearn.manifold import TSNE
from geopy.geocoders import Nominatim
import plotly.express as px

import folium
import pandas as pd
import numpy as np
import skfuzzy as fuzz
import pickle
import os
app = Flask(__name__)

# Load the data
file_path = 'data_fitri.xlsx'  # Update with your file path
data = pd.read_excel(file_path)
coordinates_file_path = 'coordinates.json'
# Function to preprocess the data
def preprocess_data():
    features = data[['stok_awal', 'permintaan', 'pemakaian', 'sisa_stok']].values
    return features

# Fuzzy C-Means clustering function
def fuzzy_cmeans_clustering(data, n_clusters=4):
    data = data.T  # Transpose for clustering
    cntr, u, _, _, _, _, _ = fuzz.cluster.cmeans(data, n_clusters, 2, error=0.005, maxiter=1000, init=None)
    cluster_labels = np.argmax(u, axis=0)
    return cluster_labels, cntr, u

# Save trained model
def save_model(centers, membership_values, filename='fcm_model.pkl'):
    with open(filename, 'wb') as f:
        pickle.dump({'centers': centers, 'membership_values': membership_values}, f)

# Load trained model
def load_model(filename='fcm_model.pkl'):
    with open(filename, 'rb') as f:
        return pickle.load(f)

# Function to predict clusters for new data
def predict_cluster(new_data, centers):
    new_data = new_data.T
    u, _, _, _, _, _ = fuzz.cluster.cmeans_predict(new_data, centers, 2, error=0.005, maxiter=1000)
    return np.argmax(u, axis=0)

def load_coordinates():
    if os.path.exists(coordinates_file_path):
        with open(coordinates_file_path, 'r') as f:
            return json.load(f)
    return {}

# Function to save coordinates to JSON file
def save_coordinates(coords):
    with open(coordinates_file_path, 'w') as f:
        json.dump(coords, f)

@app.route('/')
def home():
    # Proses data seperti biasa
    features = preprocess_data()
    cluster_labels, centers, membership_values = fuzzy_cmeans_clustering(features)
    save_model(centers, membership_values)
    
    categories = {0: "Rendah", 1: "Cukup", 2: "Tinggi", 3: "Sangat Tinggi"}
    category_labels = [categories[label] for label in cluster_labels]
    
    data['Tingkat_Kebutuhan'] = category_labels
    
    # Menghitung frekuensi per wilayah
    frequency_per_region = data.groupby(['wilayah', 'Tingkat_Kebutuhan']).size().reset_index(name='Frekuensi')

    # Grafik bar untuk frekuensi per wilayah
    fig_bar = px.bar(frequency_per_region, 
                     x='wilayah', 
                     y='Frekuensi', 
                     color='Tingkat_Kebutuhan', 
                     title='Frekuensi per Wilayah')

    # Grafik scatter, misalnya untuk visualisasi data clustering
    fig_scatter = px.scatter(data, 
                             x='stok_awal', 
                             y='permintaan', 
                             color='Tingkat_Kebutuhan', 
                             title='Scatter Plot of Stok Awal vs Permintaan')

    # t-SNE untuk visualisasi dimensi tinggi (optional)
    tsne = TSNE(n_components=2)
    tsne_result = tsne.fit_transform(features)
    fig_tsne = px.scatter(x=tsne_result[:, 0], 
                          y=tsne_result[:, 1], 
                          title='t-SNE Projection')

    # Konversi grafik ke format JSON
    bar_graphJSON = json.dumps(fig_bar, cls=plotly.utils.PlotlyJSONEncoder)
    scatter_graphJSON = json.dumps(fig_scatter, cls=plotly.utils.PlotlyJSONEncoder)
    tsne_graphJSON = json.dumps(fig_tsne, cls=plotly.utils.PlotlyJSONEncoder)

    # Ambil koordinat wilayah
    coordinates = load_coordinates()
    geolocator = Nominatim(user_agent="geoapiExercises")
    
    def get_coordinates(region):
        if region in coordinates:
            return pd.Series(coordinates[region])
        try:
            location = geolocator.geocode(region)
            if location:
                coords = [location.latitude, location.longitude]
                coordinates[region] = coords
                save_coordinates(coordinates)
                return pd.Series(coords)
            else:
                return pd.Series([None, None])
        except:
            return pd.Series([None, None])

    coords = frequency_per_region['wilayah'].apply(get_coordinates)
    frequency_per_region[['Latitude', 'Longitude']] = coords

    map_center = [frequency_per_region['Latitude'].mean(), frequency_per_region['Longitude'].mean()]
    if np.isnan(map_center[0]) or np.isnan(map_center[1]):
        map_center = [0, 0]

    m = folium.Map(location=map_center, zoom_start=5)
    markers = []

    for _, row in frequency_per_region.iterrows():
        if pd.notnull(row['Latitude']) and pd.notnull(row['Longitude']):
            # Include frequency data for the marker popup
            popup_content = f"{row['wilayah']} - {row['Tingkat_Kebutuhan']} ({row['Frekuensi']})"
            markers.append({
                'lat': row['Latitude'],
                'lon': row['Longitude'],
                'popup': popup_content,
                'frequencies': row[['Tingkat_Kebutuhan', 'Frekuensi']].to_dict()  # Pass frequencies
            })

            folium.Marker(
                location=[row['Latitude'], row['Longitude']],
                popup=popup_content,
                tooltip=row['wilayah'],
                icon=folium.Icon(color="blue")
            ).add_to(m)
# Mengkonversi data frekuensi menjadi format JSON
    frequency_json = frequency_per_region.groupby('wilayah').apply(
        lambda x: x[['Tingkat_Kebutuhan', 'Frekuensi']].to_dict(orient='records')
    ).to_json()

    # Simpan peta sebagai HTML
    map_path = "static/map.html"
    m.save(map_path)

    # Kembalikan grafik dalam format JSON ke template
    return render_template('index.html', 
                           bar_graphJSON=bar_graphJSON, 
                           scatter_graphJSON=scatter_graphJSON, 
                           tsne_graphJSON=tsne_graphJSON, 
                           table_data=data.to_dict(orient='records'),
                           map_path=map_path,
                           frequency_json=frequency_json,
                           markers=markers)  # Send markers with frequency data

if __name__ == '__main__':
    app.run(debug=True)
