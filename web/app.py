from flask import Flask, jsonify, render_template
import pandas as pd
import numpy as np
import skfuzzy as fuzz
import matplotlib.pyplot as plt
import io
import base64
import os
import json
from sklearn.manifold import TSNE
import plotly.express as px


app = Flask(__name__)

# Load data dari Excel
def load_data(file_path):
    try:
        df = pd.read_excel(file_path, sheet_name="Sheet1")
        return df
    except Exception as e:
        print(f"Error loading Excel file: {e}")
        return None

# Fuzzy C-Means Clustering dengan semua fitur
def fuzzy_c_means_clustering(df):
    # Menggunakan semua kolom numerik yang relevan
    features = ['stok_awal', 'penerimaan', 'persediaan', 'pemakaian', 'sisa_stok', 'permintaan']
    data = df[features].values.T  # Data harus dalam bentuk transpose untuk FCM

    n_clusters = 5  # 5 kategori kebutuhan

    # Inisialisasi dan jalankan FCM
    cntr, u, _, _, _, _, _ = fuzz.cluster.cmeans(
        data, c=n_clusters, m=2, error=0.005, maxiter=1000, init=None
    )
    
    # Tentukan cluster dengan keanggotaan tertinggi
    cluster_labels = np.argmax(u, axis=0)
    df['cluster'] = cluster_labels

    # Beri kategori berdasarkan cluster
    kategori = ["Sangat Rendah", "Rendah", "Sedang", "Tinggi", "Sangat Tinggi"]
    df['kategori'] = df['cluster'].apply(lambda x: kategori[x])
    
    return df, cntr

# Hitung frekuensi kategori per wilayah dan tahun
def count_frequencies(df):
    # Simply count occurrences without any additional processing
    frequency_table = df.groupby(['wilayah', 'tahun', 'kategori']).size().reset_index(name='frekuensi')
    
    # Ensure we're actually counting properly
    # Check if any preprocessing is affecting the count
    print(f"Sample counts: {frequency_table['frekuensi'].value_counts()}")
    
    return frequency_table

# Plot hasil clustering
def plot_fuzzy_cmeans(df, cntr):
    plt.figure(figsize=(8, 6))
    
    # Scatter plot berdasarkan permintaan vs. persediaan (2D visualisasi sederhana)
    plt.scatter(df['permintaan'], df['persediaan'], c=df['cluster'], cmap='viridis', marker='o', label='Data')
    plt.scatter(cntr[:, 0], np.zeros_like(cntr[:, 0]), c='red', marker='x', s=100, label='Centroids')
 # Menggunakan indeks sesuai fitur
    
    plt.xlabel('Permintaan')
    plt.ylabel('Persediaan')
    plt.title('Hasil Clustering Fuzzy C-Means')
    plt.legend()

    # Simpan gambar ke buffer
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()
    
    return plot_url

def save_to_excel(df, file_name="hasil_fuzzy_cmeans.xlsx"):
    try:
        df.to_excel(file_name, index=False)
        print(f"Hasil clustering disimpan dalam {file_name}")
    except Exception as e:
        print(f"Gagal menyimpan file: {e}")

# Load hasil clustering dari Excel jika sudah ada
def load_existing_clustering(file_name="hasil_fuzzy_cmeans.xlsx"):
    if os.path.exists(file_name):
        try:
            df = pd.read_excel(file_name)
            print("Menggunakan hasil clustering yang sudah ada.")
            return df
        except Exception as e:
            print(f"Gagal membaca file: {e}")
    return None
# Fungsi untuk menghitung dan membuat plot frekuensi tingkat kebutuhan per tahun
def plot_frequencies_per_year(df):
    # Hitung jumlah status tingkat kebutuhan per tahun
    frequency_table = df.groupby(['tahun', 'kategori']).size().reset_index(name='frekuensi')
    
    # Filter untuk tahun yang diinginkan (2021-2024)
    tahun_filter = [2021, 2022, 2023, 2024]
    frequency_table = frequency_table[frequency_table['tahun'].isin(tahun_filter)]

    # Buat plot menggunakan Plotly
    fig = px.bar(frequency_table, 
                 x="tahun", 
                 y="frekuensi", 
                 color="kategori", 
                 title="Frekuensi Tingkat Kebutuhan per Tahun",
                 labels={"frekuensi": "Jumlah", "tahun": "Tahun", "kategori": "Kategori"},
                 barmode="group")  # Menampilkan kategori dalam grup per tahun

    return fig.to_json()  # Mengembalikan JSON untuk diproses di frontend


def plot_tsne_clustering(df):
    features = ['stok_awal', 'penerimaan', 'persediaan', 'pemakaian', 'sisa_stok', 'permintaan']
    X = df[features].values
    tsne = TSNE(n_components=2, random_state=42)
    X_embedded = tsne.fit_transform(X)
    
    df_tsne = pd.DataFrame(X_embedded, columns=['tsne_1', 'tsne_2'])
    df_tsne['cluster'] = df['cluster']

    # Menentukan label berdasarkan urutan cluster
    cluster_labels = {
        0: "Sangat Rendah",
        1: "Rendah",
        2: "Sedang",
        3: "Tinggi",
        4: "Sangat Tinggi"
    }
    
    df_tsne['cluster_label'] = df_tsne['cluster'].map(cluster_labels)

    # Mengatur urutan kategori secara eksplisit
    df_tsne['cluster_label'] = pd.Categorical(df_tsne['cluster_label'], 
                                              categories=["Sangat Rendah", "Rendah", "Sedang", "Tinggi", "Sangat Tinggi"], 
                                              ordered=True)

    fig = px.scatter(df_tsne, x='tsne_1', y='tsne_2', color=df_tsne['cluster_label'],
                     title="Visualisasi Clustering dengan t-SNE", labels={'color': 'Kategori Cluster'})

    return fig.to_json()


def count_frequencies_per_year(df):
    frequency_dict = {}
    for _, row in df.iterrows():
        wilayah = row['wilayah']
        tahun = row['tahun']
        kategori = row['kategori']
        
        if wilayah not in frequency_dict:
            frequency_dict[wilayah] = {}
        if tahun not in frequency_dict[wilayah]:
            frequency_dict[wilayah][tahun] = {}
        if kategori not in frequency_dict[wilayah][tahun]:
            frequency_dict[wilayah][tahun][kategori] = 0
        
        frequency_dict[wilayah][tahun][kategori] += 1
    
    return frequency_dict

@app.route('/')
def index():
    file_path = "data_fitri.xlsx"  # Ganti dengan path file yang benar
    result_file = "hasil_fuzzy_cmeans.xlsx"
    
    df_clustered = load_existing_clustering(result_file)
    
    if df_clustered is None:
        df = load_data(file_path)
        df_clustered, cntr = fuzzy_c_means_clustering(df)
        save_to_excel(df_clustered, result_file)
    
    frequencies = count_frequencies(df_clustered)
    plot_url = plot_fuzzy_cmeans(df_clustered, cntr) if 'cntr' in locals() else ""
    plot_tsne_json = plot_tsne_clustering(df_clustered)  # JSON Plotly untuk frontend
    plot_frequencies_json = plot_frequencies_per_year(df_clustered)  # JSON Plotly untuk grafik frekuensi

    return render_template("index.html", 
                           clusters=df_clustered.to_dict(orient='records'), 
                           frequencies=frequencies.to_dict(orient='records'), 
                           plot_url=plot_url,
                           plot_tsne_json=plot_tsne_json,
                           plot_frequencies_json=plot_frequencies_json)  # Kirim ke template


@app.route('/api/coordinates', methods=['GET'])
def get_coordinates():
    file_path = "coordinates.json"
    if not os.path.exists(file_path):
        return jsonify({"error": "File coordinates.json tidak ditemukan!"}), 404
    
    try:
        with open(file_path, "r") as file:
            coordinates_data = json.load(file)
        
        # Menghitung jumlah frekuensi per kategori setiap wilayah per tahun
        result_file = "hasil_fuzzy_cmeans.xlsx"
        df_clustered = load_existing_clustering(result_file)
        frequency_per_year = count_frequencies_per_year(df_clustered) if df_clustered is not None else {}
        
        # Gabungkan koordinat dengan data frekuensi
        for wilayah, coords in coordinates_data.items():
            if wilayah in frequency_per_year:
                coordinates_data[wilayah] = {
                    "coordinates": coords,
                    "frequencies": frequency_per_year[wilayah]
                }
        
        return jsonify(coordinates_data)
    except Exception as e:
        return jsonify({"error": f"Gagal membaca file: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(debug=True)
