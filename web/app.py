from flask import Flask, jsonify, render_template, request, redirect, url_for, flash, session
from flask_session import Session
from functools import wraps
import pandas as pd
import numpy as np
import skfuzzy as fuzz
import matplotlib.pyplot as plt
import io
import base64
import os
import json
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
from sklearn.manifold import TSNE


app = Flask(__name__)
app.config['SECRET_KEY'] = 'supersecretkey'
app.config['SESSION_TYPE'] = 'filesystem'  # Simpan session di server
app.config['SESSION_PERMANENT'] = False

Session(app)

# Decorator untuk memerlukan login
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'logged_in' not in session:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

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
    features = ['stok_awal', 'penerimaan', 'persediaan', 'pemakaian', 'sisa_stok', 'permintaan']

    # Konversi ke numerik dan tangani NaN
    df[features] = df[features].apply(pd.to_numeric, errors='coerce').fillna(0)

    # Ambil data sebagai NumPy array
    data = np.array(df[features], dtype=float).T

    n_clusters = 5
    cntr, u, _, _, _, _, _ = fuzz.cluster.cmeans(
        data, c=n_clusters, m=2, error=0.005, maxiter=1000, init=None
    )

    cluster_labels = np.argmax(u, axis=0)
    df['cluster'] = cluster_labels
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

def save_to_excel(df, file_name=None):
    try:
        # Generate filename with timestamp if not provided
        if file_name is None:
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            file_name = f"hasil_simulasi_{timestamp}.xlsx"
        
        # Make sure we're saving in the hasil_simulasi directory
        hasil_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hasil_simulasi")
        
        # Create the directory if it doesn't exist
        if not os.path.exists(hasil_dir):
            os.makedirs(hasil_dir)
            
        full_path = os.path.join(hasil_dir, file_name)
        
        # Sort data by date (tahun) before saving
        if 'tahun' in df.columns:
            df = df.sort_values(by=['tahun', 'wilayah', 'nama_obat'])
            
        # Save to Excel
        df.to_excel(full_path, index=False)
        print(f"Hasil clustering disimpan dalam {os.path.abspath(full_path)}")
        return file_name
    except Exception as e:
        print(f"Gagal menyimpan file: {e}")
        return None

# Load hasil clustering dari file jika sudah ada
def load_existing_clustering(file_name="hasil_simulasi.xlsx"):
    # If file_name doesn't contain a path, check in hasil_simulasi directory
    if not os.path.dirname(file_name):
        hasil_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hasil_simulasi")
        full_path = os.path.join(hasil_dir, file_name)
        if os.path.exists(full_path):
            file_name = full_path
    
    if not os.path.exists(file_name):
        # If the file still doesn't exist, check for default file
        default_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hasil_simulasi.xlsx")
        if os.path.exists(default_file):
            file_name = default_file
        else:
            default_csv = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hasil_simulasi.csv")
            if os.path.exists(default_csv):
                file_name = default_csv
            else:
                return None

    df = None
    try:
        # Prioritaskan membaca sebagai Excel jika ekstensi .xlsx
        if file_name.endswith('.xlsx'):
            df = pd.read_excel(file_name)
            # Periksa apakah hasilnya hanya satu kolom, yang menandakan format CSV tersembunyi
            if len(df.columns) == 1:
                print("File .xlsx terdeteksi memiliki format CSV, mencoba membaca ulang...")
                # Coba baca file yang sama sebagai CSV
                df = pd.read_csv(file_name, sep=';', decimal=',')
        # Jika file adalah .csv, baca dengan format yang benar
        elif file_name.endswith('.csv'):
            df = pd.read_csv(file_name, sep=';', decimal=',')
        
        if df is not None:
            print(f"Berhasil memuat data dari {file_name}")
            return df

    except Exception as e:
        print(f"Terjadi kesalahan saat membaca {file_name}: {e}")
        # Jika semua gagal, coba metode paling dasar sebagai fallback
        try:
            print("Mencoba metode fallback untuk membaca file...")
            df = pd.read_csv(file_name, sep=';', decimal=',')
            print(f"Berhasil memuat data dari {file_name} dengan metode fallback.")
            return df
        except Exception as e2:
            print(f"Gagal total memuat data dari {file_name}: {e2}")

    return None

# Function to get a list of saved simulation files
def get_saved_simulation_files():
    hasil_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hasil_simulasi")
    
    # Create the directory if it doesn't exist
    if not os.path.exists(hasil_dir):
        os.makedirs(hasil_dir)
        return []
    
    # Get all Excel files in the directory
    files = [f for f in os.listdir(hasil_dir) if f.endswith('.xlsx') and f.startswith('hasil_simulasi_')]
    
    # Sort files by date (newest first)
    files.sort(reverse=True)
    
    # Return full paths
    return [{'name': f, 'path': os.path.join(hasil_dir, f), 'date': f.split('_')[2:4]} for f in files]
# Fungsi untuk menghitung dan membuat plot frekuensi tingkat kebutuhan per tahun
def prepare_frequencies_for_echarts(df):
    if df is None or df.empty:
        return {}

    # Hitung frekuensi
    freq_df = df.groupby(['tahun', 'kategori']).size().reset_index(name='frekuensi')
    
    # Dapatkan daftar tahun dan kategori unik
    years = sorted(freq_df['tahun'].unique().tolist())
    categories = sorted(freq_df['kategori'].unique().tolist())

    series_data = []
    for category in categories:
        # Ambil data untuk kategori ini
        category_data = freq_df[freq_df['kategori'] == category]
        # Buat daftar frekuensi untuk setiap tahun
        freq_by_year = [int(category_data[category_data['tahun'] == year]['frekuensi'].values[0]) if year in category_data['tahun'].values else 0 for year in years]
        
        series_data.append({
            'name': category,
            'type': 'bar',
            'stack': 'total',
            'emphasis': {
                'focus': 'series'
            },
            'data': freq_by_year
        })

    return {
        'legend': {'data': categories},
        'xAxis': {'type': 'category', 'data': years},
        'series': series_data
    }


def plot_frequencies_per_year(df):
    # Hitung jumlah status tingkat kebutuhan per tahun
    frequency_table = df.groupby(['tahun', 'kategori']).size().reset_index(name='frekuensi')

    # Filter untuk tahun yang diinginkan (2021-2024)
    tahun_filter = [2021, 2022, 2023]
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





def prepare_tsne_data_for_echarts(df):
    features = ['stok_awal', 'penerimaan', 'persediaan', 'pemakaian', 'sisa_stok', 'permintaan']
    data = df[features].values

    tsne = TSNE(n_components=2, perplexity=5, random_state=42, n_iter=300)
    tsne_results = tsne.fit_transform(data)

    df['tsne_x'] = tsne_results[:, 0]
    df['tsne_y'] = tsne_results[:, 1]

    # Siapkan data untuk ECharts
    echarts_data = {
        'categories': df['kategori'].unique().tolist(),
        'series': []
    }

    for category in echarts_data['categories']:
        category_df = df[df['kategori'] == category]
        series_data = []
        for _, row in category_df.iterrows():
            series_data.append({
                'value': [row['tsne_x'], row['tsne_y']],
                'name': f"{row['wilayah']} ({row['tahun']})"
            })
        
        echarts_data['series'].append({
            'name': category,
            'type': 'scatter',
            'data': series_data,
            'emphasis': {
                'focus': 'series'
            }
        })

    return echarts_data  # Return dict, not JSON string

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

# Fungsi untuk menyiapkan data t-SNE untuk ECharts
def prepare_tsne_for_echarts(df):
    if 'tsne_x' not in df.columns or 'tsne_y' not in df.columns:
        return {}

    categories = df['kategori'].unique().tolist()
    series_data = []

    for category in categories:
        category_df = df[df['kategori'] == category]
        data_points = []
        for _, row in category_df.iterrows():
            # Pastikan tsne_x dan tsne_y adalah float
            tsne_x = float(row['tsne_x']) if pd.notna(row['tsne_x']) else 0.0
            tsne_y = float(row['tsne_y']) if pd.notna(row['tsne_y']) else 0.0
            data_points.append({
                'value': [tsne_x, tsne_y],
                'name': f"{row['nama_obat']} ({row['wilayah']}, {row['tahun']})"
            })
        
        series_data.append({
            'name': category,
            'type': 'scatter',
            'data': data_points,
            'emphasis': {
                'focus': 'series'
            }
        })

    return {
        'legend': {'data': categories},
        'series': series_data
    }

# Rute untuk halaman login
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        # Kredensial statis admin/admin
        if username == 'admin' and password == 'admin':
            session['logged_in'] = True
            session['username'] = username
            flash('Login berhasil!', 'success')
            return redirect(url_for('simulasi'))
        else:
            flash('Username atau password salah!', 'danger')
    
    return render_template('login.html')

# Rute untuk logout
@app.route('/logout')
def logout():
    session.pop('logged_in', None)
    session.pop('username', None)
    flash('Anda telah berhasil logout!', 'info')
    return redirect(url_for('login'))

@app.route('/statistic')
@login_required
def statistic():
    simulasi_file = "hasil_simulasi.xlsx"  # Sesuaikan dengan nama file yang benar

    # Coba baca hasil dari simulasi
    df_simulasi = load_existing_clustering(simulasi_file)

    if df_simulasi is None:
        # Jika file simulasi tidak ada, tampilkan halaman kosong dengan pesan
        flash('File hasil_simulasi.xlsx tidak ditemukan. Silakan jalankan simulasi terlebih dahulu.', 'warning')
        return render_template("index.html", 
                               clusters=[], 
                               plot_tsne_json='{}', 
                               plot_frequencies_json='{}')

    # Siapkan data untuk ECharts
    echarts_tsne_json = json.dumps(prepare_tsne_for_echarts(df_simulasi))
    echarts_freq_json = json.dumps(prepare_frequencies_for_echarts(df_simulasi))

    # Kirim data ke template
    return render_template("index.html", 
                           clusters=df_simulasi.to_dict(orient='records'), 
                           echarts_tsne_json=echarts_tsne_json,
                           echarts_freq_json=echarts_freq_json)


@app.route('/api/coordinates', methods=['GET'])
@login_required
def get_coordinates():
    file_path = "coordinates.json"
    if not os.path.exists(file_path):
        return jsonify({"error": "File coordinates.json tidak ditemukan!"}), 404

    try:
        with open(file_path, "r") as file:
            coordinates_data = json.load(file)

        # Menghitung jumlah frekuensi per kategori setiap wilayah per tahun dari hasil simulasi
        result_file = "hasil_simulasi.xlsx"  # Sesuaikan dengan nama file yang benar
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

# Fuzzy C-Means Clustering dengan iterasi untuk simulasi
def fuzzy_c_means_clustering_with_iterations(df, n_clusters=5, m=2, error=0.005, maxiter=1000):
    features = ['stok_awal', 'penerimaan', 'persediaan', 'pemakaian', 'sisa_stok', 'permintaan']

    # Konversi ke numerik dan tangani NaN
    df[features] = df[features].apply(pd.to_numeric, errors='coerce').fillna(0)

    # Ambil data sebagai NumPy array
    data = np.array(df[features], dtype=float).T

    # Inisialisasi untuk iterasi manual
    # Use maxiter=1 instead of 0 to avoid the u2 undefined error
    cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
        data, c=n_clusters, m=m, error=error, maxiter=1, init=None
    )

    # Simpan hasil setiap iterasi
    iterations = []

    # Iterasi manual untuk menyimpan hasil setiap langkah
    i = 0
    while i < maxiter:
        u0 = u.copy()

        # Update keanggotaan (calculate_u_and_d)
        # Hitung jarak Manhattan antara data dan pusat cluster
        d = np.zeros((n_clusters, data.shape[1]))
        for j in range(n_clusters):
            d[j] = np.sum(np.abs(data - np.atleast_2d(cntr[j]).T), axis=0)

        # Hitung matriks keanggotaan baru
        u = np.zeros((n_clusters, data.shape[1]))
        for j in range(n_clusters):
            u[j] = 1.0 / np.sum((d[j] / d) ** (2 / (m - 1)), axis=0)

        # Handle divide by zero errors
        u = np.nan_to_num(u, nan=1.0)

        # Normalisasi agar jumlah keanggotaan setiap data = 1
        u = u / np.sum(u, axis=0, keepdims=True)

        # Update pusat cluster (calculate_cluster_centers)
        um = u ** m
        cntr = np.dot(um, data.T) / np.sum(um, axis=1, keepdims=True)

        # Hitung fungsi objektif (calculate_objective_function)
        jm = np.sum(um * d ** 2)

        # Hitung koefisien partisi fuzzy (calculate_partition_coefficient)
        fpc = np.sum(u ** 2) / data.shape[1]

        # Simpan hasil iterasi
        iteration_result = {
            'iteration': i + 1,
            'centers': cntr.copy(),
            'membership': u.copy(),
            'objective_function': jm,
            'partition_coefficient': fpc,
            'error': np.linalg.norm(u - u0)
        }
        iterations.append(iteration_result)

        # Cek konvergensi
        if np.linalg.norm(u - u0) < error:
            break

        i += 1

    # Tentukan cluster untuk setiap data
    cluster_labels = np.argmax(u, axis=0)
    df_result = df.copy()
    df_result['cluster'] = cluster_labels
    kategori = ["Sangat Rendah", "Rendah", "Sedang", "Tinggi", "Sangat Tinggi"]
    df_result['kategori'] = df_result['cluster'].apply(lambda x: kategori[x])

    return df_result, cntr, iterations



# Rute untuk halaman simulasi
@app.route('/simulasi', methods=['GET', 'POST'])
@login_required
def simulasi():
    # Handle loading of previous results if requested
    load_file = request.args.get('load_file')
    if load_file:
        df_clustered = load_existing_clustering(load_file)
        if df_clustered is None:
            flash(f'Gagal memuat file simulasi: {load_file}', 'danger')
            return redirect(url_for('simulasi'))
            
        # Get saved files for selection
        saved_files = get_saved_simulation_files()
        
        # Calculate t-SNE for visualization if not present
        if 'tsne_x' not in df_clustered.columns or 'tsne_y' not in df_clustered.columns:
            features = ['stok_awal', 'penerimaan', 'persediaan', 'pemakaian', 'sisa_stok', 'permintaan']
            data = df_clustered[features].values
            tsne = TSNE(n_components=2, perplexity=min(5, len(df_clustered) - 1), random_state=42, n_iter=300)
            tsne_results = tsne.fit_transform(data)
            df_clustered['tsne_x'] = tsne_results[:, 0]
            df_clustered['tsne_y'] = tsne_results[:, 1]
            
        # Process for display
        final_tsne_json = prepare_tsne_data_for_echarts(df_clustered)
        frequencies_json = prepare_frequencies_for_echarts(df_clustered)
        
        flash(f'Berhasil memuat hasil simulasi dari: {load_file}', 'success')
        return render_template('simulasi.html',
                           clusters=df_clustered.to_dict(orient='records'),
                           parameters={'n_clusters': 5, 'm': 2, 'error': 0.005, 'maxiter': 100},
                           uploaded_filename=load_file,
                           iterations=[],
                           data_preview=True,
                           final_tsne_json=final_tsne_json,
                           frequencies_json=frequencies_json,
                           saved_files=saved_files)
        
    if request.method == 'POST':
        # Check if this is upload from "Load Data Tersimpan" tab
        mode = request.args.get('mode')
        if mode == 'load':
            # Handle file upload with format seperti hasil_simulasi.csv
            format_file = request.files.get('format_file')
            if not format_file or not format_file.filename:
                flash('File tidak diunggah. Silakan unggah file dengan format yang sesuai.', 'danger')
                return redirect(url_for('simulasi'))
                
            file_name = format_file.filename
            try:
                # Attempt to read file based on extension
                if file_name.endswith('.csv'):
                    df_clustered = pd.read_csv(format_file, sep=';', decimal=',')
                else:  # Try as Excel
                    df_clustered = pd.read_excel(format_file)
                    
                # Verify required columns exist
                required_columns = ['id', 'nama_obat', 'satuan', 'stok_awal', 'penerimaan', 'persediaan', 
                                 'pemakaian', 'sisa_stok', 'permintaan', 'wilayah', 'tahun', 'cluster']
                
                # If there are missing columns, reject the file
                missing_columns = [col for col in required_columns if col not in df_clustered.columns]
                if missing_columns:
                    flash(f'Format file tidak sesuai. Kolom yang tidak ditemukan: {", ".join(missing_columns)}', 'danger')
                    return redirect(url_for('simulasi'))
                    
                # Check for tsne_x and tsne_y, calculate if missing
                if 'tsne_x' not in df_clustered.columns or 'tsne_y' not in df_clustered.columns:
                    features = ['stok_awal', 'penerimaan', 'persediaan', 'pemakaian', 'sisa_stok', 'permintaan']
                    data = df_clustered[features].values
                    tsne = TSNE(n_components=2, perplexity=min(5, len(df_clustered) - 1), random_state=42, n_iter=300)
                    tsne_results = tsne.fit_transform(data)
                    df_clustered['tsne_x'] = tsne_results[:, 0]
                    df_clustered['tsne_y'] = tsne_results[:, 1]
                    
                # Check for kategori column, set to default if missing
                if 'kategori' not in df_clustered.columns:
                    df_clustered['kategori'] = 'Tidak Terkategori'
                
                # Get saved files for selection
                saved_files = get_saved_simulation_files()
                
                # Process for display
                final_tsne_json = prepare_tsne_data_for_echarts(df_clustered)
                frequencies_json = prepare_frequencies_for_echarts(df_clustered)
                
                flash(f'Berhasil memuat data dari file: {file_name}', 'success')
                return render_template('simulasi.html',
                                   clusters=df_clustered.to_dict(orient='records'),
                                   parameters={'n_clusters': 5, 'm': 2, 'error': 0.005, 'maxiter': 100},
                                   uploaded_filename=file_name,
                                   iterations=[],
                                   data_preview=True,
                                   final_tsne_json=final_tsne_json,
                                   frequencies_json=frequencies_json,
                                   saved_files=saved_files)
                                   
            except Exception as e:
                flash(f'Gagal memproses file: {str(e)}', 'danger')
                return redirect(url_for('simulasi'))
        
        # Normal simulation flow for Upload New Data tab
        n_clusters = int(request.form.get('n_clusters', 5))
        m = float(request.form.get('m', 2))
        error = float(request.form.get('error', 0.005))
        maxiter = int(request.form.get('maxiter', 100))

        uploaded_file = request.files.get('data_file')
        if not uploaded_file or not uploaded_file.filename:
            flash('File tidak diunggah. Silakan unggah file Excel.', 'danger')
            return redirect(url_for('simulasi'))

        file_name = uploaded_file.filename
        try:
            df = pd.read_excel(uploaded_file)
        except Exception as e:
            flash(f'Gagal membaca file Excel: {e}', 'danger')
            return redirect(url_for('simulasi'))

        if df is None:
            flash('Gagal memproses file. Pastikan formatnya benar.', 'danger')
            return redirect(url_for('simulasi'))

        df_clustered, cntr, iterations = fuzzy_c_means_clustering_with_iterations(
            df, n_clusters, m, error, maxiter
        )

        # Hitung t-SNE dan tambahkan ke DataFrame sebelum menyimpan
        features = ['stok_awal', 'penerimaan', 'persediaan', 'pemakaian', 'sisa_stok', 'permintaan']
        data = df_clustered[features].values
        tsne = TSNE(n_components=2, perplexity=min(5, len(df_clustered) - 1), random_state=42, n_iter=300)
        tsne_results = tsne.fit_transform(data)
        df_clustered['tsne_x'] = tsne_results[:, 0]
        df_clustered['tsne_y'] = tsne_results[:, 1]

        # Simpan hasil ke Excel (sekarang dengan koordinat t-SNE)
        save_to_excel(df_clustered, "hasil_simulasi.xlsx")

        iteration_tables = []
        for i, it in enumerate(iterations):
            iteration_tables.append({
                'iteration': i + 1,
                'objective_function': it['objective_function'],
                'partition_coefficient': it['partition_coefficient'],
                'error': it['error']
            })

        frequencies = count_frequencies(df_clustered)
        final_tsne_json = prepare_tsne_data_for_echarts(df_clustered.copy())

        session['results'] = {
            'final_tsne_json': final_tsne_json,
            'clusters': df_clustered.to_dict(orient='records'),
            'frequencies': frequencies.to_dict(orient='records'),
            'iterations': iteration_tables,
            'data_preview': df.head(10).to_html(classes='table table-striped w-full'),
            'parameters': {'n_clusters': n_clusters, 'm': m, 'error': error, 'maxiter': maxiter},
            'uploaded_filename': file_name
        }
        
        return redirect(url_for('simulasi'))

    results = session.pop('results', None)
    if results:
        return render_template(
            "simulasi.html",
            clusters=results['clusters'],
            frequencies=results['frequencies'],
            iterations=results['iterations'],
            iteration_plots=[],
            final_tsne_json=results.get('final_tsne_json', '{}'),
            data_preview=results['data_preview'],
            parameters=results['parameters'],
            uploaded_filename=results['uploaded_filename']
        )
    else:
        default_parameters = {
            'n_clusters': 5, 'm': 2, 'error': 0.005, 'maxiter': 100
        }
        return render_template("simulasi.html", parameters=default_parameters, clusters=[], final_tsne_json='{}', data_preview=None, uploaded_filename=None)

@app.route('/reset', methods=['POST'])
@login_required
def reset_simulation():
    # Hapus file jika ada
    file_path = 'hasil_simulasi.xlsx'
    file_deleted = False
    if os.path.exists(file_path):
        os.remove(file_path)
        file_deleted = True

    # Hapus cache hasil dari session
    session_cleared = 'results' in session
    session.pop('results', None)

    if file_deleted or session_cleared:
        flash('Proses simulasi dan cache hasil telah berhasil direset.', 'success')
    else:
        flash('Tidak ada yang perlu direset.', 'info')
        
    return redirect(url_for('simulasi'))

# Rute default untuk mengarahkan ke login
@app.route('/')
def root():
    return redirect(url_for('login'))

if __name__ == '__main__':
    app.run(debug=True, port=5001)
