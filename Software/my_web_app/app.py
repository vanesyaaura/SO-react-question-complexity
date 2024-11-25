from flask import Flask, request, jsonify, render_template, Response
import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import requests
from bs4 import BeautifulSoup
import py7zr
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
import os
from io import BytesIO
import base64
from concurrent.futures import ThreadPoolExecutor
import nltk
import threading
import time
import numpy as np
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import seaborn as sns

app = Flask(__name__)
executor = ThreadPoolExecutor(max_workers=3)

crawl_status = {
    'is_running': False,
    'is_paused': False,
    'progress': 0,
    'current_file': '',
    'total_size': 0,
    'downloaded_size': 0
}

crawl_lock = threading.Lock()

try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')

def create_linear_regression_plot(X, y, X_train, y_train, regressor):
    """Helper function untuk membuat plot regresi linear"""
    plt.figure(figsize=(10, 6))
    plt.scatter(X, y, color='lightcoral')
    plt.plot(X_train, regressor.predict(X_train), color='firebrick')
    plt.title('Reputation vs pd_score')
    plt.xlabel('Reputation')
    plt.ylabel('pd_score')
    plt.box(False)
    
    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=300, bbox_inches='tight')
    plt.close()
    
    buf.seek(0)
    image_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    return f'data:image/png;base64,{image_base64}'

def get_file_size(url):
    """Get file size before downloading"""
    response = requests.head(url)
    return int(response.headers.get('content-length', 0))

def download_file(url, file_path):
    """Enhanced download function with progress tracking"""
    global crawl_status
    
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    block_size = 8192
    downloaded_size = 0

    with open(file_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=block_size):
            if chunk:
                while crawl_status['is_paused']:
                    time.sleep(1)
                    if not crawl_status['is_running']:
                        return False
                
                if not crawl_status['is_running']:
                    return False
                
                f.write(chunk)
                downloaded_size += len(chunk)
                with crawl_lock:
                    crawl_status['downloaded_size'] = downloaded_size
                    crawl_status['progress'] = (downloaded_size / total_size) * 100
    
    return True

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/crawl', methods=['POST'])
def crawl_data():
    global crawl_status
    
    with crawl_lock:
        if crawl_status['is_running']:
            return jsonify({"error": "Crawling is already in progress"}), 400
        crawl_status['is_running'] = True
        crawl_status['is_paused'] = False
        crawl_status['progress'] = 0
        crawl_status['current_file'] = ''
        crawl_status['total_size'] = 0
        crawl_status['downloaded_size'] = 0

    def async_crawl():
        global crawl_status
        base_url = "https://archive.org/download/stackexchange"
        files_to_download = ['stackoverflow.com-Posts.7z', 'stackoverflow.com-Users.7z']
        
        try:
            os.makedirs('data', exist_ok=True)
            
            for file in files_to_download:
                if not crawl_status['is_running']:
                    break
                    
                with crawl_lock:
                    crawl_status['current_file'] = file
                    crawl_status['progress'] = 0
                    crawl_status['downloaded_size'] = 0
                    crawl_status['total_size'] = get_file_size(f"{base_url}/{file}")
                
                file_path = f"data/{file}"
                if download_file(f"{base_url}/{file}", file_path):
                    if crawl_status['is_running']:
                        with py7zr.SevenZipFile(file_path, mode='r') as z:
                            z.extractall(path='data/extracted')
        
        finally:
            with crawl_lock:
                crawl_status['is_running'] = False
                crawl_status['is_paused'] = False
    
    executor.submit(async_crawl)
    return jsonify({"message": "Data crawling started"})

@app.route('/pause_crawl', methods=['POST'])
def pause_crawl():
    global crawl_status
    with crawl_lock:
        if crawl_status['is_running']:
            crawl_status['is_paused'] = True
            return jsonify({"message": "Crawling paused"})
    return jsonify({"error": "No crawling in progress"}), 400

@app.route('/resume_crawl', methods=['POST'])
def resume_crawl():
    global crawl_status
    with crawl_lock:
        if crawl_status['is_running']:
            crawl_status['is_paused'] = False
            return jsonify({"message": "Crawling resumed"})
    return jsonify({"error": "No crawling in progress"}), 400

@app.route('/stop_crawl', methods=['POST'])
def stop_crawl():
    global crawl_status
    with crawl_lock:
        if crawl_status['is_running']:
            crawl_status['is_running'] = False
            crawl_status['is_paused'] = False
            return jsonify({"message": "Crawling stopped"})
    return jsonify({"error": "No crawling in progress"}), 400

@app.route('/crawl_status', methods=['GET'])
def get_crawl_status():
    with crawl_lock:
        return jsonify(crawl_status)

@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.json
    analysis_type = data.get("analysis_type")

    if analysis_type == "linear_regression":
        try:
            merged_df = pd.read_csv('data/stackoverflow/cleaned_output_addcolumns_final_merged_cleaned_questions.csv')
            
            merged_df['pd_score'] = pd.to_numeric(merged_df['pd_score'], errors='coerce')
            merged_df = merged_df.dropna(subset=['Reputation', 'pd_score'])
            
            X = merged_df[['Reputation']]
            y = merged_df['pd_score']
            
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=0
            )
            
            regressor = LinearRegression()
            regressor.fit(X_train, y_train)
            
            y_pred_train = regressor.predict(X_train)
            y_pred_test = regressor.predict(X_test)
            
            plot_url = create_linear_regression_plot(X_test, y_test, X_train, y_train, regressor)

            return jsonify({
                "coef": float(regressor.coef_[0]),
                "intercept": float(regressor.intercept_),
                "plot_url": plot_url
            })

        except Exception as e:
            return jsonify({"error": str(e)}), 500

    elif analysis_type == "logistic_regression":
        try:
            merged_df = pd.read_csv('data/stackoverflow/cleaned_output_addcolumns_final_merged_cleaned_questions.csv')
            
            merged_df['log_Reputation'] = np.log1p(merged_df['Reputation'])
            merged_df['sqrt_question_line_count'] = np.sqrt(merged_df['question_line_count'])
            merged_df['sqrt_code_line_count'] = np.sqrt(merged_df['code_line_count'])
            
            X = merged_df[['code_snippet', 'image', 'log_Reputation', 'CommentCount', 
                          'ViewCount', 'sqrt_question_line_count', 'sqrt_code_line_count']]
            y = merged_df['answered?']
            
            scaler = StandardScaler()
            numerical_cols = ['log_Reputation', 'CommentCount', 'ViewCount', 
                            'sqrt_question_line_count', 'sqrt_code_line_count']
            X.loc[:, numerical_cols] = scaler.fit_transform(X[numerical_cols])
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
            
            os = SMOTE(random_state=0)
            X_train_balanced, y_train_balanced = os.fit_resample(X_train, y_train)
            
            X_train_balanced = sm.add_constant(X_train_balanced)
            logit_model = sm.Logit(y_train_balanced, X_train_balanced)
            result = logit_model.fit()
            
            summary_df = pd.DataFrame({
                'Variable': result.params.index,
                'Coefficient': result.params.values,
                'Std Error': result.bse.values,
                'z-value': result.tvalues.values,
                'P-value': result.pvalues.values,
                'Odds Ratio': np.exp(result.params.values)
            })
            
            summary_html = """
            <div style="padding: 20px;">
                <h3>Logistic Regression Results</h3>
                {}
                <p style="margin-top: 20px;"><strong>Note:</strong> Odds ratios > 1 indicate positive relationship with getting answers</p>
            </div>
            """.format(
                summary_df.to_html(
                    classes='table table-striped table-hover table-bordered',
                    float_format=lambda x: '{:.4f}'.format(x)
                )
            )
            
            return Response(summary_html, mimetype='text/html')

        except Exception as e:
            return jsonify({"error": str(e)}), 500

    return jsonify({"message": "Invalid analysis type."}), 400

if __name__ == '__main__':
    app.run(debug=True, threaded=True)