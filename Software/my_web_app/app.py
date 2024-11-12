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

app = Flask(__name__)
executor = ThreadPoolExecutor(max_workers=3)

try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')

def create_linear_regression_plot(X, y, regressor):
    """Fungsi helper untuk membuat plot"""
    plt.figure(figsize=(8, 6))
    plt.scatter(X, y, color='blue', label='Data points')
    plt.plot(X, regressor.predict(X), color='red', label='Regression line')
    plt.xlabel('Reputation')
    plt.ylabel('pd_score')
    plt.title('Linear Regression')
    plt.legend()
    
    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=300, bbox_inches='tight')
    plt.close()
    
    buf.seek(0)
    image_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    return f'data:image/png;base64,{image_base64}'

def download_file(url, file_path):
    """Fungsi helper untuk download file"""
    response = requests.get(url, stream=True)
    with open(file_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/crawl', methods=['POST'])
def crawl_data():
    def async_crawl():
        base_url = "https://archive.org/download/stackexchange"
        files_to_download = ['stackoverflow.com-Posts.7z', 'stackoverflow.com-Users.7z']
        
        os.makedirs('data', exist_ok=True)
        
        for file in files_to_download:
            file_path = f"data/{file}"
            download_file(f"{base_url}/{file}", file_path)
            
            with py7zr.SevenZipFile(file_path, mode='r') as z:
                z.extractall(path='data/extracted')
        
        return {"status": "completed"}
    
    executor.submit(async_crawl)
    return jsonify({"message": "Data crawling started"})

@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.json
    analysis_type = data.get("analysis_type")

    if analysis_type == "linear_regression":
        try:
            df = pd.read_csv('data/stackoverflow/pengaruhReputasi.csv')
            X = df[['Reputation']]
            y = df['pd_score']

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

            regressor = LinearRegression()
            regressor.fit(X_train, y_train)

            plot_url = create_linear_regression_plot(X, y, regressor)

            return jsonify({
                "coef": float(regressor.coef_[0]), 
                "intercept": float(regressor.intercept_), 
                "plot_url": plot_url
            })

        except Exception as e:
            return jsonify({"error": str(e)}), 500

    elif analysis_type == "logistic_regression":
        try:
            df = pd.read_csv('data/stackoverflow/cleaned_output_addcolumns_final_merged_cleaned_questions.csv')
            X = df[['code_snippet', 'image', 'Reputation', 'CommentCount', 'ViewCount', 
                   'question_line_count', 'code_line_count']]
            y = df['answered?']

            X = sm.add_constant(X)
            logit_model = sm.Logit(y, X)
            result = logit_model.fit(disp=0)

            coef_table_html = result.summary2().tables[1].to_html(
                classes='table table-striped table-hover table-bordered',
                float_format=lambda x: '{:.4f}'.format(x)
            )
            return Response(coef_table_html, mimetype='text/html')

        except Exception as e:
            return jsonify({"error": str(e)}), 500

    return jsonify({"message": "Invalid analysis type."}), 400

if __name__ == '__main__':
    app.run(debug=True, threaded=True)