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
import concurrent.futures

app = Flask(__name__)

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
    try:
        with requests.head(url) as response:
            return int(response.headers.get('Content-Length', 0))
    except:
        return 0

def download_file(url, file_path):
    try:
        with requests.get(url, stream=True) as response:
            response.raise_for_status()
            total_size = int(response.headers.get('Content-Length', 0))
            downloaded_size = 0
            with open(file_path, 'wb') as file:
                for chunk in response.iter_content(chunk_size=8192):
                    with crawl_lock:
                        if not crawl_status['is_running']:
                            return False
                        
                        while crawl_status['is_paused']:
                            time.sleep(1)
                            if not crawl_status['is_running']:
                                return False

                    if chunk:
                        file.write(chunk)
                        downloaded_size += len(chunk)
                        with crawl_lock:
                            crawl_status['downloaded_size'] = downloaded_size
                            crawl_status['progress'] = (downloaded_size / total_size) * 100 if total_size > 0 else 0
        return True
    except Exception as e:
        print(f"Download error: {e}")
        return False

executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)

@app.route('/')
def home():
    return render_template('index.html')

def async_crawl(base_url, files_to_download, tags):
    global crawl_status
    
    try:
        os.makedirs('data/extracted', exist_ok=True)
        
        for file in files_to_download:
            with crawl_lock:
                if not crawl_status['is_running']:
                    break
                
                while crawl_status['is_paused']:
                    time.sleep(1)
                    if not crawl_status['is_running']:
                        return
            
            with crawl_lock:
                crawl_status['current_file'] = file
                crawl_status['progress'] = 0
                crawl_status['downloaded_size'] = 0
                crawl_status['total_size'] = get_file_size(f"{base_url}/{file}")
            
            file_path = f"data/extracted/{file}"
            
            if download_file(f"{base_url}/{file}", file_path):
                if crawl_status['is_running']:
                    with py7zr.SevenZipFile(file_path, mode='r') as z:
                        z.extractall(path='data/extracted')
    
    except Exception as e:
        print(f"Crawling error: {e}")
    
    finally:
        with crawl_lock:
            crawl_status['is_running'] = False
            crawl_status['is_paused'] = False
            crawl_status['current_file'] = ''
            crawl_status['progress'] = 0

@app.route('/crawl', methods=['POST'])
def crawl():
    data = request.json
    base_url = data.get('base_url')
    files_to_download = data.get('files_to_download')
    tags = data.get('tags', [])
    if isinstance(tags, str):
        tags = [tag.strip() for tag in tags.split(',')]

    if not base_url or not files_to_download:
        return jsonify({"error": "Base URL and files to download are required"}), 400

    try:
        os.makedirs('data', exist_ok=True)
        logistic_path, linear_path = generate_datasets(base_url, files_to_download, tags)
        return jsonify({
            "message": "Data crawling completed",
            "logistic_regression_dataset": logistic_path,
            "linear_regression_dataset": linear_path
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

def process_data(base_url, files_to_download, tags):
    os.makedirs('data/extracted', exist_ok=True)
    
    crawl_status['is_running'] = True
    async_crawl(base_url, files_to_download, tags)
    
    posts_file_path = 'data/extracted/cleaned_questions_final.csv'
    users_split_path = 'data/extracted/users_split'

    if not os.path.exists(posts_file_path):
        raise FileNotFoundError(f"Posts file not found at {posts_file_path}")
    
    if not os.path.exists(users_split_path) or not os.listdir(users_split_path):
        raise FileNotFoundError(f"Users split files not found in {users_split_path}")

    df = pd.read_csv(posts_file_path)
    
    df['code_snippet'] = df['Body'].apply(lambda x: 1 if re.search(r'<code>.?</code>', str(x), re.DOTALL) else 0)
    df['question_line_count'] = df['Body'].apply(lambda x: len(str(x).splitlines()))
    df['code_line_count'] = df['Body'].apply(lambda x: sum(len(code_block.splitlines()) for code_block in re.findall(r'<code>(.?)</code>', str(x), re.DOTALL)))
    df['image'] = df['Body'].apply(lambda x: 1 if re.search(r'<img\s+[^>]*src=', str(x)) else 0)
    
    df['AnswerCount'] = pd.to_numeric(df['AnswerCount'], errors='coerce').fillna(0)
    df['ViewCount'] = pd.to_numeric(df['ViewCount'], errors='coerce').fillna(0)
    df['pd_score'] = (df['AnswerCount'] / df['ViewCount']) * 100

    user_files = [os.path.join(users_split_path, f) for f in os.listdir(users_split_path) if f.endswith('.csv')]
    list_of_user_dfs = [pd.read_csv(file) for file in user_files]
    users_df = pd.concat(list_of_user_dfs, ignore_index=True)
    users_df.rename(columns={'Id': 'OwnerUserId'}, inplace=True)
    
    merged_df = pd.merge(df, users_df[['OwnerUserId', 'Reputation']], on='OwnerUserId', how='left')

    def categorize_reputation(reputation):
        if reputation >= 2400:
            return 'High'
        elif 400 <= reputation < 2400:
            return 'Mid'
        elif 1 < reputation < 400:
            return 'Low'
        else:
            return None

    merged_df['ReputationCategory'] = merged_df['Reputation'].apply(categorize_reputation)
    merged_df['answered?'] = merged_df['AnswerCount'].apply(lambda x: 1 if x > 0 else 0)

    columns_to_drop = ['PostTypeId', 'AcceptedAnswerId', 'OwnerUserId', 'AnswerCount',
                       'FavoriteCount', 'CommunityOwnedDate', 'CreationDate', 'Score', 'Title', 'Body']
    merged_df.drop(columns=columns_to_drop, inplace=True)

    merged_df['Reputation'] = pd.to_numeric(merged_df['Reputation'], errors='coerce')
    merged_df = merged_df[merged_df['Reputation'] > 0].dropna(subset=['Reputation', 'ReputationCategory'])

    if tags and isinstance(tags, list) and tags[0]:
        merged_df = merged_df[merged_df['Tags'].str.contains('|'.join(tags), na=False)]

    return merged_df

def generate_datasets(base_url, files_to_download, tags):
    os.makedirs('data/stackoverflow', exist_ok=True)

    merged_df = process_data(base_url, files_to_download, tags)

    logistic_regression_df = merged_df[['Id', 'CommentCount', 'ViewCount', 'Tags', 
                                        'code_snippet', 'question_line_count', 
                                        'code_line_count', 'image', 'pd_score', 
                                        'Reputation', 'ReputationCategory', 'answered?']]

    logistic_regression_path = 'data/stackoverflow/logisticRegression.csv'
    logistic_regression_df.to_csv(logistic_regression_path, index=False)

    linear_regression_df = merged_df[['Reputation', 'pd_score']]
    linear_regression_path = 'data/stackoverflow/linearRegression.csv'
    linear_regression_df.to_csv(linear_regression_path, index=False)

    return logistic_regression_path, linear_regression_path

@app.route('/pause_crawl', methods=['POST'])
def pause_crawl():
    global crawl_status
    with crawl_lock:
        if crawl_status['is_running']:
            crawl_status['is_paused'] = True
            return jsonify({
                "message": "Crawling paused", 
                "status": {
                    "is_paused": True, 
                    "current_file": crawl_status['current_file'], 
                    "progress": crawl_status['progress']
                }
            })
    return jsonify({"error": "No crawling in progress"}), 400

@app.route('/resume_crawl', methods=['POST'])
def resume_crawl():
    global crawl_status
    with crawl_lock:
        if crawl_status['is_running'] and crawl_status['is_paused']:
            crawl_status['is_paused'] = False
            return jsonify({
                "message": "Crawling resumed", 
                "status": {
                    "is_paused": False, 
                    "current_file": crawl_status['current_file'], 
                    "progress": crawl_status['progress']
                }
            })
    return jsonify({"error": "No paused crawling in progress"}), 400

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

@app.route('/upload_dataset', methods=['POST'])
def upload_dataset():
    if 'file1' not in request.files or 'file2' not in request.files:
        return jsonify({"error": "Both datasets are required"}), 400

    file1 = request.files['file1']
    file2 = request.files['file2']

    if file1.filename == '' or file2.filename == '':
        return jsonify({"error": "No selected file"}), 400

    try:
        os.makedirs('data/stackoverflow', exist_ok=True)

        df1_path = os.path.join('data/stackoverflow', 'logisticRegression.csv')
        file1.save(df1_path)
        df1 = pd.read_csv(df1_path)

        required_cols1 = [
            'Id', 'CommentCount', 'ViewCount', 'Tags', 'code_snippet', 
            'question_line_count', 'code_line_count', 'image', 
            'pd_score', 'Reputation', 'ReputationCategory', 'answered?'
        ]
        if not all(col in df1.columns for col in required_cols1):
            os.remove(df1_path)
            return jsonify({"error": "First dataset does not have required columns"}), 400

        df2_path = os.path.join('data/stackoverflow', 'linearRegression.csv')
        file2.save(df2_path)
        df2 = pd.read_csv(df2_path)

        required_cols2 = ['Reputation', 'pd_score']
        if not all(col in df2.columns for col in required_cols2):
            os.remove(df1_path)
            os.remove(df2_path)
            return jsonify({"error": "Second dataset does not have required columns"}), 400

        return jsonify({"message": "Datasets uploaded successfully"}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 

@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.json
    analysis_type = data.get("analysis_type")

    if analysis_type == "linear_regression":
        try:
            merged_df = pd.read_csv('data/stackoverflow/linearRegression.csv')
            
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
            merged_df = pd.read_csv('data/stackoverflow/logisticRegression.csv')
            
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