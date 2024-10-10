import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import statsmodels.api as sm
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve
)

st.set_page_config(page_title="Data Visualization Dashboard", layout="wide")

sns.set(style="whitegrid", color_codes=True)

st.title("React Questions Analysis Dashboard")

st.markdown("""
**Research Overview**

React, one of the most popular JavaScript libraries, often generates a multitude of technical questions from developers. These questions are frequently posted on discussion platforms like Stack Overflow. This research aims to analyze the complexity of React-related questions posted on Stack Overflow and evaluate their difficulty levels. The primary focus is on comparing the characteristics of answered and unanswered questions by measuring complexity based on various parameters such as question structure, usage of technical terminology, proficiency in writing source code, and the context of the questions.

The initial phase involves identifying the determinants of whether a React-related question receives an answer, including code competency, complexity, and the difficulty level of the questions. Utilizing a quantitative approach based on text mining and statistical analysis, the expected outcome is the identification of patterns that influence whether a question is answered. This research contributes to a deeper understanding of the factors affecting the resolution of technical issues within the React ecosystem, aiming to assist developers in formulating more effective questions and enhancing the community's experience in providing assistance.
""")

st.sidebar.title("Navigation")
sections = ["Data Overview", "Visualizations", "Machine Learning"]
choice = st.sidebar.radio("Go to", sections)

@st.cache_data
def load_data():
    try:
        df = pd.read_csv('cleaned_output_addcolumns_final_merged_cleaned_questions.csv')
        df = df.dropna()
        return df
    except FileNotFoundError:
        st.error("CSV file ''cleaned_output_addcolumns_final_merged_cleaned_questions.csv' was not found in the same directory. Make sure the file is in the correct location.")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Error reading CSV file: {e}")
        return pd.DataFrame()
df_clean = load_data()

if choice == "Data Overview":
    st.header("Dataset Overview")
    st.write(f"Dataset Shape: {df_clean.shape}")
    st.write("Columns in Dataset:", list(df_clean.columns))
    st.dataframe(df_clean.head())

elif choice == "Visualizations":
    st.header("Data Visualizations")
    
    # Distribution of Answered Questions
    st.subheader("Distribution of Answered Questions")
    fig1, ax1 = plt.subplots()
    sns.countplot(x='answered?', data=df_clean, palette='hls', ax=ax1)
    ax1.set_xlabel('Answered Status', fontsize=12)
    ax1.set_ylabel('Count', fontsize=12)
    ax1.set_title('Distribution of Answered Questions', fontsize=15)
    ax1.set_xticklabels(['Unanswered (0)', 'Answered (1)'])
    
    # Annotate bars
    for p in ax1.patches:
        ax1.annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2., p.get_height()),
                     ha='center', va='baseline', fontsize=11, color='black', xytext=(0, 5),
                     textcoords='offset points')
    
    st.pyplot(fig1)
    
    # Answer Distribution Based on Code Snippet
    st.subheader("Answer Distribution Based on Code Snippet")
    table_x1 = pd.crosstab(df_clean['code_snippet'], df_clean['answered?'])
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    table_x1.plot(kind='bar', color=['skyblue', 'salmon'], ax=ax2)
    ax2.set_title('Answer Distribution Based on Code Snippet', fontsize=16)
    ax2.set_xlabel('Code Snippet', fontsize=12)
    ax2.set_ylabel('Frequency of Answers', fontsize=12)
    ax2.set_xticklabels(['No Code Snippet (0)', 'Has Code Snippet (1)'], rotation=0)
    ax2.legend(title='Answered?', labels=['No', 'Yes'], fontsize=12, title_fontsize=12)
    
    # Annotate bars
    for container in ax2.containers:
        ax2.bar_label(container, fmt='%d', label_type='edge', fontsize=10, padding=3)
    
    st.pyplot(fig2)
    
    # Answer Distribution Based on Image
    st.subheader("Answer Distribution Based on Image")
    table_x2 = pd.crosstab(df_clean['image'], df_clean['answered?'])
    fig3, ax3 = plt.subplots(figsize=(10, 6))
    table_x2.plot(kind='bar', color=['lightcoral', 'lightseagreen'], ax=ax3)
    ax3.set_title('Answer Distribution Based on Image', fontsize=16)
    ax3.set_xlabel('Image', fontsize=12)
    ax3.set_ylabel('Frequency of Answers', fontsize=12)
    ax3.set_xticklabels(['No Image (0)', 'Has Image (1)'], rotation=0)
    ax3.legend(title='Answered?', labels=['No', 'Yes'], fontsize=12, title_fontsize=12)
    
    # Annotate bars
    for container in ax3.containers:
        ax3.bar_label(container, fmt='%d', label_type='edge', fontsize=10, padding=3)
    
    st.pyplot(fig3)
    
    # Logarithmic Reputation Distribution
    st.subheader("Logarithmic Reputation Distribution Based on Answered Status")
    df_clean['log_Reputation'] = np.log1p(df_clean['Reputation'])
    q99 = df_clean['log_Reputation'].quantile(0.99)
    df_subset = df_clean[df_clean['log_Reputation'] <= q99]
    
    fig4, ax4 = plt.subplots(figsize=(12, 6))
    sns.boxplot(x='answered?', y='log_Reputation', data=df_subset, palette='Blues', ax=ax4)
    ax4.set_title('Logarithmic Reputation Distribution Based on Answered Status', fontsize=16)
    ax4.set_xlabel('Answered Status (0 = Unanswered, 1 = Answered)', fontsize=12)
    ax4.set_ylabel('Log(Reputation)', fontsize=12)
    ax4.set_xticklabels(['Unanswered (0)', 'Answered (1)'])
    
    median_log_reputation = df_subset['log_Reputation'].median()
    ax4.axhline(median_log_reputation, color='red', linestyle='--', label='Median Log(Reputation)')
    ax4.text(0.5, median_log_reputation + 0.1, f'Median: {median_log_reputation:.2f}', color='red', fontsize=12, ha='center')
    
    st.pyplot(fig4)
    
    # Histogram of Log Reputation
    st.subheader("Reputation Distribution Plot")
    fig5, ax5 = plt.subplots()
    sns.histplot(df_clean['log_Reputation'], kde=True, ax=ax5)
    ax5.set_title('Reputation Distribution Plot', fontsize=16)
    ax5.set_xlabel('Log(Reputation)', fontsize=12)
    ax5.set_ylabel('Frequency', fontsize=12)
    st.pyplot(fig5)
    
    # Comment Count Distribution
    st.subheader("Comment Count Distribution Based on Answered Status")
    df_clean['sqrt_question_line_count'] = np.sqrt(df_clean['question_line_count'])
    df_clean['sqrt_code_line_count'] = np.sqrt(df_clean['code_line_count'])
    
    fig6, ax6 = plt.subplots(figsize=(12, 6))
    sns.boxplot(x='answered?', y='CommentCount', data=df_clean, palette='deep', ax=ax6)
    ax6.set_title('Comment Count Distribution Based on Answered Status', fontsize=16)
    ax6.set_xlabel('Answered Status (0 = Unanswered, 1 = Answered)', fontsize=12)
    ax6.set_ylabel('Number of Comments', fontsize=12)
    ax6.set_xticklabels(['Unanswered (0)', 'Answered (1)'])
    
    median_comment_count = df_clean['CommentCount'].median()
    ax6.axhline(median_comment_count, color='red', linestyle='--', label='Median Comment Count')
    ax6.text(0.5, median_comment_count + 10, f'Median: {median_comment_count:.0f}', color='red', fontsize=12, ha='center')
    
    st.pyplot(fig6)
    
    # Histogram of Comment Count
    st.subheader("Comment Count Distribution Plot")
    fig7, ax7 = plt.subplots()
    sns.histplot(df_clean['CommentCount'], kde=True, ax=ax7)
    ax7.set_title('Comment Count Distribution Plot', fontsize=16)
    ax7.set_xlabel('Number of Comments', fontsize=12)
    ax7.set_ylabel('Frequency', fontsize=12)
    st.pyplot(fig7)
    
    # View Count Distribution
    st.subheader("View Count Distribution Based on Answered Status")
    df_clean['sqrt_question_line_count'] = np.sqrt(df_clean['question_line_count'])
    df_clean['sqrt_code_line_count'] = np.sqrt(df_clean['code_line_count'])
    
    q99_view = df_clean['ViewCount'].quantile(0.99)
    df_view_subset = df_clean[df_clean['ViewCount'] <= q99_view]
    
    fig8, ax8 = plt.subplots(figsize=(12, 6))
    sns.boxplot(x='answered?', y='ViewCount', data=df_view_subset, palette='Blues', ax=ax8)
    ax8.set_title('View Count Distribution Based on Answered Status', fontsize=16)
    ax8.set_xlabel('Answered Status (0 = Unanswered, 1 = Answered)', fontsize=12)
    ax8.set_ylabel('Number of Views', fontsize=12)
    ax8.set_xticklabels(['Unanswered (0)', 'Answered (1)'])
    
    view_count_median = df_view_subset['ViewCount'].median()
    ax8.axhline(view_count_median, color='red', linestyle='--')
    ax8.text(0.5, view_count_median, f'Median: {view_count_median:.0f}', color='red', ha='center')
    
    st.pyplot(fig8)
    
    # Histogram of View Count
    st.subheader("View Count Distribution Plot")
    fig9, ax9 = plt.subplots()
    sns.histplot(df_clean['ViewCount'], kde=True, ax=ax9)
    ax9.set_title('View Count Distribution Plot', fontsize=16)
    ax9.set_xlabel('Number of Views', fontsize=12)
    ax9.set_ylabel('Frequency', fontsize=12)
    st.pyplot(fig9)
    
    # Boxplot for Question Line Count
    st.subheader("Question Line Count Distribution Based on Answered Status")
    q99_qlc = df_clean['sqrt_question_line_count'].quantile(0.99)
    df_qlc_subset = df_clean[df_clean['sqrt_question_line_count'] <= q99_qlc]
    
    fig10, ax10 = plt.subplots(figsize=(12, 6))
    sns.boxplot(x='answered?', y='sqrt_question_line_count', data=df_qlc_subset, palette='Blues', ax=ax10)
    ax10.set_title('Question Line Count Distribution Based on Answered Status', fontsize=16)
    ax10.set_xlabel('Answered Status (0 = Unanswered, 1 = Answered)', fontsize=12)
    ax10.set_ylabel('Number of Question Lines', fontsize=12)
    ax10.set_xticklabels(['Unanswered (0)', 'Answered (1)'])
    
    median_qlc = df_qlc_subset['sqrt_question_line_count'].median()
    ax10.axhline(median_qlc, color='red', linestyle='--')
    ax10.text(0.5, median_qlc, f'Median: {median_qlc:.0f}', color='red', ha='center')
    
    st.pyplot(fig10)
    
    # Histogram of Question Line Count
    st.subheader("Question Line Count Distribution Plot")
    fig11, ax11 = plt.subplots()
    sns.histplot(df_clean['sqrt_question_line_count'], kde=True, ax=ax11)
    ax11.set_title('Question Line Count Distribution Plot', fontsize=16)
    ax11.set_xlabel('Number of Question Lines', fontsize=12)
    ax11.set_ylabel('Frequency', fontsize=12)
    st.pyplot(fig11)
    
    # Boxplot for Code Line Count
    st.subheader("Code Line Count Distribution Based on Answered Status")
    q99_clc = df_clean['sqrt_code_line_count'].quantile(0.99)
    df_clc_subset = df_clean[df_clean['sqrt_code_line_count'] <= q99_clc]
    
    fig12, ax12 = plt.subplots(figsize=(12, 6))
    sns.boxplot(x='answered?', y='sqrt_code_line_count', data=df_clc_subset, palette='Blues', ax=ax12)
    ax12.set_title('Code Line Count Distribution Based on Answered Status', fontsize=16)
    ax12.set_xlabel('Answered Status (0 = Unanswered, 1 = Answered)', fontsize=12)
    ax12.set_ylabel('Number of Code Lines', fontsize=12)
    ax12.set_xticklabels(['Unanswered (0)', 'Answered (1)'])
    
    median_clc = df_clc_subset['sqrt_code_line_count'].median()
    ax12.axhline(median_clc, color='red', linestyle='--')
    ax12.text(0.5, median_clc, f'Median: {median_clc:.0f}', color='red', ha='center')
    
    st.pyplot(fig12)
    
    # Histogram of Code Line Count
    st.subheader("Code Line Count Distribution Plot")
    fig13, ax13 = plt.subplots()
    sns.histplot(df_clean['sqrt_code_line_count'], kde=True, ax=ax13)
    ax13.set_title('Code Line Count Distribution Plot', fontsize=16)
    ax13.set_xlabel('Number of Code Lines', fontsize=12)
    ax13.set_ylabel('Frequency', fontsize=12)
    st.pyplot(fig13)
    
    st.subheader("Feature Correlation Matrix")
    df_clean['log_Reputation'] = np.log1p(df_clean['Reputation'])
    df_clean['sqrt_question_line_count'] = np.sqrt(df_clean['question_line_count'])
    df_clean['sqrt_code_line_count'] = np.sqrt(df_clean['code_line_count'])
    
    X = df_clean[['code_snippet', 'image', 'log_Reputation', 'CommentCount', 'ViewCount',
                 'sqrt_question_line_count', 'sqrt_code_line_count']]
    
    correlation_matrix = X.corr()
    fig14, ax14 = plt.subplots(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', ax=ax14)
    ax14.set_title('Feature Correlation Matrix', fontsize=16)
    st.pyplot(fig14)

elif choice == "Machine Learning":
    st.header("Machine Learning Model")
    
    df_clean['log_Reputation'] = np.log1p(df_clean['Reputation'])
    df_clean['sqrt_question_line_count'] = np.sqrt(df_clean['question_line_count'])
    df_clean['sqrt_code_line_count'] = np.sqrt(df_clean['code_line_count'])
    
    cat_vars = ['Tags', 'ReputationCategory']
    for var in cat_vars:
        cat_list = pd.get_dummies(df_clean[var], prefix=var, sparse=True)
        df_clean = df_clean.join(cat_list)
    df_clean = df_clean.drop(columns=cat_vars)
    
    st.write("Dataset columns after creating dummy variables:", df_clean.columns.tolist())
    
    X = df_clean[['code_snippet', 'image', 'log_Reputation', 'CommentCount',
                 'ViewCount', 'sqrt_question_line_count', 'sqrt_code_line_count']]
    y = df_clean['answered?']
    
    scaler = StandardScaler()
    X_scaled = X.copy()
    X_scaled[['log_Reputation', 'CommentCount', 'ViewCount',
              'sqrt_question_line_count', 'sqrt_code_line_count']] = scaler.fit_transform(
        X[['log_Reputation', 'CommentCount', 'ViewCount',
           'sqrt_question_line_count', 'sqrt_code_line_count']]
    )
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.3, random_state=0
    )
    
    X_train_sm = sm.add_constant(X_train)
    X_test_sm = sm.add_constant(X_test)
    
    smote = SMOTE(random_state=0)
    X_res, y_res = smote.fit_resample(X_train_sm, y_train)
    
    logit_model = sm.Logit(y_res, X_res)
    result = logit_model.fit()
    st.subheader("Logistic Regression Model Summary")
    st.text(result.summary2())
    
    odds_ratio = np.exp(result.params)
    st.subheader("Odds Ratio")
    st.write(odds_ratio)
    
    logreg = LogisticRegression(max_iter=1000)
    logreg.fit(X_train, y_train)
    y_pred = logreg.predict(X_test)
    accuracy = logreg.score(X_test, y_test)
    st.subheader("Logistic Regression Classifier")
    st.write(f"Accuracy on test set: {accuracy:.2f}")
    
    cm = confusion_matrix(y_test, y_pred)
    fig15, ax15 = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['No', 'Yes'], yticklabels=['No', 'Yes'], ax=ax15)
    ax15.set_xlabel('Predicted')
    ax15.set_ylabel('Actual')
    ax15.set_title('Confusion Matrix Heatmap')
    st.pyplot(fig15)
    
    st.subheader("Classification Report")
    report = classification_report(y_test, y_pred, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    st.dataframe(report_df)
    
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    st.subheader("Precision and Recall")
    st.write(f"Precision: {precision:.2f}")
    st.write(f"Recall: {recall:.2f}")
    
    fig16, ax16 = plt.subplots(figsize=(6,6))
    ax16.scatter(precision, recall, color='green', s=100)
    ax16.set_xlim(0,1)
    ax16.set_ylim(0,1)
    ax16.set_xlabel('Precision', fontsize=12)
    ax16.set_ylabel('Recall', fontsize=12)
    ax16.set_title('Precision vs Recall', fontsize=16)
    ax16.grid(True)
    ax16.text(precision + 0.01, recall, f'P={precision:.2f}, R={recall:.2f}', color='green', fontsize=12, ha='left')
    st.pyplot(fig16)
    
    st.subheader("Receiver Operating Characteristic (ROC) Curve")
    logit_roc_auc = roc_auc_score(y_test, logreg.predict(X_test))
    fpr, tpr, thresholds = roc_curve(y_test, logreg.predict_proba(X_test)[:, 1])
    fig17, ax17 = plt.subplots()
    ax17.plot(fpr, tpr, label=f'Logistic Regression (AUC = {logit_roc_auc:.2f})')
    ax17.plot([0, 1], [0, 1], 'r--')
    ax17.set_xlim([0.0, 1.0])
    ax17.set_ylim([0.0, 1.05])
    ax17.set_xlabel('False Positive Rate', fontsize=12)
    ax17.set_ylabel('True Positive Rate', fontsize=12)
    ax17.set_title('Receiver Operating Characteristic', fontsize=16)
    ax17.legend(loc="lower right")
    st.pyplot(fig17)
    
    st.subheader("Variance Inflation Factor (VIF)")
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    vif_data = pd.DataFrame()
    vif_data["Feature"] = X.columns
    vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    st.dataframe(vif_data)

st.markdown("""
---
**Developed by [Vanesya Aura ~ L200210170]**

This dashboard visualizes data and builds a logistic regression model to predict whether questions are answered based on various features.
""")
