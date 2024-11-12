import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

st.set_page_config(page_title="Reputation vs pd_score Analysis Dashboard", layout="wide")

sns.set(style="whitegrid", color_codes=True)

st.title("Reputation vs pd_score Analysis Dashboard")

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
        df_reputasi = pd.read_csv('./Software/Dashboard-Streamlit/pengaruhReputasi.csv')
        df_reputasi = df_reputasi.dropna()
        return df_reputasi
    except FileNotFoundError:
        st.error("CSV file ''pengaruhReputasi.csv' was not found in the same directory. Make sure the file is in the correct location.")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Error reading CSV file: {e}")
        return pd.DataFrame()
df_reputasi = load_data()

if choice == "Data Overview":
    st.header("Dataset Overview")
    st.markdown("### Dataset Shape")
    st.write(f"**Rows:** {df_reputasi.shape[0]} | **Columns:** {df_reputasi.shape[1]}")
    
    st.markdown("### Columns in Dataset")
    st.write(list(df_reputasi.columns))
    
    st.markdown("### Sample Data")
    st.dataframe(df_reputasi.head())

elif choice == "Visualizations":
    st.header("Data Visualizations")
    
    st.subheader("pd_score Distribution Plot")
    fig1, ax1 = plt.subplots(figsize=(8, 6))
    sns.histplot(df_reputasi['pd_score'], kde=True, color='skyblue', ax=ax1)
    ax1.set_title('pd_score Distribution Plot', fontsize=16)
    ax1.set_xlabel('pd_score', fontsize=12)
    ax1.set_ylabel('Frequency', fontsize=12)
    st.pyplot(fig1)
    
    st.subheader("Reputation vs pd_score Scatter Plot")
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    sns.scatterplot(x='Reputation', y='pd_score', data=df_reputasi, color='blue', ax=ax2)
    ax2.set_title('Reputation vs pd_score', fontsize=16)
    ax2.set_xlabel('Reputation', fontsize=12)
    ax2.set_ylabel('pd_score', fontsize=12)
    st.pyplot(fig2)
    
    st.subheader("pd_score Box Plot")
    fig3, ax3 = plt.subplots(figsize=(8, 6))
    sns.boxplot(y='pd_score', data=df_reputasi, color='lightgreen', ax=ax3)
    ax3.set_title('pd_score Box Plot', fontsize=16)
    ax3.set_ylabel('pd_score', fontsize=12)
    st.pyplot(fig3)

elif choice == "Machine Learning":
    st.header("Machine Learning Model")
    
    X = df_reputasi[['Reputation']]
    y = df_reputasi['pd_score']  
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    
    regressor = LinearRegression()
    regressor.fit(X_train, y_train)
    
    y_pred_train = regressor.predict(X_train)
    y_pred_test = regressor.predict(X_test)
    
    st.subheader("Reputation vs pd_score (Training Set)")
    fig4, ax4 = plt.subplots(figsize=(8, 6))
    ax4.scatter(X_train, y_train, color='lightcoral', label='Actual')
    ax4.plot(X_train, y_pred_train, color='firebrick', label='Predicted')
    ax4.set_title('Reputation vs pd_score (Training Set)', fontsize=16)
    ax4.set_xlabel('Reputation', fontsize=12)
    ax4.set_ylabel('pd_score', fontsize=12)
    ax4.legend(title='Legend')
    st.pyplot(fig4)
    
    st.subheader("Reputation vs pd_score (Test Set)")
    fig5, ax5 = plt.subplots(figsize=(8, 6))
    ax5.scatter(X_test, y_test, color='lightcoral', label='Actual')
    ax5.plot(X_train, y_pred_train, color='firebrick', label='Predicted')
    ax5.set_title('Reputation vs pd_score (Test Set)', fontsize=16)
    ax5.set_xlabel('Reputation', fontsize=12)
    ax5.set_ylabel('pd_score', fontsize=12)
    ax5.legend(title='Legend')
    st.pyplot(fig5)
    
    st.subheader("Model Coefficients and Intercept")
    st.write(f"**Coefficient:** {regressor.coef_[0]:.4f}")
    st.write(f"**Intercept:** {regressor.intercept_:.4f}")
    
    st.subheader("Model Performance Metrics")
    from sklearn.metrics import mean_squared_error, r2_score
    
    mse_train = mean_squared_error(y_train, y_pred_train)
    mse_test = mean_squared_error(y_test, y_pred_test)
    r2_train = r2_score(y_train, y_pred_train)
    r2_test = r2_score(y_test, y_pred_test)
    
    st.write(f"**Training Set Mean Squared Error (MSE):** {mse_train:.4f}")
    st.write(f"**Test Set Mean Squared Error (MSE):** {mse_test:.4f}")
    st.write(f"**Training Set R-squared:** {r2_train:.4f}")
    st.write(f"**Test Set R-squared:** {r2_test:.4f}")
    
    st.subheader("Residuals Plot (Test Set)")
    residuals = y_test - y_pred_test
    fig6, ax6 = plt.subplots(figsize=(8,6))
    sns.scatterplot(x=y_pred_test, y=residuals, color='purple', ax=ax6)
    ax6.axhline(0, color='red', linestyle='--')
    ax6.set_title('Residuals Plot (Test Set)', fontsize=16)
    ax6.set_xlabel('Predicted pd_score', fontsize=12)
    ax6.set_ylabel('Residuals', fontsize=12)
    st.pyplot(fig6)

st.markdown("""
---
**Developed by [Vanesya Aura ~ L200210170]**

This dashboard visualizes data and builds a linear regression model to analyze the influence of Reputation on pd_score.
""")
