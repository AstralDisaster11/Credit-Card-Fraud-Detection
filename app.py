import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import plotly.express as px
import plotly.graph_objects as go

# Page configuration
st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown('<p class="main-header">💳 Credit Card Fraud Detection System</p>', unsafe_allow_html=True)

# Initialize session state
if 'models_trained' not in st.session_state:
    st.session_state.models_trained = False
if 'train_data' not in st.session_state:
    st.session_state.train_data = None
if 'test_data' not in st.session_state:
    st.session_state.test_data = None

# Sidebar
with st.sidebar:
    st.header("📊 Navigation")
    page = st.radio("Select Page", 
                    ["Data Upload", "Data Exploration", "Model Training", "Predictions", "Model Comparison"])
    
    st.markdown("---")
    st.header("ℹ️ About")
    st.info("""
    This app detects fraudulent credit card transactions using machine learning algorithms.
    
    **Features:**
    - Upload and explore datasets
    - Train multiple ML models
    - Compare model performance
    - Make predictions on new data
    """)

def cleaning_data(df):
    """Clean and preprocess the data"""
    df = df.copy()
    
    # Drop unnecessary columns
    cols_to_drop = ['Unnamed: 0', 'cc_num', 'first', 'last', 'street', 
                    'city', 'state', 'zip', 'dob', 'trans_num', 'trans_date_trans_time']
    
    existing_cols = [col for col in cols_to_drop if col in df.columns]
    df.drop(existing_cols, axis=1, inplace=True)
    
    # Drop null values
    df.dropna(inplace=True)
    
    return df

def encode_data(df):
    """Encode categorical variables"""
    df = df.copy()
    encoder = LabelEncoder()
    
    categorical_cols = ['merchant', 'category', 'gender', 'job']
    
    for col in categorical_cols:
        if col in df.columns:
            df[col] = encoder.fit_transform(df[col].astype(str))
    
    return df

# PAGE 1: DATA UPLOAD
if page == "Data Upload":
    st.header("📁 Upload Your Data")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Training Data")
        train_file = st.file_uploader("Upload Training CSV", type=['csv'], key='train')
        
        if train_file is not None:
            st.session_state.train_data = pd.read_csv(train_file)
            st.success(f"✅ Training data loaded: {st.session_state.train_data.shape[0]} rows, {st.session_state.train_data.shape[1]} columns")
            
            with st.expander("Preview Training Data"):
                st.dataframe(st.session_state.train_data.head(10))
    
    with col2:
        st.subheader("Test Data")
        test_file = st.file_uploader("Upload Test CSV", type=['csv'], key='test')
        
        if test_file is not None:
            st.session_state.test_data = pd.read_csv(test_file)
            st.success(f"✅ Test data loaded: {st.session_state.test_data.shape[0]} rows, {st.session_state.test_data.shape[1]} columns")
            
            with st.expander("Preview Test Data"):
                st.dataframe(st.session_state.test_data.head(10))
    
    if st.session_state.train_data is not None:
        st.info("💡 Data loaded successfully! Navigate to 'Data Exploration' to analyze your data.")

# PAGE 2: DATA EXPLORATION
elif page == "Data Exploration":
    st.header("🔍 Data Exploration")
    
    if st.session_state.train_data is None:
        st.warning("⚠️ Please upload training data first!")
    else:
        train = st.session_state.train_data.copy()
        
        tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "📈 Statistics", "🔥 Heatmaps", "⚖️ Class Distribution"])
        
        with tab1:
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Dataset Info")
                st.write(f"**Shape:** {train.shape[0]} rows × {train.shape[1]} columns")
                st.write(f"**Missing Values:** {train.isnull().sum().sum()}")
                
                st.subheader("Data Types")
                dtype_df = pd.DataFrame({
                    'Column': train.dtypes.index,
                    'Type': train.dtypes.values
                })
                st.dataframe(dtype_df, use_container_width=True)
            
            with col2:
                st.subheader("First 10 Rows")
                st.dataframe(train.head(10), use_container_width=True)
        
        with tab2:
            st.subheader("Statistical Summary")
            st.dataframe(train.describe().T, use_container_width=True)
            
            # Numerical columns analysis
            numerical_cols = train.select_dtypes(include=[np.number]).columns.tolist()
            
            if len(numerical_cols) > 0:
                st.subheader("Distribution of Numerical Features")
                selected_col = st.selectbox("Select column to visualize", numerical_cols)
                
                fig = px.histogram(train, x=selected_col, nbins=50, 
                                 title=f"Distribution of {selected_col}")
                st.plotly_chart(fig, use_container_width=True)
        
        with tab3:
            st.subheader("Missing Values Heatmap")
            
            fig, ax = plt.subplots(figsize=(12, 6))
            sns.heatmap(train.isnull(), cbar=True, yticklabels=False, cmap='viridis', ax=ax)
            plt.title("Missing Values Heatmap")
            st.pyplot(fig)
            
            # Correlation heatmap for numerical columns
            st.subheader("Correlation Heatmap")
            
            # Clean and encode data for correlation
            train_clean = cleaning_data(train)
            train_encoded = encode_data(train_clean)
            
            fig, ax = plt.subplots(figsize=(14, 10))
            sns.heatmap(train_encoded.corr(), annot=True, cmap='coolwarm', 
                       fmt=".2f", linewidths=0.5, ax=ax)
            plt.title("Feature Correlation Matrix")
            st.pyplot(fig)
        
        with tab4:
            st.subheader("Fraud vs Non-Fraud Distribution")
            
            if 'is_fraud' in train.columns:
                fraud_counts = train['is_fraud'].value_counts()
                
                col1, col2 = st.columns(2)
                
                with col1:
                    fig = go.Figure(data=[go.Pie(
                        labels=['Non-Fraud', 'Fraud'],
                        values=fraud_counts.values,
                        hole=0.4,
                        marker=dict(colors=['#2ecc71', '#e74c3c'])
                    )])
                    fig.update_layout(title="Transaction Distribution")
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    st.metric("Total Transactions", f"{len(train):,}")
                    st.metric("Fraudulent Transactions", f"{fraud_counts[1.0]:,}")
                    st.metric("Fraud Percentage", f"{(fraud_counts[1.0]/len(train)*100):.2f}%")
                    st.metric("Non-Fraudulent Transactions", f"{fraud_counts[0.0]:,}")

# PAGE 3: MODEL TRAINING
elif page == "Model Training":
    st.header("🤖 Model Training")
    
    if st.session_state.train_data is None:
        st.warning("⚠️ Please upload training data first!")
    else:
        train = st.session_state.train_data.copy()
        
        st.subheader("Training Configuration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            test_size = st.slider("Test Size", 0.1, 0.4, 0.2, 0.05)
            random_state = st.number_input("Random State", 0, 100, 42)
        
        with col2:
            models_to_train = st.multiselect(
                "Select Models to Train",
                ["Logistic Regression", "Random Forest", "Decision Tree"],
                default=["Logistic Regression", "Random Forest", "Decision Tree"]
            )
        
        if st.button("🚀 Train Models", type="primary"):
            with st.spinner("Training models... Please wait..."):
                # Clean and encode data
                train_clean = cleaning_data(train)
                train_encoded = encode_data(train_clean)
                
                # Prepare data
                X = train_encoded.drop(columns=["is_fraud"])
                y = train_encoded["is_fraud"]
                
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size, random_state=random_state
                )
                
                # Store results
                results = []
                trained_models = {}
                
                progress_bar = st.progress(0)
                
                for idx, model_name in enumerate(models_to_train):
                    if model_name == "Logistic Regression":
                        model = LogisticRegression(max_iter=1000)
                    elif model_name == "Random Forest":
                        model = RandomForestClassifier(n_estimators=100, random_state=random_state)
                    else:
                        model = DecisionTreeClassifier(random_state=random_state)
                    
                    # Train
                    model.fit(X_train, y_train)
                    
                    # Predict
                    y_pred = model.predict(X_test)
                    
                    # Calculate metrics
                    accuracy = accuracy_score(y_test, y_pred)
                    
                    results.append({
                        'Model': model_name,
                        'Accuracy': accuracy
                    })
                    
                    trained_models[model_name] = {
                        'model': model,
                        'accuracy': accuracy,
                        'y_pred': y_pred,
                        'y_test': y_test
                    }
                    
                    progress_bar.progress((idx + 1) / len(models_to_train))
                
                # Store in session state
                st.session_state.models_trained = True
                st.session_state.trained_models = trained_models
                st.session_state.results = results
                st.session_state.X_train = X_train
                st.session_state.X_test = X_test
                
                st.success("✅ Models trained successfully!")
                
                # Display results
                st.subheader("📊 Training Results")
                
                results_df = pd.DataFrame(results)
                st.dataframe(results_df.style.highlight_max(subset=['Accuracy'], color='lightgreen'), 
                           use_container_width=True)
                
                # Visualize accuracies
                fig = px.bar(results_df, x='Model', y='Accuracy', 
                           title='Model Accuracy Comparison',
                           color='Accuracy',
                           color_continuous_scale='viridis')
                fig.update_layout(yaxis_range=[0.9, 1.0])
                st.plotly_chart(fig, use_container_width=True)
                
                # Show confusion matrices
                st.subheader("🔲 Confusion Matrices")
                
                cols = st.columns(len(models_to_train))
                
                for idx, (model_name, model_data) in enumerate(trained_models.items()):
                    with cols[idx]:
                        cm = confusion_matrix(model_data['y_test'], model_data['y_pred'])
                        
                        fig, ax = plt.subplots(figsize=(6, 5))
                        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
                        plt.title(f'{model_name}')
                        plt.ylabel('Actual')
                        plt.xlabel('Predicted')
                        st.pyplot(fig)

# PAGE 4: PREDICTIONS
elif page == "Predictions":
    st.header("🔮 Make Predictions")
    
    if not st.session_state.models_trained:
        st.warning("⚠️ Please train models first!")
    elif st.session_state.test_data is None:
        st.warning("⚠️ Please upload test data first!")
    else:
        test = st.session_state.test_data.copy()
        
        # Select model
        model_name = st.selectbox(
            "Select Model for Prediction",
            list(st.session_state.trained_models.keys())
        )
        
        if st.button("🎯 Generate Predictions", type="primary"):
            with st.spinner("Making predictions..."):
                # Clean and encode test data
                test_clean = cleaning_data(test)
                test_encoded = encode_data(test_clean)
                
                # Prepare test data
                X_test = test_encoded.drop(columns=["is_fraud"])
                y_test = test_encoded["is_fraud"]
                
                # Get model
                model = st.session_state.trained_models[model_name]['model']
                
                # Predict
                y_pred = model.predict(X_test)
                
                # Calculate accuracy
                accuracy = accuracy_score(y_test, y_pred)
                
                st.success(f"✅ Predictions complete! Accuracy: {accuracy:.4f}")
                
                # Show metrics
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Total Predictions", len(y_pred))
                with col2:
                    st.metric("Predicted Fraud", int(y_pred.sum()))
                with col3:
                    st.metric("Actual Fraud", int(y_test.sum()))
                with col4:
                    st.metric("Accuracy", f"{accuracy:.4f}")
                
                # Confusion matrix
                st.subheader("Confusion Matrix")
                cm = confusion_matrix(y_test, y_pred)
                
                fig, ax = plt.subplots(figsize=(8, 6))
                sns.heatmap(cm, annot=True, fmt='d', cmap='RdYlGn', ax=ax)
                plt.title(f'Confusion Matrix - {model_name}')
                plt.ylabel('Actual')
                plt.xlabel('Predicted')
                st.pyplot(fig)
                
                # Classification report
                st.subheader("Classification Report")
                report = classification_report(y_test, y_pred, output_dict=True)
                report_df = pd.DataFrame(report).transpose()
                st.dataframe(report_df.style.highlight_max(axis=0), use_container_width=True)

# PAGE 5: MODEL COMPARISON
elif page == "Model Comparison":
    st.header("📊 Model Comparison")
    
    if not st.session_state.models_trained:
        st.warning("⚠️ Please train models first!")
    else:
        results_df = pd.DataFrame(st.session_state.results)
        
        st.subheader("Performance Metrics")
        
        # Display comparison table
        st.dataframe(results_df.style.highlight_max(subset=['Accuracy'], color='lightgreen'), 
                   use_container_width=True)
        
        # Bar chart
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.bar(results_df, x='Model', y='Accuracy', 
                       title='Model Accuracy Comparison',
                       color='Accuracy',
                       color_continuous_scale='viridis',
                       text='Accuracy')
            fig.update_traces(texttemplate='%{text:.4f}', textposition='outside')
            fig.update_layout(yaxis_range=[0.9, 1.0])
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Line chart
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=results_df['Model'],
                y=results_df['Accuracy'],
                mode='lines+markers',
                marker=dict(size=12, color='blue'),
                line=dict(width=3)
            ))
            fig.update_layout(
                title='Accuracy Trend',
                xaxis_title='Model',
                yaxis_title='Accuracy',
                yaxis_range=[0.9, 1.0]
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Best model
        best_model = results_df.loc[results_df['Accuracy'].idxmax()]
        st.success(f"🏆 Best Model: **{best_model['Model']}** with accuracy of **{best_model['Accuracy']:.4f}**")
        
        # Feature importance (if Random Forest is trained)
        if 'Random Forest' in st.session_state.trained_models:
            st.subheader("🌳 Random Forest Feature Importance")
            
            rf_model = st.session_state.trained_models['Random Forest']['model']
            feature_importance = pd.DataFrame({
                'Feature': st.session_state.X_train.columns,
                'Importance': rf_model.feature_importances_
            }).sort_values('Importance', ascending=False)
            
            fig = px.bar(feature_importance.head(10), 
                       x='Importance', 
                       y='Feature',
                       orientation='h',
                       title='Top 10 Most Important Features')
            st.plotly_chart(fig, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: gray;'>
        <p>💳 Credit Card Fraud Detection System | Built with Streamlit</p>
    </div>
""", unsafe_allow_html=True)
