import streamlit as st
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

from metrics_utils import calculate_metrics
from utils import calculate_binary_metrics, calculate_roc_metrics
from visualizations import create_performance_charts
from viz_utils import plot_roc_curve, plot_gain_lift_chart, plot_ks_chart
from data_validation import prepare_binary_data

st.set_page_config(layout="wide", page_title="Decision Tree 1.0")
st.title("Decision Tree 1.0")
st.markdown("A professional grade tool for interactive decision tree analysis and prediction")

# Create main navigation
nav_data, nav_model, nav_pred, nav_docs = st.tabs([
    "📊 Data Explorer", "🔧 Model Training", "🎯 Predictions", "📘 Documentation"
])

with nav_data:
    uploaded = st.file_uploader("Upload Dataset (CSV)", type=["csv"])
    if not uploaded:
        st.info("Please upload a CSV file to begin")
        st.stop()
        
    df = pd.read_csv(uploaded)
    
    # Data Quality Check
    st.header("Data Quality Assessment")
    quality_col1, quality_col2 = st.columns([1, 3])
    
    with quality_col1:
        # Calculate quality metrics
        missing_values = df.isnull().sum().sum()
        duplicates = df.duplicated().sum()
        all_numeric = all(dt.kind in 'biufc' for dt in df.dtypes)
        
        if missing_values == 0 and duplicates == 0 and all_numeric:
            st.success("✅ Data Quality Check Passed")
        else:
            st.warning("⚠️ Data Quality Issues Detected")
        
        # Preprocessing options
        st.subheader("Data Preprocessing")
        auto_process = st.checkbox("Force Data Preprocessing", value=False)
        
    with quality_col2:
        st.write("Quality Metrics:")
        metrics_df = pd.DataFrame({
            'Metric': ['Missing Values', 'Duplicate Rows', 'Non-numeric Columns'],
            'Count': [
                missing_values,
                duplicates,
                len(df.select_dtypes(exclude='number').columns)
            ]
        })
        st.table(metrics_df)
    
    # Apply preprocessing if needed or requested
    if auto_process or missing_values > 0 or not all_numeric:
        with st.spinner("Preprocessing data..."):
            # Handle missing values
            for col in df.select_dtypes(include='number'):
                df[col] = df[col].fillna(df[col].mean())
            for col in df.select_dtypes(exclude='number'):
                df[col] = df[col].fillna(df[col].mode()[0])
            
            # Encode categorical
            for col in df.select_dtypes(exclude='number'):
                df[col] = LabelEncoder().fit_transform(df[col].astype(str))
            
            st.success("Preprocessing complete!")
    
    # Dataset Overview
    st.header("Dataset Overview")
    
    # Stats summary
    col_stats = st.columns(3)
    with col_stats[0]:
        st.metric("Total Rows", df.shape[0])
    with col_stats[1]:
        st.metric("Total Columns", df.shape[1])
    with col_stats[2]:
        st.metric("Numeric Columns", len(df.select_dtypes(include='number').columns))
    
    # Data Preview
    st.subheader("Data Preview")
    preview_options = st.columns([2, 1])
    with preview_options[0]:
        num_rows = st.slider("Number of rows to display", 5, 50, 10)
    with preview_options[1]:
        sort_by = st.selectbox("Sort by column", ["None"] + list(df.columns))
    
    # Display the data with sorting if selected
    if sort_by != "None":
        preview_df = df.sort_values(by=sort_by).head(num_rows)
    else:
        preview_df = df.head(num_rows)
    
    st.dataframe(preview_df, use_container_width=True)
    
    # Data preview with detailed statistics
    st.subheader("Detailed Statistics")
    stats_df = pd.DataFrame({
        'count': df.count(),
        'mean': df.mean(numeric_only=True),
        'std': df.std(numeric_only=True),
        'min': df.min(numeric_only=True),
        '25%': df.quantile(0.25, numeric_only=True),
        '50%': df.median(numeric_only=True),
        '75%': df.quantile(0.75, numeric_only=True),
        'max': df.max(numeric_only=True)
    }).round(4)
    
    st.dataframe(stats_df, use_container_width=True)

with nav_model:
    # Model configuration panel
    st.sidebar.header("Model Configuration")
    
    # Feature selection
    st.sidebar.subheader("Feature Selection")
    cols = df.columns.tolist()
    target = st.sidebar.selectbox("Target Variable", cols)
    features = st.sidebar.multiselect("Features", [c for c in cols if c != target])
    
    # Model parameters
    st.sidebar.subheader("Model Parameters")
    max_depth = st.sidebar.number_input("Max Tree Depth", min_value=1, max_value=20, value=5)
    criterion = st.sidebar.selectbox("Split Criterion", 
                                   ["squared_error", "friedman_mse", "absolute_error", "poisson"])
    test_size = st.sidebar.slider("Test Set Size", 10, 50, 20)
    
    if not features:
        st.warning("Please select features to train the model")
        st.stop()

    # --- 2. Train/Test split & model ---
    X_train, X_test, y_train, y_test = train_test_split(
        df[features], df[target], test_size=test_size/100, random_state=42
    )

    tree = DecisionTreeRegressor(
        max_depth=max_depth,  # Removed the conditional since min_value=1
        criterion=criterion,
        random_state=42
    )
    tree.fit(X_train, y_train)
    preds = tree.predict(X_test)

    # --- 3. Metrics calculations ---
    try:
        # Convert to binary labels first
        threshold = np.median(y_test)
        y_test_binary = (y_test >= threshold).astype(int)
        y_pred_binary = (preds >= threshold).astype(int)
        
        # Calculate metrics
        computed_metrics, cm_values, roc_values = calculate_metrics(y_test_binary, y_pred_binary, preds)
        
        if computed_metrics and cm_values:
            tn, fp, fn, tp = cm_values
            
            # Display results
            st.header("Model Performance")
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📊 Classification Metrics")
                metrics_display = pd.DataFrame([
                    {"Metric": k, "Value": v} for k, v in computed_metrics.items()
                ])
                st.table(metrics_display.set_index("Metric").style.format({"Value": "{:.4f}"}))
                
                # Display confusion matrix
                st.subheader("Confusion Matrix")
                tn, fp, fn, tp = cm_values
                cm_df = pd.DataFrame([
                    ["True Negative (TN)", tn, "False Positive (FP)", fp],
                    ["False Negative (FN)", fn, "True Positive (TP)", tp]
                ])
                cm_df.columns = ["Type", "Count", "Type_2", "Count_2"]
                st.table(cm_df)
            
            with col2:
                st.subheader("🎯 Training Summary")
                st.write(f"""
                - Features: {len(features)}
                - Train samples: {len(X_train)}
                - Test samples: {len(X_test)}
                - Tree depth: {max_depth}
                - Criterion: {criterion}
                """)

            # Visualization section
            st.header("Performance Visualizations")
            viz_tabs = st.tabs(["ROC & AUC", "Gain & Lift", "K-S Chart"])
            
            try:
                y_true, y_pred = prepare_binary_data(y_test_binary, preds)
                if y_true is None:
                    st.error("Could not prepare data for visualization")
                    st.stop()
                    
                charts = {
                    "ROC & AUC": plot_roc_curve(y_true, y_pred),
                    "Gain & Lift": plot_gain_lift_chart(y_true, y_pred),
                    "K-S Chart": plot_ks_chart(y_true, y_pred)
                }
                
                for tab, (name, fig) in zip(viz_tabs, charts.items()):
                    with tab:
                        if fig is not None:
                            st.pyplot(fig)
                        else:
                            st.warning(f"Could not generate {name}")
                            
            except Exception as e:
                st.error("Visualization error")
                st.info(f"Details: {str(e)}")

        else:
            st.warning("Could not calculate metrics")
    except Exception as e:
        st.error("Error in model evaluation")
        st.info(f"Details: {str(e)}")

with nav_pred:
    if 'tree' not in locals():
        st.info("Please train model first")
        st.stop()
    
    # Predictions interface
    st.header("Model Predictions")
    
    # Feature importance and predictions visualization
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("🌳 Feature Importances")
        fi = pd.Series(tree.feature_importances_, 
                      index=features).sort_values(ascending=True)
        fig, ax = plt.subplots(figsize=(10, 6))
        fi.plot(kind='barh')
        st.pyplot(fig)
        
        st.subheader("Top Features")
        for feat, imp in fi.iloc[-5:].items():
            st.write(f"- **{feat}**: {imp:.2%}")

    with col2:
        st.subheader("📈 Actual vs Predicted")
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(y_test, preds, alpha=0.6)
        ax.plot([y_test.min(), y_test.max()],
                [y_test.min(), y_test.max()],
                'r--', lw=2)
        ax.set_xlabel("Actual")
        ax.set_ylabel("Predicted")
        st.pyplot(fig)
    
    # Tree visualization with controls
    st.header("Decision Tree Structure")
    viz_depth = st.slider("Visualization Depth", 1, 10, 3)
    fig, ax = plt.subplots(figsize=(15, 8))
    plot_tree(tree, 
             max_depth=viz_depth,
             feature_names=features,
             filled=True,
             rounded=True,
             ax=ax)
    st.pyplot(fig)

with nav_docs:
    st.header("Documentation")
    st.markdown("""
    ### About this Tool
    This professional-grade tool provides comprehensive decision tree analysis capabilities:
    
    - **Data Exploration**: Detailed statistical analysis and data preview
    - **Model Training**: Interactive feature selection and parameter tuning
    - **Predictions**: Real-time prediction visualization and analysis
    - **Documentation**: Comprehensive guide and methodology explanation
    """)
