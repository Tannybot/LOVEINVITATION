import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from data_validation import prepare_binary_data

def validate_inputs(y_true, y_pred):
    """Validate and convert inputs to numpy arrays"""
    try:
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        if len(y_true) != len(y_pred):
            raise ValueError("Length mismatch between y_true and y_pred")
        if len(y_true) == 0:
            raise ValueError("Empty arrays")
        return y_true, y_pred
    except Exception as e:
        print(f"Input validation error: {str(e)}")
        return None, None

def plot_roc_curve(y_true, y_pred):
    """Plot ROC curve with input validation"""
    try:
        y_true, y_pred = prepare_binary_data(y_true, y_pred)
        if y_true is None:
            return None
            
        # Calculate ROC
        fpr, tpr, _ = roc_curve(y_true, y_pred)
        roc_auc = auc(fpr, tpr)
        
        # Create plot
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
        ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('Receiver Operating Characteristic (ROC)')
        ax.legend(loc="lower right")
        ax.grid(True)
        return fig

    except Exception as e:
        print(f"ROC curve error: {str(e)}")
        return None

def plot_gain_lift_chart(y_true, y_pred):
    """Plot Gain and Lift charts with input validation"""
    try:
        y_true, y_pred = prepare_binary_data(y_true, y_pred)
        if y_true is None:
            return None
            
        # Sort by predictions
        sorted_indices = np.argsort(y_pred)[::-1]
        sorted_true = y_true[sorted_indices]
        
        # Calculate percentiles and metrics
        n_samples = len(sorted_true)
        percentiles = np.linspace(0, 100, n_samples)
        total_pos = max(np.sum(sorted_true), 1)
        
        # Calculate cumulative gains
        cum_gains = np.cumsum(sorted_true) / total_pos * 100
        
        # Calculate lift
        base_rate = max(total_pos / n_samples, 1e-10)
        lifts = np.array([
            (np.sum(sorted_true[:i+1]) / (i+1)) / base_rate 
            for i in range(n_samples)
        ])
        
        # Create plots with matching dimensions
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Gain Chart
        ax1.plot(percentiles, cum_gains, 'b-', label='Model', linewidth=2)
        ax1.plot([0, 100], [0, 100], 'r--', label='Random')
        ax1.set_xlabel('Population %')
        ax1.set_ylabel('Gain %')
        ax1.set_title('Cumulative Gain Chart')
        ax1.grid(True)
        ax1.legend()
        
        # Lift Chart
        ax2.plot(percentiles, lifts, 'g-', linewidth=2)
        ax2.axhline(y=1, color='r', linestyle='--', label='Baseline')
        ax2.set_xlabel('Population %')
        ax2.set_ylabel('Lift')
        ax2.set_title('Lift Chart')
        ax2.grid(True)
        ax2.legend()
        
        plt.tight_layout()
        return fig
        
    except Exception as e:
        print(f"Gain/Lift chart error: {str(e)}")
        return None

def plot_ks_chart(y_true, y_pred):
    sorted_idx = np.argsort(y_pred)
    y_true = y_true[sorted_idx]
    
    # Calculate cumulative rates
    total_pos = np.sum(y_true == 1)
    total_neg = len(y_true) - total_pos
    
    cum_pos = np.cumsum(y_true == 1) / total_pos
    cum_neg = np.cumsum(y_true == 0) / total_neg
    
    # Calculate KS statistic
    ks_stat = np.max(np.abs(cum_pos - cum_neg))
    
    # Create K-S plot
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(np.linspace(0, 100, len(y_true)), cum_pos * 100, 
            'b-', label='Positive Class')
    ax.plot(np.linspace(0, 100, len(y_true)), cum_neg * 100, 
            'r-', label='Negative Class')
    ax.set_xlabel('Population %')
    ax.set_ylabel('Cumulative %')
    ax.set_title(f'Kolmogorov-Smirnov Chart (KS = {ks_stat:.3f})')
    ax.grid(True)
    ax.legend()
    return fig
