import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

def plot_roc_curve(fpr, tpr, auc_score, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, label=f'AUC = {auc_score:.4f}')
    ax.plot([0, 1], [0, 1], 'r--', label='Random')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curve')
    ax.legend()
    return ax

def calculate_chart_data(y_true, y_pred):
    """Calculate data for Gain, Lift and K-S charts"""
    try:
        sorted_indices = np.argsort(y_pred)[::-1]
        sorted_targets = y_true[sorted_indices]
        
        n_records = len(y_true)
        n_pos = np.sum(y_true == 1)
        
        if n_pos == 0:
            return None
            
        percentiles = np.linspace(0, 100, n_records)
        cum_gains = np.cumsum(sorted_targets == 1) / n_pos * 100
        
        baseline_rate = n_pos / n_records
        lift_values = []
        for i in range(n_records):
            pos_rate = np.sum(sorted_targets[:i+1] == 1) / (i + 1)
            lift = pos_rate / baseline_rate if baseline_rate > 0 else 1
            lift_values.append(lift)
            
        return percentiles, cum_gains, np.array(lift_values)
    except Exception:
        return None, None, None

def plot_model_charts(y_true, y_pred):
    """Plot all model performance charts"""
    percentiles, gains, lifts = calculate_chart_data(y_true, y_pred)
    if all(v is not None for v in [percentiles, gains, lifts]):
        figs = []
        
        # Gain Chart
        fig1, ax1 = plt.subplots(figsize=(8, 6))
        ax1.plot(percentiles, gains, label='Model')
        ax1.plot([0, 100], [0, 100], 'r--', label='Random')
        ax1.set_xlabel('Population %')
        ax1.set_ylabel('Gain %')
        ax1.set_title('Cumulative Gains Chart')
        ax1.grid(True)
        ax1.legend()
        figs.append(fig1)
        
        # Lift Chart
        fig2, ax2 = plt.subplots(figsize=(8, 6))
        ax2.plot(percentiles, lifts)
        ax2.axhline(y=1, color='r', linestyle='--', label='Baseline')
        ax2.set_xlabel('Population %')
        ax2.set_ylabel('Lift')
        ax2.set_title('Lift Chart')
        ax2.grid(True)
        ax2.legend()
        figs.append(fig2)
        
        return figs
    return None

def create_performance_charts(y_true, y_pred, y_scores=None):
    """Generate all performance visualization charts"""
    try:
        # Convert inputs to numpy arrays
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        
        # Sort predictions for cumulative charts
        scores = y_scores if y_scores is not None else y_pred
        sorted_indices = np.argsort(scores)[::-1]
        sorted_true = y_true[sorted_indices]
        
        # Basic calculations
        n_samples = len(y_true)
        n_pos = np.sum(y_true)
        n_neg = n_samples - n_pos
        
        if n_pos == 0 or n_pos == n_samples:
            return None
            
        # Calculate percentiles and gains
        percentiles = np.linspace(0, 100, n_samples)
        cum_pos = np.cumsum(sorted_true == 1) / n_pos
        cum_neg = np.cumsum(sorted_true == 0) / n_neg
        
        # Gain chart data
        gains = cum_pos * 100
        
        # Lift chart data
        baseline = n_pos / n_samples
        lifts = np.zeros(n_samples)
        for i in range(n_samples):
            pos_count = np.sum(sorted_true[:i+1] == 1)
            lifts[i] = (pos_count / (i + 1)) / baseline if baseline > 0 else 1
        
        # KS statistic
        ks_stat = np.max(np.abs(cum_pos - cum_neg))
        
        # Create figures dictionary
        figures = {}
        
        # ROC Curve (if scores available)
        if y_scores is not None:
            fpr, tpr, _ = roc_curve(y_true, y_scores)
            roc_auc = auc(fpr, tpr)
            fig_roc = plot_roc_curve(fpr, tpr, roc_auc)
            figures['ROC Curve'] = fig_roc
        
        # Gain Chart
        fig_gain, ax_gain = plt.subplots(figsize=(8, 6))
        ax_gain.plot(percentiles, gains, 'b-', label='Model')
        ax_gain.plot([0, 100], [0, 100], 'r--', label='Random')
        ax_gain.set_xlabel('Population %')
        ax_gain.set_ylabel('Cumulative Gain %')
        ax_gain.set_title('Gain Chart')
        ax_gain.grid(True)
        ax_gain.legend()
        figures['Gain Chart'] = fig_gain
        
        # Lift Chart
        fig_lift, ax_lift = plt.subplots(figsize=(8, 6))
        ax_lift.plot(percentiles, lifts, 'g-')
        ax_lift.axhline(y=1, color='r', linestyle='--', label='Baseline')
        ax_lift.set_xlabel('Population %')
        ax_lift.set_ylabel('Lift')
        ax_lift.set_title('Lift Chart')
        ax_lift.grid(True)
        ax_lift.legend()
        figures['Lift Chart'] = fig_lift
        
        # K-S Chart
        fig_ks, ax_ks = plt.subplots(figsize=(8, 6))
        ax_ks.plot(percentiles, cum_pos * 100, 'b-', label='Positive Class')
        ax_ks.plot(percentiles, cum_neg * 100, 'r-', label='Negative Class')
        ax_ks.set_xlabel('Population %')
        ax_ks.set_ylabel('Cumulative %')
        ax_ks.set_title(f'K-S Chart (statistic: {ks_stat:.4f})')
        ax_ks.grid(True)
        ax_ks.legend()
        figures['K-S Chart'] = fig_ks
        
        return figures
        
    except Exception as e:
        print(f"Error creating charts: {str(e)}")
        return None
