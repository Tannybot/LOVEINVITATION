import numpy as np

def prepare_binary_data(y_true, y_pred):
    """Convert and validate data for binary classification metrics"""
    try:
        # Convert to numpy arrays
        y_true = np.asarray(y_true).flatten()
        y_pred = np.asarray(y_pred).flatten()
        
        # Check lengths
        if len(y_true) != len(y_pred):
            raise ValueError("Length mismatch between inputs")
            
        # Check for empty arrays
        if len(y_true) == 0:
            raise ValueError("Empty inputs")
            
        # Ensure binary values for true labels
        if not np.all(np.isin(y_true, [0, 1])):
            y_true = (y_true >= np.median(y_true)).astype(int)
            
        return y_true, y_pred
        
    except Exception as e:
        print(f"Data preparation error: {str(e)}")
        return None, None
