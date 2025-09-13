from sklearn.datasets import make_classification

from sklearn.datasets import make_classification

def load_network_data(return_labels=False):
    """
    Generates synthetic classification data for IDS training.
    
    Args:
        return_labels (bool): If True, returns X and y.
    
    Returns:
        np.ndarray: Features (and labels if requested)
    """
    X, _ = make_classification(n_samples=200, n_features=10, n_informative=6, random_state=42)
    return (X, _) if return_labels else X
