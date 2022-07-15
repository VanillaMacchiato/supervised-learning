import numpy as np

def euclidean_distance(p1: np.ndarray, p2: np.ndarray):
    return np.sqrt(np.sum((p1 - p2)**2, axis=1))

class KNN:
    def __init__(self, k=5):
        self.k_ = k if (k > 0 and type(k) == int) else 5
        self.X_ = None
        self.y_ = None
        
    def fit(self, X: np.ndarray, y: np.ndarray):
        """
            params:
                X: matriks berdimensi (m, n) dengan m adalah jumlah fitur dan n adalah jumlah baris/row
                y: matriks berdimensi (n,)
        """
        X = X.T
        if X.shape[1] != y.shape[0]:
            raise Exception("The number of row between the feature and target is not equal")
        
        if X.shape[1] < self.k_:
            raise Exception("The number of rows less than k")
            
        self.X_ = X.T
        self.y_ = y.ravel()
        
        return self
        
    def predict(self, X: np.ndarray):
        """
            params:
                X: matriks berdimensi (m, n) dengan m adalah jumlah fitur dan n adalah jumlah baris/row
        """
        X_test = X
        
        if self.X_ is None or self.y_ is None:
            raise Exception("The model has not been fit yet")
        
        m, _ = X_test.shape
        y_hat = np.zeros(m)
        
        for i in range(m):
            distances = euclidean_distance(self.X_, X_test[i])
            index_sorted = distances.argsort() # Sort jaraknya, tapi hanya mengembalikan index-nya
            k_index = index_sorted[:self.k_]
            vals, counts = np.unique(self.y_[k_index], return_index=True)
            max_index = counts.argmax()
            label = vals[max_index]
            y_hat[i] = label
            
        return y_hat