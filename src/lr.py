import numpy as np

epsilon = 1e-7

def sigmoid(z: np.ndarray):
    return 1/(1+np.exp(-z))

def cost_function(y, y_hat):
    # Menggunakan log loss
    if y.size != y_hat.size:
        raise Exception("y and y_hat have different sizes")
    cost = y.T @ np.log(y_hat + epsilon) + (1 - y.T) @ np.log(1 - y_hat + epsilon)
    cost = - cost / len(y)
    return cost

class LogisticRegression:
    """
        Implementasi algoritma Logistic Regression untuk klasifikasi dalam bahasa Python
    """
    def __init__(self, learning_rate=0.1, num_iter=100, standardize=True) -> None:
        self.num_iter_ = num_iter
        self.lr_ = learning_rate
        self.w_ = None
        self.X_ = None
        self.y_ = None
        self.standardize_ = standardize
        self.mean_ = None
        self.std_ = None
        self.cost_list_ = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
            Melakukan fit pada Logistic Regression

            Params
                X: matrix berdimensi (m, n) dengan m adalah jumlah fitur dan n adalah jumlah row
                y: matrix berdimensi (n,)
        """
        
        # Standardize
        if self.standardize_:
            self.mean_ = X.mean(axis=0)
            self.std_ = X.std(axis=0)
            self.X_ = (X - self.mean_) / self.std_
        else:
            self.X_ = X
        
        # Ditambah bias
        self.X_ = np.append(self.X_, np.ones((self.X_.shape[0], 1)), axis=1)
        
        self.w_ = np.zeros((self.X_.shape[1], 1))
        self.y_ = y.reshape((-1, 1))
        self.cost_list_ = []
        
        for i in range(self.num_iter_):
            z = self.X_@self.w_
            y_hat = sigmoid(z)
            cost = cost_function(self.y_, y_hat)[0]
            self.cost_list_.extend(cost)
            
            grad = self.X_.T@(y_hat - self.y_)
            self.w_ = self.w_ - self.lr_ * grad

        return self

    def predict_probability(self, X: np.array):
        if (self.X_ is None) and (self.y_ is None):
            raise Exception("The model has not been fit yet")
            
        if (self.X_.shape[1] - 1) != X.shape[1]:
            raise Exception("Feature sizes do not match")
            
        X_ = X
        if self.standardize_:
            X_ = (X - self.mean_) / self.std_
            
        
        X_bias = np.append(X_, np.ones((X_.shape[0], 1)), axis=1)
        
        z = X_bias@self.w_
        y_hat = sigmoid(z)
        return y_hat
    
    def predict(self, X: np.array):
        return (self.predict_probability(X) > 0.5).astype(float).ravel()
            
    def get_cost(self):
        if self.cost_list_ is None:
            raise Exception("The model has not been fit yet")
        return self.cost_list_