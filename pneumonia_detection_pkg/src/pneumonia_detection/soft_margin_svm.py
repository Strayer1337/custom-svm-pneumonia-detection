import numpy as np


class SoftMarginSVM:
    def __init__(self, lr=1e-5, lambda_param=0.01, n_iters=500):
        self.lr = lr
        self.lambda_param = lambda_param  # Hệ số Regularization
        self.n_iters = n_iters
        self.w = None
        self.b = None

    def fit(self, X, y):
        n_samples, n_features = X.shape

        # Chuyển nhãn từ {0, 1} sang {-1, 1} để phù hợp với toán học của SVM
        y_transformed = np.where(y <= 0, -1, 1)

        self.w = np.zeros(n_features)
        self.b = 0

        for _ in range(self.n_iters):
            # Stochastic Gradient Descent: Xáo trộn dữ liệu
            indices = np.random.permutation(n_samples)

            for idx in indices:
                x_i = X[idx]
                y_i = y_transformed[idx]

                # Kiểm tra điều kiện lề (margin)
                condition = y_i * (np.dot(x_i, self.w) + self.b)

                if condition >= 1:
                    # Trường hợp nằm ngoài lề hoặc đúng phía (chỉ phạt Regularization)
                    self.w -= self.lr * (2 * self.lambda_param * self.w)
                else:
                    # Trường hợp vi phạm lề (phạt cả Regularization và Hinge Loss)
                    self.w -= self.lr * (2 * self.lambda_param * self.w - np.dot(x_i, y_i))
                    self.b -= self.lr * (-y_i)

    def predict(self, X):
        linear_output = np.dot(X, self.w) + self.b
        # Chuyển từ nhãn {-1, 1} về lại {0, 1}
        return np.where(np.sign(linear_output) <= 0, 0, 1)
