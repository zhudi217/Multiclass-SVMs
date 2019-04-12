import cvxopt
import cvxopt.solvers
import numpy as np
from Kernels import polynomial_kernel, linear_kernel, gaussian_kernel

class SVM(object):
    def __init__(self, kernel=polynomial_kernel, C=None, threshold=1e-5):
        self.kernel = kernel
        self.C = C
        self.threshold = threshold
        if self.C is not None: self.C = float(self.C)

    def fit(self, X, Y):
        n_samples, n_features = X.shape

        #Construct kernel matrix
        print('[SVM] Constructing Kernel Matrix...')
        self.K = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            for j in range(n_samples):
                self.K[i][j] = self.kernel(X[i], X[j])

        # Set up parameters for convex optimization libray CVXOPT
        print("[SVM] Preparing Data for CVXOPT")
        P = np.outer(Y, Y) * self.K
        q = np.ones((n_samples)) * -1
        if self.C is None:
            G = -np.eye(n_samples)
            h = np.zeros((n_samples))
        else:
            G = np.concatenate((np.eye(n_samples), -np.eye(n_samples)))
            h = np.concatenate((self.C * np.ones((n_samples)), np.zeros((n_samples))))
        A = Y.reshape(1, n_samples).astype(np.double)
        b = 0.0

        # Call the quadratic solver
        print('[SVM] Solving QP problem...')
        sol = cvxopt.solvers.qp(cvxopt.matrix(P), cvxopt.matrix(q),
                                cvxopt.matrix(G), cvxopt.matrix(h),
                                cvxopt.matrix(A), cvxopt.matrix(b)
        )

        # Get the Lagrange multipliers out of the solution dictionary
        alphas = np.array(sol['x'])
        alphas = np.reshape(alphas, (alphas.shape[0]))

        # Find the (indices of the) support vectors, which are the vectors with non-zero Lagrange multipliers
        print('[SVM] Finding support vectors...')
        self.sv = np.where(alphas > self.threshold)[0]
        self.nsupport = len(self.sv)
        print("[SVM]", self.nsupport, "support vectors found")

        # Just retain the data corresponding to the support vectors
        self.X = X[self.sv, :]
        self.alphas = alphas[self.sv]
        self.sv_Y = Y[self.sv]

        self.b = 0.0
        for n in range(len(self.alphas)):
            self.b += self.sv_Y[n]
            for m in range(self.nsupport):
                self.b -= self.alphas[m] * self.sv_Y[m] * self .K[self.sv[n], self.sv[m]]
        self.b /= len(self.alphas)
        print("[SVM] b=",self.b)


    def evaluate(self, X_test):
        self.Y = np.zeros(np.shape(X_test)[0])
        for j in range(np.shape(X_test)[0]):
            for i in range(self.nsupport):
                self.Y[j] += self.alphas[i] * self.sv_Y[i] * self.kernel(X_test[j], self.X[i])
            self.Y[j] += self.b
        return self.Y

    def predict(self, X_test):
        return np.sign(self.evaluate(X_test))
