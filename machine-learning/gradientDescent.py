import numpy as np
def compute_cost(X, y, theta):
    m = len(y)
    h_theta = np.dot(X, theta)
    J = 1 / (2*m) * np.square(h_theta - y).sum()
    return J

def gradientDescent(X, y, theta, alpha, num_iters):
    m = len(y)
    J_history = []
    theta_tmp = [0] * len(theta)
    for i in range(num_iters):
        h_theta = np.dot(X, theta)
        for j in range(len(theta)):
            sigma = np.multiply(h_theta-y, X[:,j]).sum()
            theta_tmp[j] = theta[j] - alpha/m * sigma
        theta = theta_tmp;
        J_history.append(compute_cost(X, y, theta))

    return theta, J_history

def main():
    arr = np.loadtxt("data2.txt", delimiter=',')
    x = arr[:,0]
    # gradient descent params
    X = np.c_[ np.ones(x.shape[0]), x ]
    y = arr[:,1]
    theta = [0] * 2
    alpha = 0.01
    num_iters = 1000

    theta, J_history = gradientDescent(X, y, theta, alpha, num_iters)
    print(theta)

if __name__ == "__main__":
    main()
