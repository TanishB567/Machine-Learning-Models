import numpy as np

def main(): 
    x = np.array([[2104, 5, 1, 45], [1416, 3, 2, 40], [852, 2, 1, 35]])
    y = np.array(
        [460, 232, 178]
    )
    initial_b = 0
    initial_w = np.zeros(4)
    m = len(x)
    optimal_w, optimal_b = gradient_descent(x, y, initial_w, initial_b, m)
    # no_of_bedrooms = 7
    # optimal_value = (optimal_w*no_of_bedrooms) + optimal_b
    for i in range(m):
        print(f"prediction: {np.dot(x[i], optimal_w) + optimal_b}, target value: {y[i]}")

def calculate_slope(x, y, w, b):
    m, n = x.shape
    dj_dw = np.zeros((n,))
    dj_db = 0
    for i in range(m):
        f = np.dot(x[i], w) + b
        # This in essence is the most import step
        # We're populating our dj_dw vector with
        # gradients at every w1 ... wn
        for j in range(n):
            # The += because there are multiple training
            # examples
            dj_dw[j] += (f - y[i]) * x[i, j]
        dj_db += f - y[i]
    dj_dw /= m
    dj_db /= m
    return dj_dw, dj_db

def gradient_descent(x, y, w, b, m):
    num_iterations = 10000
    alpha = 5e-7
    for i in range(num_iterations):
        dj_dw, dj_db = calculate_slope(x, y, w, b)
        w -= alpha * dj_dw
        b -= alpha * dj_db
    return w, b

main()
