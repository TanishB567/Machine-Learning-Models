
def main():
    # No of bedrooms
    x = [4, 5, 6, 3, 4, 3, 4, 5, 6, 6, 6, 4, 3]
    # Listed price
    y = [400, 1400, 1100, 975, 975, 975, 925, 925, 900, 850, 850, 800, 362]
    initial_w = 0
    initial_b = 0
    m = len(x)
    optimal_w, optimal_b = gradient_descent(x, y, initial_w, initial_b, m)
    no_of_bedrooms = 7
    optimal_value = (optimal_w*no_of_bedrooms) + optimal_b
    print(f"Optimal value is £{optimal_value}")

# All this does is calculate the slope at point x, y
def calculate_slope(x, y, w, b, m):
    dj_dw = 0
    dj_db = 0
    for i in range(0, m):
        f = (w*x[i]) + b
        dj_dw += (f - y[i]) * x[i]
        dj_db += f - y[i]
    dj_dw /= m
    dj_db /= m
    return dj_dw, dj_db

def gradient_descent(x, y, w, b, m):
    num_iters = 20000
    alpha = 0.05
    for i in range(0, num_iters):
        dj_dw, dj_db = calculate_slope(x, y, w, b, m)
        w -= alpha * dj_dw
        b -= alpha * dj_db
        print(w,b)
    return w,b

main()