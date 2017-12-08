from numpy import *

def compute_error_for_given_points(c, m, points):
    totalError = 0
    for i in range(0, len(points)):
        x = points[i, 0]
        y = points[i, 1]
        totalError += (y - (m*x + c)) ** 2
    return totalError / float(len(points))



def step_gradient(current_c, current_m, points, learning_rate):
    # gradient descent
    c_gradient = 0
    m_gradient = 0
    N = float(len(points))
    for i in range(0, len(points)):
        x = points[i, 0]
        y = points[i, 1]
        c_gradient += -(2/N) * (y - (current_m*x + current_c))
        m_gradient += -(2/N) * x*(y - (current_m*x + current_c))
    new_c = current_c - (learning_rate * c_gradient)
    new_m = current_m - (learning_rate * m_gradient)
    return [new_c, new_m]

def gradient_descent_runner(points, starting_c, starting_m, learning_rate, num_iterations):
    c = starting_c
    m = starting_m

    for i in range(num_iterations):
        [c, m] = step_gradient(c, m, array(points), learning_rate)
    return [c, m]

def run():
    points = genfromtxt('data.csv', delimiter=',')
    #hyperparameter
    learning_rate = 0.0001
    #y = mx + c
    initial_m = 0
    initial_c = 0
    num_iterations = 1000
    [c, m] = gradient_descent_runner(points, initial_c, initial_m, learning_rate, num_iterations)
    print(c)
    print(m)

if __name__ == '__main__':
    run()
