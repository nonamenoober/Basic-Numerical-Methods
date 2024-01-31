import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Read data from CSV file
def read_data_from_csv(filename):
    df = pd.read_csv(filename, header=None, skiprows=1)
    X = df[0].values
    Y = df[1].values
    X = np.asarray(X)
    Y = np.asarray(Y)
    return X, Y

# Create matrix M for polynomial function
def create_matrix(X):
    M = np.ones((len(X), 2))
    M[:, 0] = X
    M[:, 1] = np.sin(X)
    return M

# Calculate polynomial coefficients
def calculate_coefficients(M, Y):
    M_transpose = np.transpose(M)
    M_product = np.dot(M_transpose, M)
    M_inverse = np.linalg.inv(M_product)
    coefficients = np.dot(np.dot(M_inverse, M_transpose), Y)
    return coefficients

# Evaluate regression function
def evaluate_regression(coefficients, X):
    return coefficients[0] * X + coefficients[1] * np.sin(X)

# Main program
def main():
    # Read data from CSV file
    filename = "0.csv"
    X, Y = read_data_from_csv(filename)

    # Create matrix M for polynomial function
    M = create_matrix(X)

    # Calculate coefficients
    coefficients = calculate_coefficients(M, Y)

    print("Regression coefficients:")
    print(coefficients)

    # Scatter plot
    plt.scatter(X, Y, label='Data')

    # Evaluate regression function
    y_pred = evaluate_regression(coefficients, X)

    # Plot regression line
    plt.plot(X, y_pred, color='red', label='Regression Function')
    # Output graph is for function: y = a * x + b * sin(x)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()
