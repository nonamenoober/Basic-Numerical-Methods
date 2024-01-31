import pandas as pd
import numpy as np
import csv
import matplotlib.pyplot as plt
import math

# Read data from CSV file

def read_data_from_csv(filename):
    df = pd.read_csv(filename, header=None, skiprows=1)
    X = df[0].values
    Y = df[1].values
    X = np.asarray(X)
    Y = np.asarray(Y)
    return X, Y

# Usage example
filename = "lsqdata.csv"
X, Y = read_data_from_csv(filename)

# Create matrix M for polynomial function
def create_matrix(X, n):
    M = np.ones((len(X), n + 1))
    for i in range(1, n + 1):
        M[:, i] = X ** i
    return M

# Calculate polynomial coefficients
def calculate_polynomial_coefficients(M, Y):
    M_transpose = np.transpose(M)
    M_product = np.dot(M_transpose, M)
    M_inverse = np.linalg.inv(M_product)
    coefficients = np.dot(np.dot(M_inverse, M_transpose), Y)
    return coefficients

# Evaluate polynomial function
def evaluate_polynomial(coefficients, x):
    y = np.sum(coefficients * np.power(x[:, np.newaxis], np.arange(len(coefficients))), axis=1)
    return y

def calculate_rmse(y_true, y_pred):
    mse = math.sqrt(np.mean((y_true - y_pred) ** 2))
    return mse

def calculate_R_squared(y_true, y_pred):
    SSR = np.sum((y_true - y_pred) ** 2)
    SST = np.sum((y_true - np.mean(y_true)) ** 2)
    R_squared = 1 - (SSR / SST)
    return R_squared

# Modify Y array for exponential function
def modify_exponential_Y(Y):
    modified_Y = np.copy(Y)
    if np.any(Y < 0):
        k = abs(np.min(Y)) + 0.0000000001
        modified_Y += k
    modified_Y = np.log(modified_Y)
    return modified_Y

# Create matrix M for exponential function
def create_exponential_matrix(X):
    M = np.ones((len(X), 2))
    M[:, 1] = X
    return M

# Calculate exponential coefficients
def calculate_exponential_coefficients(M, modified_Y):
    M_transpose = np.transpose(M)
    M_product = np.dot(M_transpose, M)
    M_inverse = np.linalg.inv(M_product)
    coefficients = np.dot(np.dot(M_inverse, M_transpose), modified_Y)
    return coefficients

# Main program
def main():
    '''filename = "data_bptt.csv"  # Replace with your CSV file path
    
    X, Y = read_data_from_csv(filename)

    # Convert Y to pandas Series
    Y_series = pd.Series(Y)

    p_range = range(1, 20)

    # Initialize an empty list to store the MSE values
    rmse_list = []
    Rsquared_list = []

    # Iterate over each polynomial degree
    for p in p_range:
        # Create matrix M for polynomial function
        M = create_matrix(X, p)

        # Calculate polynomial coefficients
        coefficients = calculate_polynomial_coefficients(M, Y)

        # Evaluate polynomial function
        y_pred = evaluate_polynomial(coefficients, X)

        # Calculate MSE
        rmse = calculate_rmse(Y, y_pred)

        R_squared = calculate_R_squared (Y, y_pred)

        # Append MSE for the current degree to the list
        rmse_list.append(rmse)
        Rsquared_list.append(R_squared)

    # Plot MSE vs. polynomial degree
    plt.plot(p_range, rmse_list, 'b', marker='o')
    plt.xlabel('Polynomial Degree')
    plt.ylabel('MSE')
    plt.title('MSE vs. Polynomial Degree')
    plt.show()

    plt.plot(p_range, Rsquared_list, 'b', marker='o')
    plt.xlabel('Polynomial Degree')
    plt.ylabel('R^2')
    plt.title('R^2 vs. Polynomial Degree')
    plt.show()'''

    # Calculate rolling mean and standard deviation
    while True:
        print("Menu:")
        print("1. Polynomial Function")
        print("2. Exponential Function")
        print("0. Exit")
        
        choice = input("Enter your choice: ")
        
        if choice == "1":
            n = int(input("Enter the degree of the polynomial function: "))

            # Read data from CSV file
            X, Y = read_data_from_csv(filename)

            # Create matrix M for polynomial function
            M = create_matrix(X, n)

            # Calculate polynomial coefficients
            coefficients = calculate_polynomial_coefficients(M, Y)

            print("Polynomial coefficients:")
            print(coefficients)

            # Scatter plot
            plt.scatter(X, Y, label='Data')

            y_pred = evaluate_polynomial(coefficients, X)
            
            RMSE = calculate_rmse(Y, y_pred)
            print("Root mean squared error:", RMSE)

            R_squared = calculate_R_squared(Y, y_pred)
            print("R^2 Error: ", R_squared)

            # Generate line points
            x_line = np.linspace(np.min(X), np.max(X), 100)
            y_line = evaluate_polynomial(coefficients, x_line)

            # Plot polynomial line
            plt.plot(x_line, y_line, color='red', label='Polynomial Function')

            plt.xlabel('X')
            plt.ylabel('Y')
            plt.legend()
            plt.show()
        
        elif choice == "2":
            # Read data from CSV file
            X, Y = read_data_from_csv(filename)

            # Modify Y array for exponential function
            modified_Y = modify_exponential_Y(Y)

            # Create matrix M for exponential function
            M = create_exponential_matrix(X)

            # Calculate exponential coefficients
            coefficients = calculate_exponential_coefficients(M, modified_Y)

            # Replace first coefficient with e^a
            coefficients[0] = math.exp(coefficients[0])

            print("Exponential coefficients:")
            print(coefficients)

            y_pred = coefficients[0] * np.exp(coefficients[1] * X) - np.max(modified_Y)

            RMSE = calculate_rmse(Y, y_pred)
            print("Root mean squared error:", RMSE)

            R_squared = calculate_R_squared(Y, y_pred)
            print("R^2 Error: ", R_squared)

             # Scatter plot
            plt.scatter(X, Y, label='Data')
            plt.show()
            # Scatter plot
            plt.scatter(X, Y, label='Data')

            # Generate line points
            x_line = np.linspace(np.min(X), np.max(X), 100)
            y_line = coefficients[0] * np.exp(coefficients[1] * x_line) - np.max(modified_Y)

            # Plot exponential line
            plt.plot(x_line, y_line, color='red', label='Exponential Function')

            plt.xlabel('X')
            plt.ylabel('Y')
            plt.legend()
            plt.show()
        
        elif choice == "0":
            break
        
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()