#	Starter code for linear regression problem
#	Below are all the modules that you'll need to have working to complete this problem
#	Some helpful functions: np.polyfit, scipy.polyval, zip, np.random.shuffle, np.argmin, np.sum, plt.boxplot, plt.subplot, plt.figure, plt.title
import sys
import csv
import numpy as np
import scipy
import matplotlib.pyplot as plt
import math


def nfoldpolyfit(X, Y, maxK, n, verbose):
#	NFOLDPOLYFIT Fit polynomial of the best degree to data.
#   NFOLDPOLYFIT(X,Y,maxDegree, nFold, verbose) finds and returns the coefficients 
#   of a polynomial P(X) of a degree between 1 and N that fits the data Y 
#   best in a least-squares sense, averaged over nFold trials of cross validation.
#
#   P is a vector (in numpy) of length N+1 containing the polynomial coefficients in
#   descending powers, P(1)*X^N + P(2)*X^(N-1) +...+ P(N)*X + P(N+1). use
#   numpy.polyval(P,Z) for some vector of input Z to see the output.
#
#   X and Y are vectors of datapoints specifying  input (X) and output (Y)
#   of the function to be learned. Class support for inputs X,Y: 
#   float, double, single
#
#   maxDegree is the highest degree polynomial to be tried. For example, if
#   maxDegree = 3, then polynomials of degree 0, 1, 2, 3 would be tried.
#
#   nFold sets the number of folds in nfold cross validation when finding
#   the best polynomial. Data is split into n parts and the polynomial is run n
#   times for each degree: testing on 1/n data points and training on the
#   rest.
#
#   verbose, if set to 1 shows mean squared error as a function of the 
#   degrees of the polynomial on one plot, and displays the fit of the best
#   polynomial to the data in a second plot.
#   
#
#   AUTHOR: Sonia Nigam (This is where you put your name)
#
    x_sets = np.split(X, n)
    y_sets = np.split(Y, n)
    min_MSE = float("inf")
    max_MSE = 0
    min_k = float("inf")
    best_polynomial = []
    
    y_means = []
    x_kvalues = []
    
    for k in xrange(maxK+1):
        k_error = 0
        
        for trial in xrange(n):
            x_testing_data = x_sets[trial]
            y_testing_data = y_sets[trial]
        
            x_training_lists = x_sets[:trial] + x_sets[trial+1:]
            y_training_lists = y_sets[:trial] + y_sets[trial+1:]
            
            x_training_data = np.array(x_training_lists).flatten()
            y_training_data = np.array(y_training_lists).flatten()
            
            polynomial = np.polyfit(x_training_data, y_training_data, k)
            k_error += validation_error(x_testing_data, y_testing_data, polynomial)
                
        if k_error < min_MSE:
            min_MSE = k_error
            min_k = k
            best_polynomial = polynomial
        elif k_error > max_MSE:
            max_MSE = k_error
        
        y_means.append(k_error)
        x_kvalues.append(k)
    
    print "The k that yielded the best results is " + str(min_k) + " with the error of " + str(min_MSE)
        
    #graph k value with average means
    if verbose == 1:
        plt.plot(x_kvalues, y_means, "ro")
        plt.ylabel('mean square error')
        plt.xlabel('k value')
        plt.axis([0, k+.2, 0, max_MSE+.5])
        plt.title("Linear Regression MSE")
        print "saving plot image"
        plt.savefig('MSE.png')
        
        graph_best_function(best_polynomial, X)
                    

def validation_error(x_values, y_values, polynomial):
    y_actuals = []
    MSE_total = 0.0
    
    for x in x_values:
        actual = np.polyval(polynomial, x)
        y_actuals.append(actual)
    
    for i in xrange(len(y_values)):
        sum = math.pow((y_values[i]-y_actuals[i]),2)
        MSE_total += sum
        
    error = MSE_total/len(y_values) 
    return error
    
def graph_best_function(polynomial, x_data):
    y_values = []
    upper_bound_y = 0
    lower_bound_y = float("inf")
    upper_bound_x = 0
    lower_bound_x = float("inf")

    for x in x_data:
        val = np.polyval(polynomial, x)
        y_values.append(val)
        if val > upper_bound_y:
            upper_bound_y = val
        elif val < lower_bound_y:
            lower_bound_y = val
            
        if x > upper_bound_x:
            upper_bound_x = x
        elif x < lower_bound_x:
            lower_bound_x = x
    
    plt.plot(x_data, y_values, "ro")
    plt.ylabel('estimated values')
    plt.xlabel('attribute vector')
    plt.axis([lower_bound_x-.5, upper_bound_x+.5, lower_bound_x-.5, upper_bound_y+.5])
    plt.title("Best Polynomial Fit")
    print "saving plot image"
    plt.savefig('bestfit.png')
    
    

def main():
	# read in system arguments, first the csv file, max degree fit, number of folds, verbose
	rfile = sys.argv[1]
	maxK = int(sys.argv[2])
	nFolds = int(sys.argv[3])        
	verbose = bool(int(sys.argv[4]))
	
	csvfile = open(rfile, 'rb')
	dat = csv.reader(csvfile, delimiter=',')
	X = []
	Y = []
	# put the x coordinates in the list X, the y coordinates in the list Y
	for i, row in enumerate(dat):
		if i > 0:
			X.append(float(row[0]))
			Y.append(float(row[1]))
	X = np.array(X)
	Y = np.array(Y)
	nfoldpolyfit(X, Y, maxK, nFolds, verbose)

if __name__ == "__main__":
	main()
