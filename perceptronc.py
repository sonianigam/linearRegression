import sys
import csv
import numpy as np
import scipy
import matplotlib.pyplot as plt
import math


def perceptronc(w_init, X, Y):
	#figure out (w, k) and return them here. w is the vector of weights, k is how many iterations it took to converge.
    #initialize weighted vector
    w = w_init
    #initiliaze number of times it takes to converge to answer
    e = 0
    #incremement to keep track of correct vs errors
    correct = 0
    
    #execute until a paramter vector is found, which will return out of the loop
    while True: 
        #reset correct counter
        correct = 0       
        for component_one, component_two in zip(X,Y):
            #find value based on weighted vector and initialize classification variable
            xk = w[0]+w[1]*component_one + w[2]*math.pow(component_one,2)
            classification = int()
            
            #classify data based on value found with weighted param vector
            if xk <= 0:
                classification = -1
            else:
                classification = 1

            #correct classification, incremenent correct counter
            if classification == component_two:
                correct += 1
                
            #if classification is incorrect update param vector
            else:
                w = (component_two * np.array([1,component_one, math.pow(component_one,2)])) + w
                
        #increment number of completed iterations        
        e += 1 
        #check if all datapoints correctly classified, if so return answer
        if correct == len(X):
            return (w, e)
            


def main():
	rfile = sys.argv[1]
	
	#read in csv file into np.arrays X1, X2, Y1, Y2
	csvfile = open(rfile, 'rb')
	dat = csv.reader(csvfile, delimiter=',')
	X1 = []
	Y1 = []
	X2 = []
	Y2 = []
	for i, row in enumerate(dat):
		if i > 0:
			X1.append(float(row[0]))
			X2.append(float(row[1]))
			Y1.append(float(row[2]))
			Y2.append(float(row[3]))
	X1 = np.array(X1)
	X2 = np.array(X2)
	Y1 = np.array(Y1)
	Y2 = np.array(Y2)
    #initilaize weight vector to be [0,0]
	w_init = np.array([0,0,0])
    
	#grab parameter vector and number of epochs to correctly classify X2: this is commented out because it cannot terminate given then present algorithm. Please see perceptronc.py to find the classification solution to X2 data set.
    
	w_2, e_2 = perceptronc(w_init, X2, Y2)
	print "It took " + str(e_2) + " epochs to correctly classify X2."
	print "The parameter vector is: " + str(w_2)
    
    
    ######### Graph of X2, Y2 Data ################
        
    # plt.plot(X2, X2_new, "ro")
    # plt.ylabel('Y2 Data')
    # plt.xlabel('X2 Data')
    # plt.title('Detecting Data Trends')
    # plt.savefig('X2transform.png')

    

if __name__ == "__main__":
	main()
