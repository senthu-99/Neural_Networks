
# ASSESSED EXERCISE 1:
# See if you can set up the XOR problem  and see what happens when you try and learn the weights.
# Input: (-1,-1); Output: 0
# Input: (1,-1); Output: 1
# Input: (-1,1); Output: 1
# Input: (1,1); Output: 0
# Does it learn to correctly classify all the inputs?

# Load neuralnet package in
library(neuralnet)

# Set up training set using XOR gate input and output data
trainin = rbind(c(1,1), c(1,-1), c(-1,1), c(-1,-1));
trainout = rbind(0, 1, 1, 0);

# Combine XOR gate data
XOR_data = cbind(trainout, trainin)

# Fit neural network with no hidden layers
# Use neuralnet function to train the single binary perceptron
# Set seed function allows the same random values to be produced every time you run the code - used to create reproducible results
# Threshold set to 0.001 - meaning if change in error during an iteration is less than 0.1% ,then the model will stop further optimization
set.seed(2)
NN = neuralnet(XOR_data[,1]~., XOR_data[,-1], hidden = 0, threshold = 0.001,stepmax = 1e+05, linear.output = FALSE)

# Visualize the NN
plot(NN)

# Check Weights
NN$weights

# Set up test input sequence
testin = rbind(c(1,1), c(1,-1), c(-1,1), c(-1,-1));

# Use compute function to see if the network responds to input sequence
predict_testNN = compute(NN, testin)

# Gets prediction for each test input
predict_testNN$net.result

# To calculate the discrete class we threshold it at 0.5
# So you can see if it correctly predicts a 1 as output
# Values greater than 0.5 are mapped to one class and all other values are mapped to another
predict_out = as.numeric(predict_testNN$net.result>0.5)

# Should be (0,1,1,0)
predict_out

# Does it learn to classify all the inputs?
# No, it doesn't classify the inputs correctly because there are no hidden layers
# And XOR data is non-linearly classifiable data so requires MNN which uses sigmoid function.


#-------------------------------------------Multilayer Neural Networks--------------------------------------------

# Set up a Neural Network with 2 hidden layers, each with 3 neurons 
# Use neuralnet function to train neural network on the XOR data
set.seed(2)
NN = neuralnet(XOR_data[,1]~., XOR_data[,-1], hidden = c(3,3) , threshold = 0.001, stepmax = 1e+05, linear.output = FALSE)

# Visualize Neural Network
plot(NN)

# Use compute function to see if the network responds to input sequence
# Uses the same input sequence as above
predict_testNN = compute(NN, testin)

# Activation of the output neuron
# Gets prediction for each test input
predict_testNN$net.result

# To calculate the discrete class we threshold it at 0.5
# So you can see if it correctly predicts a 1 as output
predict_out = as.numeric(predict_testNN$net.result>0.5)

# Should be (0,1,1,0)
predict_out


# ASSESSED EXERCISE 2:
# Build a Neural network classifier of the wine data.
# 1. Read in winedata2.csV
# 2. Build the architecture of your neural network. The output must be between one and zero.
# 3. Using any two variables from the wine data, set up the data as you did for the linear classifier with a train and test set
# 4. Train the neural network on half of the data and test it on the remaining.
# 5. Calculate the accuracy


# Read in winedata2 dataset
winedata = read.csv('C:/Users/senth/Documents/RDatasets/winedata2.csv', sep=",")

# Gets winedata class values
wine_class = winedata[,1]

# Gets two variables from the winedata set - column 2 and 3 ( Alcohol and Malic Acid)
wine_values = winedata[,2:3]

# Replace values in WineClass to binary since they are 1 and 2
# Since MNN uses activation functions between 0 and 1( sigmoid function)
wine_class = winedata$WineClass
wine_class[which(wine_class == 1)] <- 0
wine_class[which(wine_class == 2)] <- 1

# Split the data so half is used to train and test on remaining
# Set up a training set
wine_class_train = wine_class[1:65]
wine_values_train = wine_values[1:65,]

# Set up test set
wine_class_test = wine_class[66:130]
wine_values_test = wine_values[66:130,]

# Set up a Neural Network with 2 hidden layers, 1st hidden layer with 3 neurons and 2nd with 2 neurons 
# This gave the highest accuracy and and lowest error
# Use neuralnet function to train neural network on the winedata
set.seed(2)
NN <- neuralnet(wine_class_train~., wine_values_train, hidden=c(3,2), threshold=0.001, stepmax=1e+05, linear.output=FALSE)

# Visualize Neural Network
plot(NN)

# Use compute function to see if the network responds to wine test data
predict_testNN = compute(NN, wine_values_test)

# Activation of the output neuron
# Gets prediction for each test input
predict_testNN$net.result


# To calculate the discrete class we threshold it at 0.5:
# Used to see if it correctly predicts class as 0 or 1
predict_out = as.numeric(predict_testNN$net.result > 0.50)
print(predict_out)


# Calculate accuracy of NN
# Compare to actual test values to get accuracy
n = length(wine_class_test) # the number of test cases
ncorrect = sum(predict_out == wine_class_test) # the number of correctly predicted
accuracy=ncorrect/n
print(accuracy)

