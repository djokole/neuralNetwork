import numpy as np
import cv2
import os
import pickle

def prepareImages(folder_path):
    #Prepare and import the data
    number_folders_all = os.listdir(folder_path)
    number_folders = [x for x in number_folders_all if not x.startswith(".")]
    number_folders.sort()
    print(number_folders)

    image_arrays = []
    labels_arrays = []

    for folder in number_folders:
        files_all = os.listdir(folder_path + "/" + folder)
        files = [x for x in files_all if not x.startswith(".")]
        for file_name in files:
            # Read the image
            file_path = os.path.join(folder_path, folder, file_name)
            print(file_path)
            img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            
            # Resize the image to 28x28
            img = cv2.resize(img, (28, 28))
            
            # Invert the colors (black background, white digits)
            img = cv2.bitwise_not(img)
            
            # Convert the image to a numpy array
            img_array = np.array(img)
            
            # Flatten the array
            img_array = img_array.flatten()
            
            # Append the image array to the list
            image_arrays.append(img_array)

            # Append the label array to the list
            Y = np.zeros(10)
            Y[int(folder)] = 1
            labels_arrays.append(Y)


    image_arrays = np.array(image_arrays).T
    labels_arrays = np.array(labels_arrays).T

    np.random.seed(10)
    num_of_columns = labels_arrays.shape[1]
    column_indices = np.random.permutation(num_of_columns)
    image_arrays = image_arrays[:, column_indices]
    labels_arrays = labels_arrays[:, column_indices]

    return image_arrays, labels_arrays


#Make a model of the network, intialize the values
#Parametars for the network: input data, output data
def init_parametars(Xn, Yn, nodes_per_layer = [240,80]):
    W1 = np.random.randn(Xn,nodes_per_layer[0])*0.01
    b1 = np.zeros((nodes_per_layer[0],1))
    W2 = np.random.randn(nodes_per_layer[0],nodes_per_layer[1])*0.01
    b2 = np.zeros((nodes_per_layer[1],1))
    W3 = np.random.randn(nodes_per_layer[1],Yn)*0.01
    b3 = np.zeros((Yn,1))
    parametars = {"W1": W1,
                  "W2": W2,
                  "W3": W3,
                  "b1": b1,
                  "b2": b2,
                  "b3": b3}
    return parametars

#Forward propagation
def forward_propagation(X, parameters):

    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    W3 = parameters["W3"]
    b3 = parameters["b3"]

    Z1 = np.dot(W1.T,X) + b1
    A1 = np.tanh(Z1)
    Z2 = np.dot(W2.T,A1) + b2
    A2 = np.tanh(Z2)
    Z3 = np.dot(W3.T,A2) + b3
    A3 = sigmoid(Z3)
    
    assert(A3.shape == (10, X.shape[1]))
    
    cache = {"Z1": Z1,
             "A1": A1,
             "Z2": Z2,
             "A2": A2,
             "Z3": Z3,
             "A3": A3}
    
    return A3, cache

def sigmoid(x):
    return 1/(1 + np.exp(-x))

#Cost function
def compute_cost(Y, A3):
    logprobs = np.multiply(Y, np.log(A3)) + np.multiply((1-Y),np.log(1-A3))
    m = Y.shape[1]
    cost = (-1/m)*np.sum(logprobs)
    cost = float(np.squeeze(cost))
    assert(isinstance(cost,float))
    return cost
    
#Back propagation
def backward_propagation(parameters, cache, X, Y):
    m = X.shape[1]
    
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    W3 = parameters["W3"]
    b3 = parameters["b3"]

    A1 = cache["A1"]
    A2 = cache["A2"]
    A3 = cache["A3"]
    Z1 = cache["Z1"]
    Z2 = cache["Z2"]
    Z3 = cache["Z3"]

    dZ3 = A3 - Y
    dW3 = (1/m) * np.dot(dZ3,A2.T)
    db3 = (1/m) *(np.sum(dZ3,axis=1,keepdims=True))
    dZ2 = np.dot(W3,dZ3) * (1 - np.power(A2,2))
    dW2 = (1/m) * np.dot(dZ2,A1.T)
    db2 = (1/m) *(np.sum(dZ2,axis=1,keepdims=True))
    dZ1 = np.dot(W2,dZ2) * (1 - np.power(A1,2))
    dW1 = (1/m) *(np.dot(dZ1,X.T))
    db1 = (1/m) *(np.sum(dZ1, axis=1, keepdims=True))

    
    grads = {"dW1": dW1,
             "db1": db1,
             "dW2": dW2,
             "db2": db2,
             "dW3": dW3,
             "db3": db3}
    
    return grads


#Update weights
def update_parameters(parameters, grads, learning_rate):

    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    W3 = parameters["W3"]
    b3 = parameters["b3"]
    

    dW1 = grads["dW1"]
    db1 = grads["db1"]
    dW2 = grads["dW2"]
    db2 = grads["db2"]
    dW3 = grads["dW3"]
    db3 = grads["db3"]

    
    W1 = W1 - learning_rate * dW1.T
    b1 = b1 - learning_rate * db1
    W2 = W2 - learning_rate * dW2.T
    b2 = b2 - learning_rate * db2
    W3 = W3 - learning_rate * dW3.T
    b3 = b3 - learning_rate * db3
    
    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2,
                  "W3": W3,
                  "b3": b3}
    
    return parameters

#Main
def nn_model(X, Y, learning_rate, num_iterations = 10000, print_cost=True):
    
    #Only need the first time, when there are no saved weights
    #parameters = init_parametars(X.shape[0], Y.shape[0])
    with open('parameters.pkl', 'rb') as f:
        parameters = pickle.load(f)
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]    
    W3 = parameters["W3"]
    b3 = parameters["b3"]
    
    for i in range(0, num_iterations):
        A3, cache = forward_propagation(X, parameters)
        cost = compute_cost(Y, A3)
        grads = backward_propagation(parameters, cache, X, Y)
        parameters = update_parameters(parameters, grads, learning_rate)
        if i % 100 == 0:
            with open('parameters.pkl', 'wb') as f:
                pickle.dump(parameters, f)
                print("Parametars saved to a file")
        if print_cost and i % 10 == 0:
            print ("Cost after iteration %i: %f" %(i, cost))
    return parameters

def testImage(X):
    with open('parameters.pkl', 'rb') as f:
        parameters = pickle.load(f)
    A3, cache = forward_propagation(X, parameters)
    A3 = A3.flatten()
    confidence = max(A3)
    guess = np.where(A3 == confidence)
    print("I think the answer is: {}, I am {} percent sure".format(guess[0], round(confidence, 2)))


#Train
#image_arrays,labels_arrays = prepareImages('mnist/testing')
#nn_model(image_arrays,labels_arrays, 0.002)

#Test Image
test_image_array, test_label_array = prepareImages('mnist/guess')
testImage(test_image_array)

