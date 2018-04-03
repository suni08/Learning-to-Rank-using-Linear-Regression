# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

from sklearn.cluster import KMeans
import numpy as np
import scipy.stats
import pandas as pd
import random
import matplotlib.pyplot as pt

#Print details of each team member
print("UBitName\t=\tBhumika Khatwani\t\tSunita Pattanayak")
print("personNumber\t=\t50247656\t\t\t50249134")

#Read dataset from given dataset

#initialization
sheet3=[]
sheet4=[]

#initialising hyperparams
M=10
M_syn=200
L2_lambda=0.1
L2_lambda_syn=0.1
learning_rate=0.01
learning_rate_syn=0.01
epochs=1000
epochs_syn=1000

#Reading LeToR dataset
filepath = "Querylevelnorm.txt"
file_object  = open(filepath)

#extracting label vector and dataset matrix from MQ2007 dataset
for row in file_object.readlines():
    item1=[item.split(':')[1] for item in row.split(' ')[2:48]]
    item1=np.asfarray(item1,float)
    sheet3.append(item1)
    sheet4.append(row.split(' ')[0])
file_object.close()
    
data=np.matrix(sheet3)          #2D matrix with 69623 rows and 46 columns
labels= np.array(sheet4)        #list vector of labels, target vector
print("\n 2D matrix containing 46 feature vectors with 69623 rows : \n")
print(data)
print("\n List vector containing 69623 values of labels : \n")
print(labels)

#Reading Synthetic dataset

#Reading input dataset from input.csv
filepath1 = "input.csv"
sheet1  = pd.read_csv(filepath1,encoding = "ISO-8859-1")

#Reading output dataset from outout.csv
filepath2 = "output.csv"
sheet2  = pd.read_csv(filepath2,encoding = "ISO-8859-1")

#Converting string values to float in the read values
sheet1 = sheet1.convert_objects(convert_numeric=True)
sheet2 = sheet2.convert_objects(convert_numeric=True)

data_syn=np.matrix(sheet1)
labels_syn= np.array(sheet2)
print("\n 2D matrix containing 10 columns with 20000 rows : \n")
print(data_syn)
print("\n List vector containing 20000 values of outputs : \n")
print(labels_syn)


#data partitioning

#input data of LeToR partitioning
df=pd.DataFrame(data)
train, validate, test = np.split(df.sample(frac=1), [int(.8*len(df)), int(.9*len(df))])
print("Training data : \n")
print(train)
print("Validation data : \n")
print(validate)
print("Test data : \n")
print(test)

#output data of LeToR partitioning
df1=pd.DataFrame(labels)
train_output, validate_output, test_output = np.split(df1.sample(frac=1), [int(.8*len(df)), int(.9*len(df))])
print("Training data label : \n")
print(train_output)
print("Validation data label : \n")
print(validate_output)
print("Test data label : \n")
print(test_output)


#input data of Synthetic dataset partitioning
df=pd.DataFrame(data_syn)
train_syn, validate_syn, test_syn = np.split(df.sample(frac=1), [int(.8*len(df)), int(.9*len(df))])
print("Training data : \n")
print(train_syn)
print("Validation data : \n")
print(validate_syn)
print("Test data : \n")
print(test_syn)

#output data of Synthetic dataset partitioning
df1=pd.DataFrame(labels_syn)
train_output_syn, validate_output_syn, test_output_syn = np.split(df1.sample(frac=1), [int(.8*len(df)), int(.9*len(df))])
print("Training data label : \n")
print(train_output_syn)
print("Validation data label : \n")
print(validate_output_syn)
print("Test data label : \n")
print(test_output_syn)

#defining methods
#METHOD1: method for finding closed form solution i.e calculating Wml 
def closed_form_sol_with_least_squared_regularizarion (L2_lambda, design_matrix, output_data):
    return np.linalg.solve(L2_lambda * np.identity(design_matrix.shape[1]) +np.matmul(design_matrix.T, design_matrix),np.matmul(design_matrix.T, output_data)).flatten()

#method for computing design matrix
def compute_design_matrix(X, centers, spreads):
    basis_func_outputs = np.exp(np.sum(np.matmul(X - centers, spreads)*(X - centers),axis=2)/(-2.0)).T
    return np.insert(basis_func_outputs, 0, 1, axis=1)  # inserting 1's to the 1st col
    
#METHOD2: method for computing sgd form solution 
def SGD_sol(learning_rate,minibatch_size,num_epochs,L2_lambda,design_matrix,output_data):
    N,D = design_matrix.shape
    weights = np.zeros([1,D])

    for epoch in range(num_epochs):
        for i in range(int(N / minibatch_size)):
            lower_bound = i * minibatch_size
            upper_bound = min((i+1)*minibatch_size, N)
            Phi = design_matrix[lower_bound : upper_bound, :]
            t = np.array(output_data,dtype=np.int)[lower_bound : upper_bound, :]
            #print("Phi",Phi.shape)
            #print("t",t.shape)
            #print("weights",weights.shape)
            E_D = np.matmul((np.matmul(Phi, weights.T)-t).T,Phi)
            E = (E_D + L2_lambda * weights)/ minibatch_size
            weights = weights - (learning_rate * E)
    return weights.flatten()

def choose_centers(data,kmeans):
      centers = kmeans.cluster_centers_
      return centers     

def get_labels(data,kmeans):
      labels = kmeans.labels_
      return labels

def get_spread_diagonal_matrix(data,kmeans,labels):
      kmeans_list={i: data[np.where(labels == i)] for i in range(kmeans.n_clusters)}
      spreads=list((np.cov(kmeans_list[i].T)) for i in range(kmeans.n_clusters))
      i,j,k = np.shape(spreads)
      spread_diagonal=[]
      for x in range(i):
          spread_diagonal.append(np.diagflat(np.diagonal(spreads[x])))
      return spread_diagonal
      
def err_func (data, closed_form):
      N,D = data.shape
      erms_dist=0
      output_data=np.array(data,dtype=np.int)
      for i in range(N):
          erms_dist=erms_dist+((closed_form[i]-output_data[i])**2)
      err=0.5*erms_dist
      return err                  

def derivative_err_func (data, weights, centers):
    return derivative_err

#train model parameter
### LeToR dataset ###

#Calculating centers,spreads and deign matrix of training data
print("\n Considering Training data, implement kmeans algorithm : \n")
X=np.array(train)                                                              #data to be considered
kmeans = KMeans(n_clusters=M, random_state=0).fit(X)
centers = choose_centers(X,kmeans)                                             #calculating centers after implementing kmean clustering algo
labels = get_labels(X,kmeans)                                                  #getting labels after implementing kmean clustering algo
spread= get_spread_diagonal_matrix(X,kmeans,labels)

print("\n Centers of clusters after implementing kmean : \n")
print(centers)
print("\n Spreads of clusters after implementing kmean (diagonal matrix): \n")
print(spread)

centers = centers[:,np.newaxis, :]                                              #changing into 3D for computation of design matrix

print("\n Training data design matrix : \n")
design_matrix = compute_design_matrix(X, centers, spread)
print(design_matrix)

#Calculating centers,spreads and deign matrix of validation data
print("\n Considering Validation data, implement kmeans algorithm : \n")
X=np.array(validate)                                                              #data to be considered
kmeans = KMeans(n_clusters=M, random_state=0).fit(X)
centers_val = choose_centers(X,kmeans)                                             #calculating centers after implementing kmean clustering algo
labels_val = get_labels(X,kmeans)                                                  #getting labels after implementing kmean clustering algo
spread_val = get_spread_diagonal_matrix(X,kmeans,labels_val)

print("\n Centers of clusters after implementing kmean : \n")
print(centers_val)
print("\n Spreads of clusters after implementing kmean (diagonal matrix): \n")
print(spread_val)

centers_val = centers_val[:,np.newaxis, :]                                              #changing into 3D for computation of design matrix

print("\n Validation data design matrix : \n")
design_matrix_val = compute_design_matrix(X, centers_val, spread_val)           #computing design matrix Phi
print(design_matrix_val)
 
# Closed-form with least squared regularizarion solution for training data
print("Training data Closed form solution with least squared regularizarion :")
wml= closed_form_sol_with_least_squared_regularizarion (L2_lambda,np.array(design_matrix),np.array(train_output,dtype=np.float))
print("\n Wml value for training data in closed form solution is :\n",wml)                                                      #print Wml (weights) for training data
closed_form=np.matmul(wml.T,design_matrix.T)                                                                                    #multiplying weights to design matrix to get closed form solution
print("\n Closed form solution for training data in closed form solution is :\n",closed_form)                                   #printing closed form solution found for training data


# Closed-form with least squared regularizarion solution for validation data     
print("Validation data Closed form solution with least squared regularizarion :")
wml_val= closed_form_sol_with_least_squared_regularizarion (L2_lambda,np.array(design_matrix_val),np.array(validate_output,dtype=np.float))
print("\n Wml value for validation data in closed form solution is :\n",wml_val)                                                #print Wml (weights) for validation data
closed_form_val=np.matmul(wml.T,design_matrix.T)                                                                                #multiplying weights to design matrix to get closed form solution
print("\n Closed form solution for validation data in closed form solution is :\n",closed_form_val)                                 #printing closed form solution found for training data


#finding error function
#training value error
N,D = train_output.shape
err_train=err_func(train_output,closed_form)
print("\n Training error: ",err_train)                                            #error in training
ems=err_train/N                                                                   #normalised error
print("\n Training error normalised: ",ems)


#validate value error
N,D = validate_output.shape
err_val=err_func(validate_output,closed_form)
print("\n Training error: ",err_val)                                         #error in validation
ems_val=err_val/N                                                            #normalised error
print("\n Training error normalised: ",ems_val)

#Finding error difference
error_difference=ems-ems_val
print("\n Error difference : ",error_difference)

#To minimise error difference, we varied the values of M and L2_lambda and finalised a single value which we use on the test data

#Test data tuned with the fixed hyper parameters
#Calculating centers,spreads and deign matrix of test data
print("\n Considering Test data, implement kmeans algorithm : \n")
X=np.array(test)                                                                    #data to be tested
kmeans = KMeans(n_clusters=M, random_state=0).fit(X)
centers_test = choose_centers(X,kmeans)                                             #calculating centers after implementing kmean clustering algo
labels_test = get_labels(X,kmeans)                                                  #getting labels after implementing kmean clustering algo
spread_test = get_spread_diagonal_matrix(X,kmeans,labels_test)

print("\n Centers of clusters after implementing kmean : \n")
print(centers_test)
print("\n Spreads of clusters after implementing kmean (diagonal matrix): \n")
print(spread_test)

centers_test = centers_test[:,np.newaxis, :]                                    #changing into 3D for computation of design matrix

print("\n Test data design matrix : \n")
design_matrix_test = compute_design_matrix(X, centers_test, spread_test)           #computing design matrix Phi
print(design_matrix_test)

# Closed-form with least squared regularizarion solution for validation data     
print("Test data Closed form solution with least squared regularizarion :")
wml_test= closed_form_sol_with_least_squared_regularizarion (L2_lambda,np.array(design_matrix_test),np.array(test_output,dtype=np.float))
print("\n Wml value for validation data in closed form solution is :\n",wml_test)                                              #print Wml (weights) for test data
closed_form_test=np.matmul(wml_test.T,design_matrix.T)                                                                         #multiplying weights to design matrix to get closed form solution
print("\n Closed form solution for test data is :\n",closed_form_test)                                                         #printing closed form solution found for test data

#test value error
N,D = test_output.shape
err_test=err_func(test_output,closed_form)
print("\n Test error: ",err_test)                                           #error in test
ems_test=err_test/N                                                             #normalised error
print("\n Test error normalised: ",ems_test)
erms=(2.0*(ems_test))**0.5
print("\n Test error rms for closed form solution : ",erms)                                                 #print Erms value from closed form solution

### sgd approach ###

#Gradient descent solution for training data
N,D = train_output.shape
print("\n Training data SGD solution :\n")                                      
wml_sgd= SGD_sol(learning_rate,N,epochs,L2_lambda,design_matrix,train_output)         
print("\n SGD form Wml for training data is :\n",wml_sgd)                       #print Wml (weights) for training data

sgd_form=np.matmul(wml_sgd.T,design_matrix.T)                                   #multiplying weights to design matrix to get SGD form solution
print("Final SGD form solution for training data : \n")
print("\n SGD form solution for training data is :\n",sgd_form)                 #printing SGD form solution found for training data

#Gradient descent solution for validation data
N,D = validate_output.shape
print("\n Validation data SGD solution :\n")                                      
wml_val_sgd= SGD_sol(learning_rate,N,epochs,L2_lambda,design_matrix_val,validate_output)         
print("\n SGD form Wml for validation data is :\n",wml_sgd)                             #print Wml (weights) for training data

sgd_form_val=np.matmul(wml_val_sgd.T,design_matrix_val.T)                               #multiplying weights to design matrix to get SGD form solution
print("Final SGD form solution for validation data : \n")
print("\n SGD form solution for validation data is :\n",sgd_form_val)                   #printing SGD form solution found for training data

#finding error function
#training value error
N,D = train_output.shape
err_train=err_func(train_output,sgd_form)
print("\n Training error: ",err_train)                                            #error in training
ems=err_train/N                                                                   #normalised error
print("\n Training error normalised: ",ems)

#validate value error
N,D = validate_output.shape
err_val=err_func(validate_output,sgd_form)
print("\n Validation error: ",err_val)                                         #error in validation
ems_val=err_val/N                                                            #normalised error
print("\n Validation error normalised: ",ems_val)

#Finding error difference
error_difference=ems-ems_val
print("\n Error difference : ",error_difference)

#To minimise error difference, we varied the values of M and L2_lambda and finalised a single value which we use on the test data

#We already have the design matrix of test data so we calculate the error

#Gradient descent solution for test data
N,D = test_output.shape
print("\n Test data SGD solution :\n")                                      
wml_test_sgd= SGD_sol(learning_rate,N,epochs,L2_lambda,design_matrix_test,test_output)         
print("\n SGD form Wml for test data is :\n",wml_test_sgd)                             #print Wml (weights) for test data

sgd_form_test=np.matmul(wml_test_sgd.T,design_matrix_test.T)                            #multiplying weights to design matrix to get SGD form solution
print("Final SGD form solution for test data : \n")
print("\n SGD form solution for test data is :\n",sgd_form_test)                   #printing SGD form solution found for test data

#test value error
N,D = test_output.shape
err_test=err_func(test_output,sgd_form_test)
print("\n Test error: ",err_test)                                               #error in test
ems_test=err_test/N                                                             #normalised error
print("\n Test error normalised: ",ems_test)
erms=(2.0*(ems_test))**0.5
print("\n Test error rms for SGD form solution : ",erms)                     #print Erms value from SGD form solution

### Synthetic dataset ####

#Calculating centers,spreads and deign matrix of training data
print("\n Considering Training data, implement kmeans algorithm : \n")
X=np.array(train_syn)                                                              #data to be considered
kmeans = KMeans(n_clusters=M_syn, random_state=0).fit(X)
centers = choose_centers(X,kmeans)                                             #calculating centers after implementing kmean clustering algo
labels = get_labels(X,kmeans)                                                  #getting labels after implementing kmean clustering algo
spread= get_spread_diagonal_matrix(X,kmeans,labels)

print("\n Centers of clusters after implementing kmean : \n")
print(centers)
print("\n Spreads of clusters after implementing kmean (diagonal matrix): \n")
print(spread)

centers = centers[:,np.newaxis, :]                                              #changing into 3D for computation of design matrix

print("\n Training data design matrix : \n")
design_matrix = compute_design_matrix(X, centers, spread)
print(design_matrix)

#Calculating centers,spreads and deign matrix of validation data
print("\n Considering Validation data, implement kmeans algorithm : \n")
X=np.array(validate_syn)                                                              #data to be considered
kmeans = KMeans(n_clusters=M_syn, random_state=0).fit(X)
centers_val = choose_centers(X,kmeans)                                             #calculating centers after implementing kmean clustering algo
labels_val = get_labels(X,kmeans)                                                  #getting labels after implementing kmean clustering algo
spread_val = get_spread_diagonal_matrix(X,kmeans,labels_val)

print("\n Centers of clusters after implementing kmean : \n")
print(centers_val)
print("\n Spreads of clusters after implementing kmean (diagonal matrix): \n")
print(spread_val)

centers_val = centers_val[:,np.newaxis, :]                                              #changing into 3D for computation of design matrix

print("\n Validation data design matrix : \n")
design_matrix_val = compute_design_matrix(X, centers_val, spread_val)           #computing design matrix Phi
print(design_matrix_val)
 
# Closed-form with least squared regularizarion solution for training data
print("Training data Closed form solution with least squared regularizarion :")
wml= closed_form_sol_with_least_squared_regularizarion (L2_lambda_syn,np.array(design_matrix),np.array(train_output_syn,dtype=np.float))
print("\n Wml value for training data in closed form solution is :\n",wml)                                                      #print Wml (weights) for training data
closed_form=np.matmul(wml.T,design_matrix.T)                                                                                    #multiplying weights to design matrix to get closed form solution
print("\n Closed form solution for training data in closed form solution is :\n",closed_form)                                   #printing closed form solution found for training data


# Closed-form with least squared regularizarion solution for validation data     
print("Validation data Closed form solution with least squared regularizarion :")
wml_val= closed_form_sol_with_least_squared_regularizarion (L2_lambda_syn,np.array(design_matrix_val),np.array(validate_output_syn,dtype=np.float))
print("\n Wml value for validation data in closed form solution is :\n",wml_val)                                                #print Wml (weights) for validation data
closed_form_val=np.matmul(wml_val.T,design_matrix_val.T)                                                                                #multiplying weights to design matrix to get closed form solution
print("\n Closed form solution for validation data in closed form solution is :\n",closed_form_val)                                 #printing closed form solution found for training data


#finding error function
#training value error
N,D = train_output_syn.shape
err_train=err_func(train_output_syn,closed_form)
print("\n Training error: ",err_train)                                            #error in training
ems=err_train/N                                                                   #normalised error
print("\n Training error normalised: ",ems)


#validate value error
N,D = train_output_syn.shape
err_val=err_func(validate_output_syn,closed_form_val)
print("\n Training error: ",err_val)                                         #error in validation
ems_val=err_val/N                                                            #normalised error
print("\n Training error normalised: ",ems_val)

#Finding error difference
error_difference=ems-ems_val
print("\n Error difference : ",error_difference)

#To minimise error difference, we varied the values of M and L2_lambda and finalised a single value which we use on the test data

#Test data tuned with the fixed hyper parameters
#Calculating centers,spreads and deign matrix of test data
print("\n Considering Test data, implement kmeans algorithm : \n")
X=np.array(test_syn)                                                                    #data to be tested
kmeans = KMeans(n_clusters=M_syn, random_state=0).fit(X)
centers_test = choose_centers(X,kmeans)                                             #calculating centers after implementing kmean clustering algo
labels_test = get_labels(X,kmeans)                                                  #getting labels after implementing kmean clustering algo
spread_test = get_spread_diagonal_matrix(X,kmeans,labels_test)

print("\n Centers of clusters after implementing kmean : \n")
print(centers_test)
print("\n Spreads of clusters after implementing kmean (diagonal matrix): \n")
print(spread_test)

centers_test = centers_test[:,np.newaxis, :]                                    #changing into 3D for computation of design matrix

print("\n Test data design matrix : \n")
design_matrix_test = compute_design_matrix(X, centers_test, spread_test)           #computing design matrix Phi
print(design_matrix_test)

# Closed-form with least squared regularizarion solution for validation data     
print("Test data Closed form solution with least squared regularizarion :")
wml_test= closed_form_sol_with_least_squared_regularizarion (L2_lambda_syn,np.array(design_matrix_test),np.array(test_output_syn,dtype=np.float))
print("\n Wml value for test data in closed form solution is :\n",wml_test)                                              #print Wml (weights) for test data
closed_form_test=np.matmul(wml_test.T,design_matrix_test.T)                                                                         #multiplying weights to design matrix to get closed form solution
print("\n Closed form solution for test data is :\n",closed_form_test)                                                         #printing closed form solution found for test data

#test value error
N,D = train_output_syn.shape
err_test=err_func(test_output_syn,closed_form_test)
print("\n Test error: ",err_test)                                           #error in test
ems_test=err_test/N                                                             #normalised error
print("\n Test error normalised: ",ems_test)
erms=(2.0*(ems_test))**0.5
print("\n Test error rms for closed form solution : ",erms)                                                 #print Erms value from closed form solution

### sgd approach ###

#Gradient descent solution for training data
N,D = train_output.shape
print("\n Training data SGD solution :\n")                                      
wml_sgd= SGD_sol(learning_rate_syn,N,epochs_syn,L2_lambda_syn,design_matrix,train_output_syn)         
print("\n SGD form Wml for training data is :\n",wml_sgd)                       #print Wml (weights) for training data

sgd_form=np.matmul(wml_sgd.T,design_matrix.T)                                   #multiplying weights to design matrix to get SGD form solution
print("Final SGD form solution for training data : \n")
print("\n SGD form solution for training data is :\n",sgd_form)                 #printing SGD form solution found for training data

#Gradient descent solution for validation data
N,D = validate_output.shape
print("\n Validation data SGD solution :\n")                                      
wml_val_sgd= SGD_sol(learning_rate_syn,N,epochs_syn,L2_lambda_syn,design_matrix_val,validate_output_syn)         
print("\n SGD form Wml for validation data is :\n",wml_val_sgd)                             #print Wml (weights) for training data

sgd_form_val=np.matmul(wml_val_sgd.T,design_matrix_val.T)                               #multiplying weights to design matrix to get SGD form solution
print("Final SGD form solution for validation data : \n")
print("\n SGD form solution for validation data is :\n",sgd_form_val)                   #printing SGD form solution found for training data

#finding error function
#training value error
N,D = train_output_syn.shape
err_train=err_func(train_output_syn,sgd_form)
print("\n Training error: ",err_train)                                            #error in training
ems=err_train/N                                                                   #normalised error
print("\n Training error normalised: ",ems)

#validate value error
N,D = train_output_syn.shape
err_val=err_func(validate_output_syn,sgd_form_val)
print("\n Validation error: ",err_val)                                         #error in validation
ems_val=err_val/N                                                            #normalised error
print("\n Validation error normalised: ",ems_val)

#Finding error difference
error_difference=ems-ems_val
print("\n Error difference : ",error_difference)

#To minimise error difference, we varied the values of M and L2_lambda and finalised a single value which we use on the test data

#We already have the design matrix of test data so we calculate the error

#Gradient descent solution for test data
N,D = test_output_syn.shape
print("\n Test data SGD solution :\n")                                      
wml_test_sgd= SGD_sol(learning_rate_syn,N,epochs_syn,L2_lambda_syn,design_matrix_test,test_output_syn)         
print("\n SGD form Wml for test data is :\n",wml_test_sgd)                             #print Wml (weights) for test data

sgd_form_test=np.matmul(wml_test_sgd.T,design_matrix_test.T)                               #multiplying weights to design matrix to get SGD form solution
print("Final SGD form solution for test data : \n")
print("\n SGD form solution for test data is :\n",sgd_form_test)                   #printing SGD form solution found for test data

#test value error
N,D = train_output_syn.shape
err_test=err_func(test_output_syn,sgd_form_test)
print("\n Test error: ",err_test)                                               #error in test
ems_test=err_test/N                                                             #normalised error
print("\n Test error normalised: ",ems_test)
erms=(2.0*(ems_test))**0.5
print("\n Test error rms for SGD form solution : ",erms)                     #print Erms value from SGD form solution









      

