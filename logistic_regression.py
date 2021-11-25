# -*- coding: utf-8 -*-
"""
Created on Thu Nov 25 16:35:18 2021

@author: amanullah.awan
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings( "ignore")
#from sklearn.linear_model import LogisticRegression
#from sklearn.metrics import confusion_matrix, f1_score, accuracy_score


class Logistic_Regression() :
    def __init__( self, learning_rate, iterations ) :        
        self.learning_rate = learning_rate        
        self.iterations = iterations
          
   
    def fit( self, X, Y ) :                 
        self.m, self.n = X.shape        
        # weight initialization        
        self.W = np.zeros( self.n )        
        self.b = 0        
        self.X = X        
        self.Y = Y
          
        # gradient descent learning
                  
        for i in range( self.iterations ) :            
            self.update_weights()            
        return self
      
    #Updating Weights Function     
    def update_weights( self ) :           
        A = 1 / ( 1 + np.exp( - ( self.X.dot( self.W ) + self.b ) ) )
          
        # Gradient Calculation        
        temp = ( A - self.Y.T )        
        temp = np.reshape( temp, self.m )        
        dW = np.dot( self.X.T, temp ) / self.m         
        db = np.sum( temp ) / self.m 
          
        # update weights    
        self.W = self.W - self.learning_rate * dW    
        self.b = self.b - self.learning_rate * db
          
        return self
      
    
    #Prediction Function  
    def predict( self, X ) :    
        Z = 1 / ( 1 + np.exp( - ( X.dot( self.W ) + self.b ) ) )        
        Y = np.where( Z > 0.5, 1, 0 )        
        return Y




df = pd.read_csv("D:/kaggle/diabetes.csv")
df.head()



X = df.iloc[:,:-1].values
Y = df.iloc[:,-1:].values




X_train, X_test, Y_train, Y_test = train_test_split(
      X, Y, test_size = 0.45, random_state = 0 )



model = Logistic_Regression( learning_rate = 0.01, iterations = 1000 )


model.fit( X_train, Y_train )    


Y_pred = model.predict( X_test )    



correctly_classified = 0    



count = 0
for count in range( np.size( Y_pred ) ) :
    if Y_test[count] == Y_pred[count] :
        correctly_classified = correctly_classified + 1
    count = count + 1
    
    

print( "Accuracy on test set by our model:  ", (correctly_classified / count ) * 100 )




