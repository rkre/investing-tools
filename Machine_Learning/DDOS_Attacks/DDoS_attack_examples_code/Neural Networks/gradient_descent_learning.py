                           
                           
                           
# Single Perceptron: Gradient Descent Learning and Delta Rule              
                  
# %matplotlib inline
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.utils import shuffle
class1 = pd.read_csv("Class1.txt")
class2 = pd.read_csv("Class2.txt")
print(class1.shape)
print(class2.shape)
class1.head()





# add lables to the data. .insert() will directly modify the dataframe
posLabel=1          # positive class Label
negLabel=-1         # negative class label
T=(posLabel+negLabel)/2 # This is the threshold to classify positive vs. negative

class1.insert(class1.shape[1],'label',posLabel)
class2.insert(class2.shape[1],'label',negLabel)
# combine both datasets as one
class12 = class1.append(class2)
print(class12.shape)
class12.head()




colors=["red","black","blue","green"]
plt.scatter(class12.iloc[:,0],class12.iloc[:,1],color=[colors[idx+1] for idx in class12.iloc[:,2]])
# The dataset is not a linearly searable problem.
# So we will use gradient descent learning to learn decision boundary



# partitioning the dataset into training vs. test sets
# shuffle the data 
# because we want random dataset to train 
class12_rand=shuffle(class12)
# machine learning models have tensor (feature) and labels
features,labels=class12_rand.iloc[:,0:-1],class12_rand.loc[:,['label']]
from sklearn.model_selection import train_test_split

# test size is 40% test_size = .4
# y test only has labels
# x has data
# random_state allows you to repeat the results if it is the same 
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=.4, random_state=42)

# covert data from dataframe into matrix format for arithemetic calculation
# numpy array is much quicker in calculation than pandas dataframe so we convert:
X_train_m=np.asmatrix(X_train, dtype = 'float64')
X_test_m=np.asmatrix(X_test, dtype = 'float64')
y_train_m=np.asmatrix(y_train, dtype = 'float64')
y_test_m=np.asmatrix(y_test, dtype = 'float64')

def GradientDescentLearning(features, labels, max_iter, learning_rate, err_threshold, test_features, test_labels):
    
    # random initialize weight values between rage: [-0.5,0.5]
    w = np.random.rand(features.shape[1]+1)-0.5
    
    totalSquaredErr_ = [] #d(n) - o(n)^2
    totalSquaredErrTest_ = []
    accuracy_= []
    epoch=0
    err=9999.0
    while (epoch<max_iter) and (err>err_threshold):
        misclassified = 0
        deltaw=[0]*(features.shape[1]+1)
        for i, x in enumerate(features):
            # for all instance, accumulate
            # calculate weights
            # add bias
            x = np.insert(x,0,1)

            v = np.dot(w, x.transpose())
            
            diff = learning_rate*(labels[i] - v) # e(m) - o(n)
            deltaw=deltaw+diff*x 
        
        #update weights
        #print(deltaw)
        w=w+deltaw
        
        # now calculate training error using new weights
        this_err=0
        for i, x in enumerate(features):
            x = np.insert(x,0,1)
            v = np.dot(w, x.transpose())
            this_err=this_err+(labels[i] - v)*(labels[i] - v)
        this_err=np.ndarray.item(this_err) 
        this_err=this_err/2.0
        # mean squared error
        # normalize # of instances:
        err=this_err/features.shape[0]        
        totalSquaredErr_.append(err)
        
        # now calculate test error using new weights
        this_err=0
        for i, x in enumerate(test_features):
            x = np.insert(x,0,1)
            v = np.dot(w, x.transpose())
            this_err=this_err+(test_labels[i] - v)*(test_labels[i] - v)
        this_err=np.ndarray.item(this_err) 
        this_err=this_err/2.0
        totalSquaredErrTest_.append(this_err/test_features.shape[0])
        # now calculate test classification accuracy
        # found if positive or negative
        # T is threshold or center point
        # w - T >= 0, classify as posivie but if test_label[i] == negLabel, it's a misclassification
        this_err=0
        for i, x in enumerate(test_features):
            x = np.insert(x,0,1)
            v = np.dot(w, x.transpose())
            if(((v-T)>=0 and test_labels[i]==negLabel) or ((v-T)<0 and test_labels[i]==posLabel)):
                this_err=this_err+1
        this_err=float(this_err) 
        this_err=this_err/test_features.shape[0]
        accuracy_.append(1-this_err)        
        #next epoch
        epoch=epoch+1
    return (w, totalSquaredErr_, totalSquaredErrTest_, accuracy_)


# we use maximum 500 iterations. Because gradient descent leanring weight updating is accumulated across all 
# training instances. We use set learning rate as inverse of # of training instances, to avoid weight values
# w continuosly increasing. 
max_iter = 500
# we want to adjust learning rate eta with more data
eta=1.0/X_train.shape[0] # learning rate is approx 0.001

#eta=0.02
print("Learning rate is: %.5f" % eta)
err_threshold=0.05
w, misclassified, testError, accuracy= GradientDescentLearning(X_train_m, y_train_m, max_iter, eta, err_threshold,X_test_m,y_test_m)
print(misclassified[0:10])
print(testError[0:10])
print(accuracy[0:10])

# converges at about 50 iterations or so 
epochs = np.arange(1, max_iter+1)
plt.plot(epochs, misclassified)
plt.plot(epochs, testError)
plt.plot(epochs, accuracy)
plt.xlabel('iterations')
plt.ylabel('misclassified')
plt.show()







# Now we create a plot to show learned decision boundaries (find slope and intercept)
# The decision boundary line is W2.X2 + W1.X1+ W0=T (where T is threshold, 
# which is the middle point between positive and negative class)
# If positive is labled as 1, and negative is labled as -1. The middle point T
# So we have W2.X2 + W1.X1+ W0=T
# The line is X2=-(W1/W2).X1 + (T-W0)/W2
# Therefore the slope is -(W1/W2), and the y-intercept is - W0/W2
print(w)
slope=w[0,1]/w[0,2]*(-1)
intercept=(T-w[0,0])/w[0,2]
print(slope,intercept)






xvalues=class12.iloc[:,0]
yvalues=xvalues*slope+intercept
plt.scatter(class12.iloc[:,0],class12.iloc[:,1],color=[colors[idx+1] for idx in class12.iloc[:,2]])
plt.plot(xvalues,yvalues,"g-")






# now we implement Delta rule leanring. Which use a signle instance to update the network weight value
import random
def Delta(features, labels, max_iter, learning_rate, err_threshold):
    
    # random initialize weight values between rage: [-0.5,0.5]
    w = np.random.rand(features.shape[1]+1)-0.5
    
    totalSquaredErr_ = []
    epoch=0
    err=9999.0
    while (epoch<max_iter) and (err>err_threshold):
        misclassified = 0
        deltaw=[0]*(features.shape[1]+1)
        # random select an instance
        i=random.randrange(features.shape[0])
        x=features[i,]
        x = np.insert(x,0,1)
        # update the width
        v = np.dot(w, x.transpose())
            
        diff = learning_rate*(labels[i] - v)
        deltaw=deltaw+diff*x
        
        #update weights
        #print(deltaw)
        w=w+deltaw
        
        # now calculate error using new weights
        this_err=0
        for i, x in enumerate(features):
            x = np.insert(x,0,1)
            v = np.dot(w, x.transpose())
            this_err=this_err+(labels[i] - v)*(labels[i] - v)
        this_err=np.asscalar(this_err)
        this_err=this_err/2.0
        totalSquaredErr_.append(this_err)
        #mean squared error
        err=this_err/features.shape[0] # normalize
        epoch=epoch+1
    return (w, totalSquaredErr_)

max_iter = 500
eta=0.1
err_threshold=0.05
w, misclassified= Delta(X_train_m, y_train_m, max_iter, eta, err_threshold)
print(misclassified[0:10])

epochs = np.arange(1, max_iter+1)
plt.plot(epochs, misclassified)
plt.xlabel('iterations')
plt.ylabel('Squared Errors')
plt.text(10,10,"Perceptron learning rule convergence")
plt.show()

# Now we create a plot to show learned decision boundaries (find slope and intercept)
print(w)
slope=w[0,1]/w[0,2]*(-1)
intercept=(T-w[0,0])/w[0,2]
print(slope,intercept)

xvalues=class12.iloc[:,0]
yvalues=xvalues*slope+intercept
plt.scatter(class12.iloc[:,0],class12.iloc[:,1],color=[colors[idx+1] for idx in class12.iloc[:,2]])
plt.plot(xvalues,yvalues,"g-")
print(w)






