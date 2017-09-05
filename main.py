'''
 # Author:    Narasimha Murthy
 # Created:   11.05.2009

'''
import numpy as np
from mnist import MNIST
import matplotlib.pyplot as plt
import pickle
import random

mndata = MNIST('./data')
train_images, train_labels = mndata.load_training()
train_images = np.array(train_images)
train_labels =np.array(train_labels)
test_images, test_labels = mndata.load_testing()
test_images = np.array(test_images)
test_labels =np.array(test_labels)

#display a random image from the dataset
rand = np.random.randint(100)
plt.imshow(np.reshape(train_images[rand], (28,28)))
plt.title( 'LABEL = '+str(train_labels[rand]))
#plt.show()

#Network architecture init
inp = 784
h1 = 1000
h2 = 500
h3 = 250
out =10
alpha =0.001 #learning rate
alpha_decay = 7e-4  #learning rate decay here set to 0.85 times for every 250 iterations using 1/t decay
l2_reg = 0.005
alp = 0.7
learning_deacy = False
activation = 0 # 1 sigmoid and default ReLU
train_network_from_scratch = True
batch_size = 64 #choose mini batch size
train_loss= []  #list to plot training loss
test_loss= []  #list to plot test loss
acc=[] #list to plot test_accuracy
number_of_hidden_layers =3
n_neurons= [h1,h2,h3]

def initialize(n_hidden_layers, n_neurons, x_size, y_unique):
    W = []
    b = []
    n_neurons += [y_unique]
    for i in range(n_hidden_layers+1):
        if  not i==0:
            var = 2/(n_neurons[i-1] + n_neurons[i])  #Xavier initialization
            W += [np.reshape(np.random.normal(0,  var, x_size*n_neurons[i]), (x_size, n_neurons[i]))]
        else:
            var = 2/(x_size + n_neurons[i])  #Xavier initialization
            W += [np.reshape(np.random.normal(0,  var, x_size*n_neurons[i]), (x_size, n_neurons[i]))]
        b += [np.zeros((n_neurons[i],1))]
    v= np.zeros(n_hidden_layers+1)
    vb =v= np.zeros(n_hidden_layers+1)
    return W,b,v,vb
#initialization of weights

def load_model():
    with open(r"model.txt", "rb") as input_file:
        [W,b,v,vb] = pickle.load(input_file)


if train_network_from_scratch:
    W,b,v,vb =initialize(number_of_hidden_layers,n_neurons,784,10)

else:
    load_model('model.txt')


#evaluate some important functions
def sigmoid(x):
    return 1/(1+np.exp(-x))

def softmax(out):
    C =np.max(out) #to achieve numerical stability by making all values negative
    out =out - C
    denominator =np.sum(np.exp(out))
    return np.exp(out)/denominator
def ReLU(x):
    x = x.clip(min=0)
    return x

#code for forward propogation
def forward_propogation(X): #takes a 784 dimensional vector as input and outputs logitd
    X =np.reshape(X,(784,1))
    global W, b, activation
    if activation == 1:
        z1 = np.matmul(np.transpose(W[0]),X) +b[0]
        a1 = sigmoid(z1)
        z2 = np.matmul(np.transpose(W[1]),a1)+b[1]
        a2 = sigmoid(z2)
        z3 = np.matmul(np.transpose(W[2]),a2)+b[2]
        a3 = sigmoid(z3)
        z4 = np.matmul(np.transpose(W[3]),a3)+b[3]
        logits =softmax(z4)
    else:
        z1 = np.matmul(np.transpose(W[0]),X) +b[0]
        a1 = ReLU(z1)
        z2 = np.matmul(np.transpose(W[1]),a1)+b[1]
        a2 = ReLU(z2)
        z3 = np.matmul(np.transpose(W[2]),a2)+b[2]
        a3 = ReLU(z3)
        z4 = np.matmul(np.transpose(W[3]),a3)+b[3]
        logits =softmax(z4)
    return z1, a1, z2, a2, z3, a3, z4, logits

#notion of loss function and mean loss
def cross_entropy_loss(X_data, Y_data):
    global l2_reg
    loss = 0
    for x,y in zip(np.array(X_data), np.array(Y_data)):
        z1, a1, z2, a2, z3, a3, z4, logits = forward_propogation(x)
        max_activation = np.max(z4)
        a_true = z4[y]-max_activation
        loss += a_true - np.log(np.sum(np.exp(z4 - max_activation))) #as true probability is 1 for y and 0 for everthing else

    return -loss/Y_data.shape[0] + (l2_reg/2)*((np.linalg.norm(W1))**2+(np.linalg.norm(W2))**2+ (np.linalg.norm(W3))**2+(np.linalg.norm(W4))**2)


#functions for some derivatives
def dlog(x):
    return 1/(x+e-10) #prevent exploding gradient
def dlogit(x, label):
    X = x
    X[label] =X[label]- 1
    return np.array(X)
def dsigmoid(x):
    return np.array(sigmoid(x)*(1-sigmoid(x)))
def dReLU(x):
    x =x>0
    return x


#code for backward propogation of gradients
def back_propogation(x,z1, a1, z2, a2, z3, a3, z4, logits,train_label):
    global W, activation
    if activation ==1:
        x =np.reshape(x,(784,1))
        dz4 = dlogit(logits, train_label)
        db4 =dz4
        dW4 = np.matmul(a3,dz4.T)
        da3  = np.matmul(W[3],dz4)
        dz3 = da3*dsigmoid(z3)
        db3 =dz3
        dW3 = np.matmul(a2,dz3.T)
        da2 = np.matmul(W[2],dz3)
        dz2 = da2*dsigmoid(z2)
        db2 =dz2
        dW2 = np.matmul(a1,dz2.T)
        da1 = np.matmul(W[1],dz2)
        dz1 = da1*dsigmoid(z1)
        db1 =dz1
        dW1 = np.matmul(x,dz1.T)
    else:
        x =np.reshape(x,(784,1))
        dz4 = dlogit(logits, train_label)
        db4 =dz4
        dW4 = np.matmul(a3,dz4.T)
        da3  = np.matmul(W4,dz4)
        dz3 = da3*dReLU(z3)
        db3 =dz3
        dW3 = np.matmul(a2,dz3.T)
        da2 = np.matmul(W3,dz3)
        dz2 = da2*dReLU(z2)
        db2 =dz2
        dW2 = np.matmul(a1,dz2.T)
        da1 = np.matmul(W2,dz2)
        dz1 = da1*dReLU(z1)
        db1 =dz1
        dW1 = np.matmul(x,dz1.T)
    return [dW1,dW2,dW3,dW4],[db1,db2,db3,db4]

def predict(x_data):
    y=[]
    for x in x_data:
        z1, a1, z2, a2, z3, a3, z4, logits = forward_propogation(x)
        y.append(np.argmax(logits))
        #print np.max(logits)
    return np.array(y)
def accuracy(x_data, y_data):
    y_prediction = predict(x_data)
    true = y_prediction == y_data
    return float(np.sum(true))/y_data.shape[0]

def update_params(dW,db):
    #SGD with momentum update
    global W,b,alpha,alp,v,vb
    g = alpha*(l2_reg*W + dW)
    gb = alpha*db
    v = alp*v - g
    vb = alp*vb - gb
    W += v
    b += vb



pair =zip(train_images, train_labels)
train_data = [list(t) for t in pair]
random.shuffle(train_data)  #make random permutation of training data

#training the network for 8000 iterations
i=0
num_of_training_iterations =2000


for a in range(num_of_training_iterations):
    dW =0
    dB =0

    if i < len(train_data)/batch_size -1:
         i +=1
    else:
        i=0

    data = np.array(train_data[i*batch_size : (i+1)*(batch_size)])
    for j in range(batch_size):
        x =data[j,0]
        y =data[j,1]
        z1, a1, z2, a2, z3, a3, z4, logits = forward_propogation(x)
        dw,db = back_propogation(x,z1, a1, z2, a2, z3, a3, z4, logits,y)
        dW +=dw
        dB +=db
    dW/=batch_size
    dB/=batch_size
    update_params(dW,dB)

    if learning_deacy:  #anneal the learning rate
        alpha = alpha/(1+alpha_decay*a)

    loss = cross_entropy_loss(data[:,0], data[:,1])
    print "iteration "+str(a)+" : "+str(loss)
    train_loss +=[loss]
    if a%250 == 0:
        test_loss += [cross_entropy_loss(np.array(test_images), np.array(test_labels))]
        acc += [accuracy(test_images, test_labels)]
        print "test_loss is :"+str(test_loss[int(a/250)])
        print "accuracy sigmoid 01b : "+ str(acc[int(a/250)])


print "accuracy = "+ str(accuracy(test_images, test_labels))

#take 20 random images from test data and show the top three predicitons predictions
np.random.seed(5)
limit = test_labels.shape[0]
index  = np.random.randint(limit, size=20)
predicted = predict(test_images[index], test_labels[index])
print predicted

#saving the model
model = [W, b, v, vb]
with open("model.txt", "wb") as fp:
    pickle.dump(model, fp)


#saving numpy arrays to plot
np.save("train_loss0001r.npy", np.array(train_loss))
np.save("test_loss0001r.npy", np.array(test_loss))
np.save("test_accuracy0001r.npy", np.array(acc))
