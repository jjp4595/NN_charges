import numpy as np
import pandas as pd
import os

from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop, Adadelta, Adagrad, Adam, Nadam, SGD
from keras.callbacks import EarlyStopping, TerminateOnNaN
from keras import backend as K
from keras.losses import mean_squared_error


from sklearn.preprocessing import MinMaxScaler, PowerTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

import matplotlib.pyplot as plt

def JP_lowZ(m,Z,theta):
    return (0.383*(Z**-1.858)) * np.exp((-theta**2)/1829) * (m**(1/3))
def JP_highZ(m,Z,theta):
    return (0.557*(Z**-1.663)) * np.exp((-theta**2)/2007) * (m**(1/3)) 




#function to compute the room_mean_squared_error given the ground truth (y_true) and the predictions(y_pred)
def root_mean_squared_error(y_true, y_pred):
        return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1)) 



def phy_loss_mean(params):
	# useful for cross-checking training
    zimpdiff, lam, thetaimpdiff, lam2 = params
    def loss(y_true,y_pred):
        return K.mean(K.relu(zimpdiff)) + K.mean(K.relu(thetaimpdiff))
    return loss


#function to calculate the combined loss = sum of rmse and phy based loss
def combined_loss(params):
    zimpdiff, lam, thetaimpdiff, lam2 = params
    def loss(y_true,y_pred):
        return mean_squared_error(y_true, y_pred) + lam * K.mean(K.relu(zimpdiff)) + lam2 * K.mean(K.relu(thetaimpdiff))
    return loss


def PGNN_train_test(optimizer_name, optimizer_val, drop_frac, lamda, lamda2):
        
    # Hyper-parameters of the training process
    batch_size = 50
    num_epochs = 100
    val_frac = 0.1
    patience_val = int(0.4*num_epochs)
    
    # # Initializing results filename
    # exp_name = optimizer_name + '_drop' + str(drop_frac) + '_usePhy' + str(use_YPhy) + '_lamda' + str(lamda)
    # exp_name = exp_name.replace('.','pt')
    # results_dir = '../results/'
    # model_name = results_dir + exp_name + '_model.h5' # storing the trained model
    # results_name = results_dir + exp_name + '_results.mat' # storing the results of the model
    
    # Load features (Xc) and target values (Y)
     
    filename = os.environ['USERPROFILE'] + r"\Dropbox\Papers\PaperNN_charges\datasets\spherical.csv"
    data = pd.read_csv(filename, header = None)
    data= data.values
    
    # split into input (X) and output (Y) variables
    X = data[:,[2,3]]
    y = data[:,4]
    y_og = y.reshape(len(y),1)
    y = y_og
    
    X_scaled, y_scaled = X, y
    #Scaling X
    scaler = MinMaxScaler(feature_range=(0,1))
    X_scaled = scaler.fit_transform(X)
    
    #scaling y
    scaler2 = PowerTransformer()
    #scaler2 = MinMaxScaler(feature_range=(0,1))
    scaler_y = scaler2.fit(y)
    y_scaled = scaler_y.transform(y)
    
    scaler_y2 = scaler.fit(y_scaled)
    y_scaled = scaler_y2.transform(y_scaled)


    #Split data to 90% train & 10% unseen
    X_train, X_unseen, y_train, y_unseen = train_test_split(X_scaled, y_scaled, test_size=0.10, random_state=32)

        

       
    
    # Creating the model
    model = Sequential()     
    
    for layer in np.arange(n_layers):
        if layer == 0:
            model.add(Dense(n_nodes, input_shape=(np.shape(X_scaled)[1],), activation='relu'))
        else:
            model.add(Dense(n_nodes, activation='relu'))
        model.add(Dropout(drop_frac))
    model.add(Dense(1, activation='linear'))
        
    

    
    
    # Defining data for physics-based regularization, Z Condition
    zX1 =  X_scaled[0:3400,:]  # X at Z i for every pair of consecutive Z values
    zX2 =  X_scaled[200::,:]# X at Z i + 1 for every pair of consecutive Z values
    zin1 = K.constant(value=zX1) # input at Z i
    zin2 = K.constant(value=zX2) # input at Z i + 1
    lam = K.constant(value=lamda) # regularization hyper-parameter
    zout1 = model(zin1) # model output at Z i
    zout2 = model(zin2) # model output at Z i + 1
    zimpdiff = zout2 - zout1 # difference in impulse estimates at every pair of consecutive z values    
    
    # Defining data for physics-based regularization, Z Condition
    thetaX1 =  X_scaled[np.argsort(X_scaled[:,1])][0:-18,:]  # X at theta i for every pair of consecutive theta values
    thetaX2 =  X_scaled[np.argsort(X_scaled[:,1])][18::,:]   # X at theta i + 1 for every pair of consecutive theta values
    thetaval1, thetaval2 = [], []
    inds = 0
    while inds < len(thetaX1): 
        thetaval1.append(thetaX1[inds:inds+18,:][np.argsort(thetaX1[inds:inds+18,0])])
        inds+=18
    inds = 0
    while inds < len(thetaX2): 
        thetaval2.append(thetaX2[inds:inds+18,:][np.argsort(thetaX2[inds:inds+18,0])])
        inds+=18   
    thetaX1, thetaX2  = np.concatenate(thetaval1), np.concatenate(thetaval2)
    
    thetain1 = K.constant(value=thetaX1) # input at theta i
    thetain2 = K.constant(value=thetaX2) # input at theta i + 1
    lam2 = K.constant(value=lamda2) # regularization hyper-parameter
    thetaout1 = model(thetain1) # model output at theta i
    thetaout2 = model(thetain2) # model output at theta i + 1
    thetaimpdiff = thetaout2 - thetaout1 # difference in impulse estimates at every pair of consecutive z values  
    
    
    totloss = combined_loss([zimpdiff, lam, thetaimpdiff, lam2])
    phyloss = phy_loss_mean([zimpdiff, lam, thetaimpdiff, lam2]) #for cross checking

    model.compile(loss=totloss,
                  optimizer=optimizer_val,
                  metrics=[phyloss, root_mean_squared_error])

    early_stopping = EarlyStopping(monitor='val_loss_1', patience=patience_val,verbose=1)
    
    print('Running...' + optimizer_name)
    history = model.fit(X_train, y_train,
                        batch_size=batch_size,
                        epochs=num_epochs,
                        verbose=1,
                        validation_split=val_frac, 
                        callbacks=[early_stopping, TerminateOnNaN()])
    
    test_score = model.evaluate(X_unseen, y_unseen, verbose=0)
    print('lamda: ' + str(lamda) + ' TestRMSE: ' + str(test_score[2]) + ' PhyLoss: ' + str(test_score[1]))
    
    
    #model.save(model_name)
    #spio.savemat(results_name, {'train_loss_1':history.history['loss_1'], 'val_loss_1':history.history['val_loss_1'], 'train_rmse':history.history['root_mean_squared_error'], 'val_rmse':history.history['val_root_mean_squared_error'], 'test_rmse':test_score[2]})
    # results = {'train_loss_1':history.history['loss_1'], 
    #              'val_loss_1':history.history['val_loss_1'],
    #              'train_rmse':history.history['root_mean_squared_error'], 
    #              'val_rmse':history.history['val_root_mean_squared_error'], 
    #              'test_rmse':test_score[2]}
    
    hist_df = pd.DataFrame(history.history) 
    
    
    
    
    
    
    
    fig, ax = plt.subplots(3, 3, figsize=(8,8))
    fax = ax.ravel()
    for i in range(0,9):
        exp = 200 * i
        y2 = X_scaled[np.arange(0,200,1)+exp,:]
        pred = model.predict(y2)
        pred = pred.reshape(200,1)
        theta = np.linspace(0,80,200).reshape(200,1)
        
        
        fax[i].plot(theta, y[np.arange(0,200,1)+exp]/1000, 'k', label = 'CFD')
        
        if data[exp,2] < 0.21:
            fax[i].plot(theta, JP_lowZ(data[exp,0],data[exp,2],theta), 'k--', label = 'Pannell et al. (2020)') 
        else:
            fax[i].plot(theta, JP_highZ(data[exp,0],data[exp,2],theta), 'k--', label = 'Pannell et al. (2020)') 
            
        fax[i].plot(theta, scaler_y.inverse_transform(scaler_y2.inverse_transform(pred))/1000, 'r', label = 'model')    
        #fax[i].plot(theta, scaler_y.inverse_transform(pred)/1000, 'r', label = 'model')
        
        fax[i].set_title("Z ="+str(np.round(data[exp,2],3)))
        fax[i].set_ylabel('specific impulse (MPa.ms)', fontsize='x-small')
        fax[i].set_xlabel('theta', fontsize='x-small')
        fax[i].set_xlim(0,80)
        fax[i].minorticks_on()
        fax[i].grid(which='major', ls = '-', color = [0.15, 0.15, 0.15], alpha=0.15)
        fax[i].grid(which='minor', ls=':',  dashes=(1,5,1,5), color = [0.1, 0.1, 0.1], alpha=0.25)   
    plt.tight_layout()
    handles, labels = fax[0].get_legend_handles_labels()
    fax[0].legend(handles, labels, loc='upper right', prop={'size':6})
    #fig.savefig("NN1.png")
    
    
    fig, ax = plt.subplots(3, 3, figsize=(8,8))
    fax = ax.ravel()
    
    for i in range(9,18):
        exp = 200 * i
        y2 = X_scaled[np.arange(0,200,1)+exp,:]
        pred = model.predict(y2) 
        pred = pred.reshape(200,1)
        theta = np.linspace(0,80,200).reshape(200,1)
        
        
        fax[i-9].plot(theta, y[np.arange(0,200,1)+exp]/1000, 'k', label = 'CFD')
        
        if data[exp,2] < 0.21:
            fax[i-9].plot(theta, JP_lowZ(data[exp,0],data[exp,2],theta), 'k--', label = 'Pannell et al. (2020)') 
        else:
            fax[i-9].plot(theta, JP_highZ(data[exp,0],data[exp,2],theta), 'k--', label = 'Pannell et al. (2020)') 
        fax[i-9].plot(theta, scaler_y.inverse_transform(scaler_y2.inverse_transform(pred))/1000, 'r', label = 'model')
        #fax[i-9].plot(theta, scaler_y.inverse_transform(pred)/1000, 'r', label = 'model')
        
        fax[i-9].set_title("Z ="+str(np.round(data[exp,2],3)))
        fax[i-9].set_ylabel('specific impulse (MPa.ms)', fontsize='x-small')
        fax[i-9].set_xlabel('theta', fontsize='x-small')
        fax[i-9].set_xlim(0,80)
        fax[i-9].minorticks_on()
        fax[i-9].grid(which='major', ls = '-', color = [0.15, 0.15, 0.15], alpha=0.15)
        fax[i-9].grid(which='minor', ls=':',  dashes=(1,5,1,5), color = [0.1, 0.1, 0.1], alpha=0.25)   
    plt.tight_layout()      
    #fig.savefig("NN2.png")
    
    
    # fig, ax = plt.subplots(1,1, figsize=(3,3))
    # ax.scatter(y_unseen, model.predict(X_unseen), s=7., color='black')
    # text = "$R^2 = {:.3f}$".format(r2_score(y_unseen,model.predict(X_unseen)))
    # ax.text(0.2, 0.8, text, fontsize = 'small', transform=ax.transAxes)
    # ax.set_ylabel('Predicted response')
    # ax.set_xlabel('Actual response')
    # ax.set_title('Unseen validation data')
    # ax.set_xlim(0,1)
    # ax.set_ylim(0,1)
    # ax.minorticks_on()
    # ax.grid(which='major', ls = '-', color = [0.15, 0.15, 0.15], alpha=0.15)
    # ax.grid(which='minor', ls=':',  dashes=(1,5,1,5), color = [0.1, 0.1, 0.1], alpha=0.25) 
    # plt.tight_layout()
    
    
    
    
    
   
    
    
    
    
    
    
    
    return hist_df, model


if __name__ == '__main__':
	# Main Function
	
	# List of optimizers to choose from    
    optimizer_names = ['Adagrad', 'Adadelta', 'Adam', 'Nadam', 'RMSprop', 'SGD', 'NSGD']
    optimizer_vals = [Adagrad(), Adadelta(), Adam(), Nadam(), RMSprop(), SGD(), SGD()]
    
    # selecting the optimizer
    optimizer_num = 2 #Adam
    optimizer_name = optimizer_names[optimizer_num]
    optimizer_val = optimizer_vals[optimizer_num]
    
    # Selecting Other Hyper-parameters
    drop_frac = 0.2 # Fraction of nodes to be dropped out

    n_layers = 2 # Number of hidden layers
    n_nodes = 100 # Number of nodes per hidden layer


    
    #set lamdas=0 for pgnn0
    lamda = 1 # Physics-based regularization constant - Z
    lambda2 = 0.1 #Physics-based regularization constant - theta

    results, model = PGNN_train_test(optimizer_name, optimizer_val, drop_frac, lamda, lambda2)
    
    
    

    
