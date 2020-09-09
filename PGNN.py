import numpy as np
import pandas as pd
import os

from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop, Adadelta, Adagrad, Adam, Nadam, SGD
from keras.callbacks import EarlyStopping, TerminateOnNaN
from keras import backend as K
from keras.losses import mean_squared_error
from keras.regularizers import l2

from sklearn.preprocessing import MinMaxScaler, PowerTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

import matplotlib.pyplot as plt

def JP_lowZ(Z,theta):#scaled impulse
    return (0.383*(Z**-1.858)) * np.exp((-theta**2)/1829)
def JP_highZ(Z,theta):#scaled impulse
    return (0.557*(Z**-1.663)) * np.exp((-theta**2)/2007) 




#function to compute the room_mean_squared_error given the ground truth (y_true) and the predictions(y_pred)
def root_mean_squared_error(y_true, y_pred):
        return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1)) 



def phy_loss_mean(params):
	# useful for cross-checking training
    zimpdiff, lam, thetaimpdiff, lam2, thetaimpratio, lam3 = params
    #zimpdiff, lam, thetaimpdiff, lam2 = params
    def loss(y_true,y_pred):
        if lam3 != 0:
            return K.mean(K.relu(zimpdiff)) + K.mean(K.relu(thetaimpdiff)) + lam3 * K.mean(K.relu(thetaimpratio - 1.06))
        else:
            return K.mean(K.relu(zimpdiff)) + K.mean(K.relu(thetaimpdiff))  
    return loss


#function to calculate the combined loss = sum of rmse and phy based loss
def combined_loss(params):
    zimpdiff, lam, thetaimpdiff, lam2, thetaimpratio, lam3 = params
    #zimpdiff, lam, thetaimpdiff, lam2 = params
    def loss(y_true,y_pred):
        if lam3 != 0:
            return mean_squared_error(y_true, y_pred) + lam * K.mean(K.relu(zimpdiff)) + lam2 * K.mean(K.relu(thetaimpdiff)) + lam3 * K.mean(K.relu(thetaimpratio - 1.06))
        else:
            return mean_squared_error(y_true, y_pred) + lam * K.mean(K.relu(zimpdiff)) + lam2 * K.mean(K.relu(thetaimpdiff))
    return loss


def PGNN_train_test(optimizer_name, optimizer_val, drop_frac, lamda, lamda2, lamda3, scaling_input, remove_first, remove_last):
        
    # Hyper-parameters of the training process
    batch_size = 32
    num_epochs = 200
    val_frac = 0.1
    patience_val = int(0.25*num_epochs)
    
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
    
    y = data[:,4]/1000/(0.1**(1/3))
    y_og = y.reshape(len(y),1)
    y = y_og
    
    
    
    if scaling_input == 1:
        #Scaling X
        scaler = MinMaxScaler(feature_range=(0,1))
        scaler_x = scaler.fit(X)
        X_scaled = scaler_x.transform(X)
        
        #scaling y
        scaler2 = PowerTransformer()
        
        scaler_y = scaler2.fit(y)
        y_scaled = scaler_y.transform(y)
        
        #scaler_y2 = scaler.fit(y_scaled)
        #y_scaled = scaler_y2.transform(y_scaled)
    else:
        X_scaled, y_scaled = X, y
        
    X_scaled_og = X_scaled
    
    #Testing extrapolation
    if remove_last != 0:
        X_scaled, y_scaled = X_scaled[:int(-200*remove_last),:], y_scaled[:int(-200*remove_last),:]
    else:
        pass
    
    if remove_first != 0:
        X_scaled, y_scaled = X_scaled[int(200*remove_first)::,:], y_scaled[int(200*remove_first)::,:]
    else:
        pass

    #Split data to 90% train & 10% unseen
    X_train, X_unseen, y_train, y_unseen = train_test_split(X_scaled, y_scaled, test_size=0.005, random_state=32)
 
    
    # Creating the model
    model = Sequential()     
    
    for layer in np.arange(n_layers):
        if layer == 0:
            model.add(Dense(n_nodes, input_shape=(np.shape(X_scaled)[1],), activation='relu'))
            model.add(Dropout(0.2))
        else:
            model.add(Dense(n_nodes, activation='relu'))
            model.add(Dropout(drop_frac))
    model.add(Dense(1, activation='linear'))
        
    

    
    
    # Defining data for physics-based regularization, Z Condition
    zX1 =  X_scaled[0:-200,:]  # X at Z i for every pair of consecutive Z values
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
    
    lam3 = K.constant(value = lamda3)
    thetaimpratio = thetaout1 / thetaout2
    

        
    totloss = combined_loss([zimpdiff, lam, thetaimpdiff, lam2, thetaimpratio, lam3])
    phyloss = phy_loss_mean([zimpdiff, lam, thetaimpdiff, lam2, thetaimpratio, lam3]) #for cross checking

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
    
    hist_df = pd.DataFrame(history.history) 
    
    
    
    
    
    
    
    fig, ax = plt.subplots(3, 3, figsize=(8,8))
    fax = ax.ravel()
    for i in range(0,9):
        exp = 200 * i
        pred = model.predict(X_scaled_og[np.arange(0,200,1)+exp,:]).reshape(200,1)

        theta = np.linspace(0,80,200).reshape(200,1)
        
        
        fax[i].plot(theta, y_og[np.arange(0,200,1)+exp], 'k', label = 'CFD')
        
        if data[exp,2] < 0.21:
            fax[i].plot(theta, JP_lowZ(data[exp,2],theta), 'k--', label = 'Pannell et al. (2020)') 
        else:
            fax[i].plot(theta, JP_highZ(data[exp,2],theta), 'k--', label = 'Pannell et al. (2020)') 
        
        if scaling_input == 1:
            #fax[i].plot(theta, scaler_y.inverse_transform(scaler_y2.inverse_transform(pred)), 'r', label = 'model')    
            fax[i].plot(theta, scaler_y.inverse_transform(pred), 'r', label = 'model')
        else:
            fax[i].plot(theta, pred, 'r', label = 'model')
        
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
        pred = model.predict(X_scaled_og[np.arange(0,200,1)+exp,:]).reshape(200,1)
        theta = np.linspace(0,80,200).reshape(200,1)
        
        
        fax[i-9].plot(theta, y_og[np.arange(0,200,1)+exp], 'k', label = 'CFD')
        
        if data[exp,2] < 0.21:
            fax[i-9].plot(theta, JP_lowZ(data[exp,2],theta), 'k--', label = 'Pannell et al. (2020)') 
        else:
            fax[i-9].plot(theta, JP_highZ(data[exp,2],theta), 'k--', label = 'Pannell et al. (2020)') 
            
        if scaling_input == 1:
            #fax[i-9].plot(theta, scaler_y.inverse_transform(scaler_y2.inverse_transform(pred)), 'r', label = 'model')    
            fax[i-9].plot(theta, scaler_y.inverse_transform(pred), 'r', label = 'model')
        else:
            fax[i-9].plot(theta, pred, 'r', label = 'model')
        
        fax[i-9].set_title("Z ="+str(np.round(data[exp,2],3)))
        fax[i-9].set_ylabel('specific impulse (MPa.ms)', fontsize='x-small')
        fax[i-9].set_xlabel('theta', fontsize='x-small')
        fax[i-9].set_xlim(0,80)
        fax[i-9].minorticks_on()
        fax[i-9].grid(which='major', ls = '-', color = [0.15, 0.15, 0.15], alpha=0.15)
        fax[i-9].grid(which='minor', ls=':',  dashes=(1,5,1,5), color = [0.1, 0.1, 0.1], alpha=0.25)   
    plt.tight_layout()      
    #fig.savefig("NN2.png")
    
    
    
    #validation against completely unseen data
    fn_val_highZ = os.environ['USERPROFILE'] + r"\Dropbox\Papers\PaperNN_charges\datasets\spherical_val_highZ.csv"
    val_highZ = pd.read_csv(fn_val_highZ, header = None)
    val_highZ = val_highZ.values
    fn_val_lowZ = os.environ['USERPROFILE'] + r"\Dropbox\Papers\PaperNN_charges\datasets\spherical_val_lowZ.csv"
    val_lowZ = pd.read_csv(fn_val_lowZ, header = None)
    val_lowZ = val_lowZ.values
    
    # split into input (X) and output (Y) variables
    X_val_highZ = val_highZ[:,[2,3]]
    y_val_highZ = val_highZ[:,4]/1000/(250**(1/3))
    y_val_highZ = y_val_highZ.reshape(len(y_val_highZ),1)
    # split into input (X) and output (Y) variables
    X_val_lowZ = val_lowZ[:,[2,3]]
    y_val_lowZ = val_lowZ[:,4]/1000/(5**(1/3))
    y_val_lowZ = y_val_lowZ.reshape(len(y_val_lowZ),1)
    
    
    if scaling_input == 1:
        X_val_highZ = scaler_x.transform(X_val_highZ)
        X_val_lowZ = scaler_x.transform(X_val_lowZ)
    else:
        pass
    
    fig, [ax, ax1] = plt.subplots(1,2, figsize = (6,3))
    ax.plot(theta, y_val_lowZ, 'k', label = 'CFD')
    ax.plot(theta, scaler_y.inverse_transform(model.predict(X_val_lowZ).reshape(200,1)), 'r', label = 'model')   
    ax.set_title("Z = 0.17 m/kg^{1/3}")
    ax.set_ylabel('specific impulse (MPa.ms)', fontsize='x-small')
    ax.set_xlabel('theta', fontsize='x-small')
    ax.set_xlim(0,80)
    ax.minorticks_on()
    ax.grid(which='major', ls = '-', color = [0.15, 0.15, 0.15], alpha=0.15)
    ax.grid(which='minor', ls=':',  dashes=(1,5,1,5), color = [0.1, 0.1, 0.1], alpha=0.25) 
    
    ax1.plot(theta, y_val_highZ, 'k', label = 'CFD')
    ax1.plot(theta, scaler_y.inverse_transform(model.predict(X_val_highZ).reshape(200,1)), 'r', label = 'model')   
    ax1.set_title("Z = 0.40 m/kg^{1/3}")
    ax1.set_ylabel('specific impulse (MPa.ms)', fontsize='x-small')
    ax1.set_xlabel('theta', fontsize='x-small')
    ax1.set_xlim(0,80)
    ax1.minorticks_on()
    ax1.grid(which='major', ls = '-', color = [0.15, 0.15, 0.15], alpha=0.15)
    ax1.grid(which='minor', ls=':',  dashes=(1,5,1,5), color = [0.1, 0.1, 0.1], alpha=0.25) 
    plt.tight_layout()    
    
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
    
    
    #Input data
    scaling_input = 1 # 1 to scale
    
    #remove n samples?
    remove_first = 4
    remove_last = 4
    
    
    #Hyperparameters
    n_layers = 2 # Number of hidden layers
    n_nodes = 100 # Number of nodes per hidden layer
    drop_frac = 0.5 # Fraction of nodes to be dropped out
    
    #set lamdas=0 for pgnn0
    lamda = np.linspace(0,2,4) # Physics-based regularization constant - Z
    lamda2 = np.linspace(0,2,4)  #Physics-based regularization constant - theta monotonic
    lamda3 = 0 #theta smoothness

    #results, model = PGNN_train_test(optimizer_name, optimizer_val, drop_frac, lamda, lamda2, lamda3, scaling_input, remove_first, remove_last)
    all_results, all_models = [], []
    for i in range(len(lamda)):
        for j in range(len(lamda2)):
            
            results, model = PGNN_train_test(optimizer_name, optimizer_val, drop_frac, lamda[i], lamda2[j], lamda3, scaling_input, remove_first, remove_last)
            all_results.append(results)
            all_models.append(model)


# #checking theta ratios
# lol = data[:,4]
# new = []
# i = 0
# while i < len(lol)-1:
#     new.append(lol[i]/lol[i+1])
#     i += 1