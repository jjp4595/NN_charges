import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.optimizers import SGD, Adam
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler, PowerTransformer
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

def JP_lowZ(m,Z,theta):
    return (0.383*(Z**-1.858)) * np.exp((-theta**2)/1829) * (m**(1/3))
def JP_highZ(m,Z,theta):
    return (0.557*(Z**-1.663)) * np.exp((-theta**2)/2007) * (m**(1/3)) 

#1) Training with one angle sample at a time ---------------------------------
data = pd.read_csv('spherical.csv', header = None)
data= data.values
# split into input (X) and output (Y) variables
X = data[:,[0,1,3]]
y = data[:,4]
y_og = y.reshape(len(y),1)
y = y_og
# lol = X[:, 1] / (X[:,0]**(1/3)) #scaled distance
# X = np.column_stack((lol, X[:,2]))

# # difference data
# y_gauss = JP_highZ(X[:,0], (X[:, 1] / (X[:,0]**(1/3)) ), X[:,2]) *1000
# y_gauss = y_gauss.reshape(len(y_gauss),1)
# y = y_gauss - y_og



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



# create model
def baseline_model():
    model = Sequential()
    model.add(Dense(200, input_dim=3, kernel_initializer='he_uniform', activation='relu'))
    #model.add(Dropout(0.2))
    model.add(Dense(100,  kernel_initializer='he_uniform', activation='relu'))
    #model.add(Dropout(0.2))
    model.add(Dense(1, activation='linear'))
    # Compile model
    #opt = SGD(lr=0.02, momentum=0.8)
    opt = Adam(learning_rate = 0.01)
    model.compile(loss='mean_squared_error', optimizer=opt)
    return model





#Sample removal....
#n = 2
##removing last n samples --------
#X_FINAL, Y_FINAL = X_scaled[(-200*n)::,:], y_scaled[(-200*n)::,:]
#X_scaled, y_scaled = X_scaled[0:(-200*n),:], y_scaled[0:(-200*n),:]
# #removing first n samples --------
# X_FINAL, y_FINAL = X_scaled[0:(200*n),:], y_scaled[0:(200*n),:]
# X_scaled, y_scaled = X_scaled[(200*n)::,:], y_scaled[(200*n)::,:]
# #remove every other sample
# keep = np.linspace(0,17,18)[0:18:2]
# remove = np.linspace(0,17,18)[1:18:2]
# keep_inds, remove_inds = [], []
# for i in keep:
#     keep_inds.append(np.linspace(i*200, (i*200)+199, 200))
# for i in remove:
#     remove_inds.append(np.linspace(i*200, (i*200)+199, 200))
# keep_inds = np.concatenate(keep_inds)
# remove_inds = np.concatenate(remove_inds)    
# keep_inds = keep_inds.astype(int)
# remove_inds = remove_inds.astype(int)
# keep = keep.astype(int)
# remove = remove.astype(int)    
# X_FINAL, y_FINAL = X_scaled[remove_inds,:], y_scaled[remove_inds,:]
# X_scaled, y_scaled = X_scaled[keep_inds,:], y_scaled[keep_inds,:]
#--------------------

#Split data to 90% train & 10% unseen
X_train, X_unseen, y_train, y_unseen = train_test_split(X_scaled, y_scaled, test_size=0.10, random_state=32)

kf = KFold(n_splits=4, shuffle = True)
fig, ax = plt.subplots(1,1,figsize=(8,8))
fig2, ax2 = plt.subplots(1,1,figsize=(8,8))
for train_index, test_index in kf.split(X_train, y=y_train):
    
    model = KerasRegressor(build_fn = baseline_model, epochs = 100)
    history = model.fit(X_train[train_index], 
                        y_train[train_index],
                        validation_data=(X_train[test_index],
                                        y_train[test_index]))
    
    
    ax.plot(history.history['loss'], label = 'loss')
    ax.plot(history.history['val_loss'], label = 'validation loss')
    ax.set_ylabel('Loss')
    ax.set_xlabel('Epoch')
    ax.legend()
    ax.minorticks_on()
    ax.grid(which='major', ls = '-', color = [0.15, 0.15, 0.15], alpha=0.15)
    ax.grid(which='minor', ls=':',  dashes=(1,5,1,5), color = [0.1, 0.1, 0.1], alpha=0.25) 
    
    ax2.scatter(y_train[train_index], model.predict(X_train[train_index]), s=10., color='blue', label = 'Training data')
    ax2.scatter(y_train[test_index], model.predict(X_train[test_index]), s=10., color='black', label = 'In-sample validation data')
    ax2.set_ylabel('Predicted response')
    ax2.set_xlabel('Actual response')
    ax2.legend()
    ax2.set_xlim(0,1)
    ax2.set_ylim(0,1)
    ax2.minorticks_on()
    ax2.grid(which='major', ls = '-', color = [0.15, 0.15, 0.15], alpha=0.15)
    ax2.grid(which='minor', ls=':',  dashes=(1,5,1,5), color = [0.1, 0.1, 0.1], alpha=0.25) 
    
 



def figs():
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
    
    
    fig, ax = plt.subplots(1,1, figsize=(3,3))
    ax.scatter(y_unseen, model.predict(X_unseen), s=7., color='black')
    text = "$R^2 = {:.3f}$".format(r2_score(y_unseen,model.predict(X_unseen)))
    ax.text(0.2, 0.8, text, fontsize = 'small', transform=ax.transAxes)
    ax.set_ylabel('Predicted response')
    ax.set_xlabel('Actual response')
    ax.set_title('Unseen validation data')
    ax.set_xlim(0,1)
    ax.set_ylim(0,1)
    ax.minorticks_on()
    ax.grid(which='major', ls = '-', color = [0.15, 0.15, 0.15], alpha=0.15)
    ax.grid(which='minor', ls=':',  dashes=(1,5,1,5), color = [0.1, 0.1, 0.1], alpha=0.25) 
    plt.tight_layout()
    
figs()


def lastNsamplefigs(n):
    fig, ax = plt.subplots(1, 3, figsize=(8,3))
    fax = ax.ravel()
    
    j = [17 - ind for ind in range(n)]
    j.reverse()
    for i in range(len(j)):
        exp = 200 * i
        y2 = X_FINAL[np.arange(0,200,1)+exp,:]
        pred = model.predict(y2) 
        pred = pred.reshape(200,1)
        theta = np.linspace(0,80,200).reshape(200,1)
        
        
        fax[i].plot(theta, y[np.arange(0,200,1)+(200*j[i])]/1000, 'k', label = 'CFD')
        
        if data[exp,2] < 0.21:
            fax[i].plot(theta, JP_lowZ(data[(200*j[i]),0],data[(200*j[i]),2],theta), 'k--', label = 'Pannell et al. (2020)') 
        else:
            fax[i].plot(theta, JP_highZ(data[(200*j[i]),0],data[(200*j[i]),2],theta), 'k--', label = 'Pannell et al. (2020)') 
            
        fax[i].plot(theta, scaler_y.inverse_transform(scaler_y2.inverse_transform(pred))/1000, 'r', label = 'model')    
        #fax[i].plot(theta, scaler_y.inverse_transform(pred)/1000, 'r', label = 'model')
        
        fax[i].set_title("Z ="+str(np.round(data[(200*j[i]),2],3)))
        fax[i].set_ylabel('specific impulse (MPa.ms)', fontsize='x-small')
        fax[i].set_xlabel('theta', fontsize='x-small')
        fax[i].set_xlim(0,80)
        fax[i].minorticks_on()
        fax[i].grid(which='major', ls = '-', color = [0.15, 0.15, 0.15], alpha=0.15)
        fax[i].grid(which='minor', ls=':',  dashes=(1,5,1,5), color = [0.1, 0.1, 0.1], alpha=0.25)   
    plt.tight_layout()
    handles, labels = fax[0].get_legend_handles_labels()
    fax[0].legend(handles, labels, loc='upper right', prop={'size':6})
    
def firstNsamplefigs(n):
    fig, ax = plt.subplots(1, 2, figsize=(8*0.66,3))
    fax = ax.ravel()
    
    j = [0 + ind for ind in range(n)]
    j.reverse()
    for i in range(len(j)):
        exp = 200 * i
        y2 = X_FINAL[np.arange(0,200,1)+exp,:]
        pred = model.predict(y2) 
        pred = pred.reshape(200,1)
        theta = np.linspace(0,80,200).reshape(200,1)
        
        
        fax[i].plot(theta, y[np.arange(0,200,1)+(200*j[i])]/1000, 'k', label = 'CFD')
        
        if data[exp,2] < 0.21:
            fax[i].plot(theta, JP_lowZ(data[(200*j[i]),0],data[(200*j[i]),2],theta), 'k--', label = 'Pannell et al. (2020)') 
        else:
            fax[i].plot(theta, JP_highZ(data[(200*j[i]),0],data[(200*j[i]),2],theta), 'k--', label = 'Pannell et al. (2020)') 
            
        fax[i].plot(theta, scaler_y.inverse_transform(scaler_y2.inverse_transform(pred))/1000, 'r', label = 'model')    
        #fax[i].plot(theta, scaler_y.inverse_transform(pred)/1000, 'r', label = 'model')
        
        fax[i].set_title("Z ="+str(np.round(data[(200*j[i]),2],3)))
        fax[i].set_ylabel('specific impulse (MPa.ms)', fontsize='x-small')
        fax[i].set_xlabel('theta', fontsize='x-small')
        fax[i].set_xlim(0,80)
        fax[i].minorticks_on()
        fax[i].grid(which='major', ls = '-', color = [0.15, 0.15, 0.15], alpha=0.15)
        fax[i].grid(which='minor', ls=':',  dashes=(1,5,1,5), color = [0.1, 0.1, 0.1], alpha=0.25)   
    plt.tight_layout()
    handles, labels = fax[0].get_legend_handles_labels()
    fax[0].legend(handles, labels, loc='upper right', prop={'size':6})
    
def everyothersamplefigs():
    fig, ax = plt.subplots(3, 3, figsize=(8,8))
    fax = ax.ravel()
    
    j = remove
    for i in range(len(j)):
        exp = 200 * i
        y2 = X_FINAL[np.arange(0,200,1)+exp,:]
        pred = model.predict(y2) 
        pred = pred.reshape(200,1)
        theta = np.linspace(0,80,200).reshape(200,1)
        
        
        fax[i].plot(theta, y[np.arange(0,200,1)+(200*j[i])]/1000, 'k', label = 'CFD')
        
        if data[exp,2] < 0.21:
            fax[i].plot(theta, JP_lowZ(data[(200*j[i]),0],data[(200*j[i]),2],theta), 'k--', label = 'Pannell et al. (2020)') 
        else:
            fax[i].plot(theta, JP_highZ(data[(200*j[i]),0],data[(200*j[i]),2],theta), 'k--', label = 'Pannell et al. (2020)') 
            
        fax[i].plot(theta, scaler_y.inverse_transform(scaler_y2.inverse_transform(pred))/1000, 'r', label = 'model')    
        #fax[i].plot(theta, scaler_y.inverse_transform(pred)/1000, 'r', label = 'model')
        
        fax[i].set_title("Z ="+str(np.round(data[(200*j[i]),2],3)))
        fax[i].set_ylabel('specific impulse (MPa.ms)', fontsize='x-small')
        fax[i].set_xlabel('theta', fontsize='x-small')
        fax[i].set_xlim(0,80)
        fax[i].minorticks_on()
        fax[i].grid(which='major', ls = '-', color = [0.15, 0.15, 0.15], alpha=0.15)
        fax[i].grid(which='minor', ls=':',  dashes=(1,5,1,5), color = [0.1, 0.1, 0.1], alpha=0.25)   
    plt.tight_layout()
    handles, labels = fax[0].get_legend_handles_labels()
    fax[0].legend(handles, labels, loc='upper right', prop={'size':6})

