import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D



#1) Training with one angle sample at a time ---------------------------------
data = pd.read_csv('cylindrical.csv', header = None)
data= data.values
# split into input (X) and output (Y) variables
X = data[:,[1,2,3]] #dropped mass as this is constant
y = data[:,4]
y = y.reshape(len(y),1)



scaler = MinMaxScaler(feature_range=(0,1))

X_scaled = scaler.fit_transform(X)
scaler_y = scaler.fit(y) 
y_scaled = scaler_y.transform(y)




# create NN model
def baseline_model():
    model = Sequential()
    model.add(Dense(18, input_dim=3, kernel_initializer='he_uniform', activation='relu'))
    model.add(Dense(18,  kernel_initializer='he_uniform', activation='relu'))
    model.add(Dense(1))
    # Compile model
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model

#Single network
# model = KerasRegressor(build_fn = baseline_model, epochs = 250)
# model.fit(X_scaled, y_scaled)



#K-Fold
#X_train, X_unseen, y_train, y_unseen = train_test_split(X_scaled, y_scaled, test_size=0, random_state=32) #Split data to 90% train & 10% unseen
X_train, y_train = X, y

kf = KFold(n_splits=28)

for train_index, test_index in kf.split(X_train, y=y_train):
    
    model = KerasRegressor(build_fn = baseline_model, epochs = 250)
    history = model.fit(X_train[train_index], 
                        y_train[train_index],
                        validation_data=(X_train[test_index],
                                        y_train[test_index]))
    num = int(test_index[0]/200)
    plt.plot(history.history['loss'], label = 'loss'+str(num))
    plt.plot(history.history['val_loss'], label = 'validation loss'+str(num))
    plt.legend()
    



def figs():
    fig, ax = plt.subplots(3, 3, figsize=(8,8))
    fax = ax.ravel()
    l_d = np.unique(X[:,-1])
    
    for i in range(len(l_d)):
        exp1 = 800 * i
        for j in range(4):
            exp = 200 * j
            y2 = X_scaled[np.arange(0,200,1)+exp+exp1,:]
            pred = model.predict(y2) 
            pred = pred.reshape(200,1)
            theta = np.linspace(0,80,200).reshape(200,1)
            
            
            fax[i].plot(theta, y[np.arange(0,200,1)+exp1+exp]/1000, 'k') #CFD           
            fax[i].plot(theta, scaler_y.inverse_transform(pred)/1000, 'r')#Model 
        
        fax[i].set_title("l/d ="+str(l_d[i]))
        fax[i].set_ylabel('specific impulse (MPa.ms)', fontsize='x-small')
        fax[i].set_xlabel('theta', fontsize='x-small')
        fax[i].set_xlim(0,80)
        fax[i].minorticks_on()
        fax[i].grid(which='major', ls = '-', color = [0.15, 0.15, 0.15], alpha=0.15)
        fax[i].grid(which='minor', ls=':',  dashes=(1,5,1,5), color = [0.1, 0.1, 0.1], alpha=0.25)   
    plt.tight_layout()

    labels = ['CFD', 'Model']
    colors = ['k', 'r']
    lws = [1, 1]
    lss = ['-', '-']
    lines = [Line2D([0], [0],  lw=lws[i], ls = lss[i], color=colors[i]) for i in range(len(labels))]
    fax[0].legend(lines,labels, loc='upper right', prop={'size':6})
    #fig.savefig("NNcylinder.png")
    
figs()   


