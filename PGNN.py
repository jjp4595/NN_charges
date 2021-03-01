import numpy as np
import pandas as pd
import os
import pickle

from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.wrappers.scikit_learn import KerasRegressor
from keras.optimizers import RMSprop, Adadelta, Adagrad, Adam, Nadam, SGD, Adamax
from keras.callbacks import EarlyStopping, TerminateOnNaN
from keras import backend as K
from keras.losses import mean_squared_error, mean_absolute_error
from keras.regularizers import l2


from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler, PowerTransformer
from sklearn.model_selection import train_test_split, KFold, RepeatedKFold

from sklearn.metrics import r2_score
from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import cross_val_score

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from matplotlib import colors
from matplotlib.lines import Line2D
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D

import multiprocessing

params = {'font.family':'serif',
        'axes.labelsize':'small',
        'axes.titlesize':'large',
        'xtick.labelsize':'x-small',
        'ytick.labelsize':'x-small', 
        'axes.linewidth':0.5,
        
        'xtick.major.width':0.5,
        'xtick.minor.width':0.4,
        'ytick.major.width':0.5,
        'ytick.minor.width':0.4,
        'xtick.major.size':3.0,
        'xtick.minor.size':1.5,
        'ytick.major.size':3.0,
        'ytick.minor.size':1.5,
        
        'legend.fontsize':'small',
        'legend.title_fontsize':'small',
        'legend.fancybox': False,
        'legend.framealpha': 1,
        'legend.shadow': False,
        'legend.frameon': True,
        'legend.edgecolor':'black',
        'patch.linewidth':0.5,
        
        'scatter.marker': 's',
        
        
        'grid.linewidth':'0.5',
        
        'lines.linewidth':'0.5'}
plt.rcParams.update(params)



def JP_lowZ(Z,theta):#scaled impulse
    return (0.383*(Z**-1.858)) * np.exp((-theta**2)/1829)
def JP_highZ(Z,theta):#scaled impulse
    return (0.557*(Z**-1.663)) * np.exp((-theta**2)/2007) 


def MSE(y_true, y_pred):
    return (1/len(y_pred)) * (sum(((y_true - y_pred)**2)) )
def MAE(y_true, y_pred):
    return (1/len(y_pred)) * (sum(abs(y_true - y_pred)))




#function to calculate the combined loss = sum of rmse and phy based loss
def combined_loss(params):
    zimpdiff, lam, thetaimpdiff, lam2 = params
    #zimpdiff, lam, thetaimpdiff, lam2 = params
    def loss(y_true,y_pred):
            return mean_squared_error(y_true, y_pred) + lam * K.mean(K.relu(zimpdiff)) + lam2 * K.mean(K.relu(thetaimpdiff))
    return loss


def load_data(test_frac = None, 
              remove_mean = None, remove_smallest = None, remove_largest = None,
              r_z_mean = None, r_z_smallest = None, r_z_largest = None, 
              r_theta_mean = None, r_theta_smallest = None, r_theta_largest = None):
    """
    Controls the TRAIN and TEST (unseen) data split. 
    IE this is where extrpolation & interpolation stress-test data are sorted.
    Data is scaled within the main code, not here. 
    """
    # Load features (Xc) and target values (Y) 
    filename = os.environ['USERPROFILE'] + r"\Dropbox\Papers\PaperPGNN\datasets\spherical.csv"
    data = pd.read_csv(filename, header = None)
    data= data.values
    
    # split into input (X) and output (Y) variables
    X = data[:,[2,3]]
    y = data[:,4]/1000/(0.1**(1/3))
    y_og = y.reshape(len(y),1)
    y = y_og
    X_og = X       
       
    #custom data removal for stress-testing
    if remove_mean != None:
        limit = int(len(y_og) * remove_mean * 0.5)
        y_og_sort_inds = np.argsort(y_og.reshape(len(y_og))) #sort low to high
        inds_above_mean = np.where(y_og[y_og_sort_inds] > y_og.mean())[0][0:limit]
        inds_below_mean = np.where(y_og[y_og_sort_inds] < y_og.mean())[0][-limit::]
        remove_inds = np.concatenate((inds_above_mean, inds_below_mean))        
        X_unseen, y_unseen = X[y_og_sort_inds][remove_inds], y[y_og_sort_inds][remove_inds]
        X_train, y_train = np.delete(X[y_og_sort_inds], remove_inds, 0),  np.delete(y[y_og_sort_inds], remove_inds)
        
    elif remove_smallest != None:
        inds = np.argsort(y_og.reshape(len(y_og)))
        limit = int(len(y_og) * remove_smallest)
        X_unseen, y_unseen = X[inds][0:limit], y[inds][0:limit]
        X_train, y_train = X[inds][limit::], y[inds][limit::]
        
    elif remove_largest != None:
        inds = np.flip(np.argsort(y_og.reshape(len(y_og))))
        limit = int(len(y_og) * remove_largest)
        X_unseen, y_unseen = X[inds][0:limit], y[inds][0:limit]
        X_train, y_train = X[inds][limit::], y[inds][limit::]
        
    elif r_z_mean != None:
        limit = int(len(X[:,0]) * r_z_mean * 0.5)
        og_sort_inds = np.argsort(X[:,0]) #sort low to high
        inds_above_mean = np.where(X[:,0][og_sort_inds] > X[:,0].mean())[0][0:limit]
        inds_below_mean = np.where(X[:,0][og_sort_inds] < X[:,0].mean())[0][-limit::]
        remove_inds = np.concatenate((inds_above_mean, inds_below_mean))        
        X_unseen, y_unseen = X[og_sort_inds][remove_inds], y[og_sort_inds][remove_inds]
        X_train, y_train = np.delete(X[og_sort_inds], remove_inds, 0),  np.delete(y[og_sort_inds], remove_inds)
    
    elif r_z_smallest != None:
        inds = np.argsort(X[:,0].reshape(len(y_og)))
        limit = int(len(X) * r_z_smallest)
        X_unseen, y_unseen = X[inds][0:limit], y[inds][0:limit]
        X_train, y_train = X[inds][limit::], y[inds][limit::]
        
    elif r_z_largest != None:
        inds = np.flip(np.argsort(X[:,0].reshape(len(y_og))))
        limit = int(len(X) * r_z_largest)
        X_unseen, y_unseen = X[inds][0:limit], y[inds][0:limit]
        X_train, y_train = X[inds][limit::], y[inds][limit::]
    
    elif r_theta_mean != None:
        limit = int(len(X[:,1]) * r_theta_mean * 0.5)
        og_sort_inds = np.argsort(X[:,1]) #sort low to high
        inds_above_mean = np.where(X[:,1][og_sort_inds] > X[:,1].mean())[0][0:limit]
        inds_below_mean = np.where(X[:,1][og_sort_inds] < X[:,1].mean())[0][-limit::]
        remove_inds = np.concatenate((inds_above_mean, inds_below_mean))        
        X_unseen, y_unseen = X[og_sort_inds][remove_inds], y[og_sort_inds][remove_inds]
        X_train, y_train = np.delete(X[og_sort_inds], remove_inds, 0),  np.delete(y[og_sort_inds], remove_inds)
    
    elif r_theta_smallest != None:
        inds = np.argsort(X[:,1].reshape(len(y_og)))
        limit = int(len(X) * r_theta_smallest)
        X_unseen, y_unseen = X[inds][0:limit], y[inds][0:limit]
        X_train, y_train = X[inds][limit::], y[inds][limit::]
    
    elif r_theta_largest != None:
        inds = np.flip(np.argsort(X[:,1].reshape(len(y_og))))
        limit = int(len(X) * r_theta_largest)
        X_unseen, y_unseen = X[inds][0:limit], y[inds][0:limit]
        X_train, y_train = X[inds][limit::], y[inds][limit::]
      
    else:
        #Normal data split
        X_train, X_unseen, y_train, y_unseen = train_test_split(X, y, test_size=test_frac, shuffle = True, random_state=32)
    y_unseen, y_train = y_unseen.reshape(-1,1), y_train.reshape(-1,1)
    return X_train, X_unseen, y_train, y_unseen, X_og, y_og

def load_loss(X_scaled_og, model, lamda, lamda2):
    # Defining data for physics-based regularization, Z Condition
    zin1 = K.constant(value=X_scaled_og[0:-150,:]) 
    zin2 = K.constant(value=X_scaled_og[150::,:]) 
    lam = K.constant(value=lamda) 
    zout1 = model(zin1) 
    zout2 = model(zin2) 
    zimpdiff = zout2 - zout1   
    
    # Defining data for physics-based regularization, theta Condition    
    tX =  X_scaled_og[np.argsort(X_scaled_og[:,1])] 
    tX = tX.reshape(-1,18,2) 
    ind = np.argsort(tX[:,:,0]) 
    tX = tX.reshape(-1,2)
    ind = ind.reshape(-1)
    ind += np.repeat(np.arange(0,150,1)*18,18)
    tX = tX[ind,:]
    thetain1 = K.constant(value=tX[0:-18,:])
    thetain2 = K.constant(value=tX[18::,:]) 
    lam2 = K.constant(value=lamda2)
    thetaout1 = model(thetain1)
    thetaout2 = model(thetain2)
    thetaimpdiff = thetaout2 - thetaout1     

    totloss = combined_loss([zimpdiff, lam, thetaimpdiff, lam2])
    
    return totloss



def NN_train_test(epochs, batch, nodes, opt_str, 
                  learn_rate = None, dropout = None, 
                  lamda1 = None, lamda2 = None,  
                  **kwarg):
     
    #set lamdas=0 for pgnn0
    if lamda1 == None:
        lamda = 0 
    else:
        lamda = lamda1
    
    if lamda2 == None:
        lamda2 = 0 
    else:
        lamda2 = lamda2  
    
    # Hyper-parameters of the training process
    n_layers = 1
    
    batch_size = batch
    num_epochs = epochs
    n_nodes = nodes
    
    if dropout == None:
        drop_frac = 0
    else:
        drop_frac = dropout

    patience_val = int(0.05 * num_epochs)
    scaling_input = 1
     
    
    X_train, X_unseen, y_train, y_unseen, X_og, y_og = load_data(**kwarg)
    
    # Creating the model
    model = Sequential()     
    for layer in np.arange(n_layers):
        if layer == 0:
            model.add(Dense(n_nodes, input_shape=(np.shape(X_train)[1],), activation='tanh'))
            #model.add(Dropout(drop_frac))
        else:
            model.add(Dense(n_nodes, activation='tanh'))
            #model.add(Dropout(drop_frac))
    #model.add(Dense(1, activation='linear'))    
    model.add(Dense(1))  
    
    #This does have data leakage but OK as need to know for MLC.
    X_scaled_og = MinMaxScaler(feature_range=(0,1)).fit_transform(X_og)
    totloss = load_loss(X_scaled_og, model, lamda, lamda2)
    
    if lamda1 == None and lamda2 == None:
        model.compile(loss='mean_squared_error',
                      optimizer=opt_str)
    else:
        model.compile(loss=totloss,
                      optimizer=opt_str,
                      metrics = ['mean_squared_error'])            
    
    if learn_rate != None:
        K.set_value(model.optimizer.learning_rate,learn_rate)
    else:
        pass


    kf = KFold(n_splits=5, shuffle = True)
    
    hist_df, datasets, test_scores =[], [], []
    
    if lamda1 == None and lamda2 == None:
        early_stopping = EarlyStopping(monitor='val_loss', patience=patience_val, verbose=1)
    else:
        early_stopping = EarlyStopping(monitor='val_mean_squared_error', patience=patience_val, verbose=1)
    
    for train_index, val_index in kf.split(X_train, y=y_train):
        
        #Scaling X
        scaler = MinMaxScaler(feature_range=(0,1))
        scaler_x = scaler.fit(X_train[train_index])
        X_train_scaled = scaler_x.transform(X_train[train_index])
        X_val_scaled = scaler_x.transform(X_train[val_index])
        X_test_scaled = scaler_x.transform(X_unseen)
        
        #scaling y
        scaler2 = PowerTransformer()
        scaler_y = scaler2.fit(y_train[train_index])
        y_train_scaled = scaler_y.transform(y_train[train_index])
        y_val_scaled = scaler_y.transform(y_train[val_index])
        y_test_scaled = scaler_y.transform(y_unseen)
        
        history = model.fit(X_train_scaled, 
                            y_train_scaled,
                            batch_size=batch_size,
                            epochs=num_epochs,
                            verbose = 1,
                            validation_data=(X_val_scaled,
                                             y_val_scaled),
                            validation_freq=1,
                            callbacks=[early_stopping])
        model.summary()

        train_pred = scaler_y.inverse_transform(model.predict(X_train_scaled))
        val_pred =  scaler_y.inverse_transform(model.predict(X_val_scaled))
        test_pred =  scaler_y.inverse_transform(model.predict(X_test_scaled))
        
        train_MSE = MSE(scaler_y.inverse_transform(y_train_scaled), train_pred)[0]
        train_MAE = MAE(scaler_y.inverse_transform(y_train_scaled), train_pred)[0]
        train_R2 = r2_score(scaler_y.inverse_transform(y_train_scaled), train_pred)
        
        val_MSE = MSE(scaler_y.inverse_transform(y_val_scaled), val_pred)[0]
        val_MAE = MAE(scaler_y.inverse_transform(y_val_scaled), val_pred)[0]
        val_R2 = r2_score(scaler_y.inverse_transform(y_val_scaled), val_pred)
        
        test_MSE = MSE(scaler_y.inverse_transform(y_test_scaled), test_pred)[0]
        test_MAE = MAE(scaler_y.inverse_transform(y_test_scaled), test_pred)[0]
        test_R2 = r2_score(scaler_y.inverse_transform(y_test_scaled), test_pred)
        
        score = {'train_MSE':train_MSE, 'train_MAE':train_MAE, 'train_R2':train_R2,
                  'val_MSE':val_MSE, 'val_MAE':val_MAE, 'val_R2':val_R2,
                  'test_MSE':test_MSE, 'test_MAE':test_MAE, 'test_R2':test_R2}
        test_scores.append(score)

        history = pd.DataFrame(history.history)
        hist_df.append(history)  

        datasets.append({'X_train':X_train_scaled, 'y_train':y_train_scaled,
                         'X_val':X_val_scaled, 'y_val':y_val_scaled,
                         'X_test':X_test_scaled, 'y_test_scaled':y_test_scaled,
                         'X_og':X_og, 'y_og':y_og,
                         'scaler_x':scaler_x, 'scaler_y':scaler_y})   
        
    test_score_df = pd.DataFrame(test_scores)
    return hist_df, model, datasets, test_score_df



if __name__ == '__main__':
    def dataset_transform_prepostscaler():	
        X_train, X_unseen, y_train, y_unseen, X_og, y_og = load_data(0.00001)
        X = X_og
        gX, gY = X[:,0].reshape(18,150), X[:,1].reshape(18,150)
        z = y_og.reshape(18,150)
        
        #Scaling X
        scaler = MinMaxScaler(feature_range=(0,1))
        scaler_x = scaler.fit(X_og)
        X_scaled = scaler_x.transform(X)        
        #scaling y
        scaler2 = PowerTransformer()
        scaler_y = scaler2.fit(y_og)
        y_scaled = scaler_y.transform(y_og)

        
        fig, ax = plt.subplots(1,1)
        fig.set_size_inches(3.5, 2.5)    
        ax = plt.axes(projection ='3d')
        ax.view_init(27, -36)
        CS = ax.plot_surface(gX, gY, z, vmin = 0, vmax = 25, cmap = plt.cm.magma_r)
        cbar = fig.colorbar(CS, format='%.0f', ax = ax,  
                            shrink = 0.8,
                            ticks = np.linspace(0,25,6),
                            pad = -0.05)
        ax.set_zticklabels([])
        titletext = ['Peak specific impulse '+r'$(MPa.ms$)']
        ax.set_proj_type('ortho')
        plt.tight_layout()
        
        fig, ax1 = plt.subplots(1,1)
        fig.set_size_inches(3.5, 2.5) 
        ax1 = plt.axes(projection ='3d')
        ax1.view_init(27,-36)
        CS = ax1.plot_surface(X_scaled[:,0].reshape(18,150), X_scaled[:,1].reshape(18,150),
                             y_scaled.reshape(18,150), vmin = -2.5, vmax = 2.5, cmap = plt.cm.magma_r)
        cbar = fig.colorbar(CS, format='%.1f', ax = ax1,  
                            shrink = 0.8,
                            ticks = np.linspace(-2.5,2.5,6),
                            pad = -0.05)
        ax1.set_zticklabels([])
        titletext = ['Peak specific impulse '+r'$(MPa.ms$)']        
        ax1.set_proj_type('ortho')
        plt.tight_layout()
        
    def datatransformgraphs():
        X_train, X_unseen, y_train, y_unseen, X_og, y_og = load_data(0.00001)          
        #scaling y
        scaler2 = PowerTransformer()
        scaler_y = scaler2.fit(y_og)
        y_scaled = scaler_y.transform(y_og)
        
        #Data transformation
        fig, [ax, ax1] = plt.subplots(1,2, figsize = (5,2.5), sharey = True, tight_layout = True)
        ax.hist(y_og, bins = 25, histtype = 'stepfilled', color = 'black', density = True)
        ax.set_ylabel("Count (density)")
        ax.set_xlabel("Y")
        ax.set_ylim(0,0.5)
        ax.set_xlim(0,25)
        ax.set_xticks(np.linspace(0,25,6))
        ax.set_yticks(np.linspace(0,0.5,6))
        ax.minorticks_on()
        
        ax1.hist(y_scaled, bins = 25, histtype = 'stepfilled', color = 'black', density = True)
        ax1.set_xlabel("Y (scaled)")
        ax1.minorticks_on()
        ax1.set_ylim(0,0.5)
        ax1.set_xlim(-2,2)
        ax1.set_xticks(np.linspace(-2,2,6))
        ax1.set_yticks(np.linspace(0,0.5,6))
        fig.savefig(os.environ['USERPROFILE'] + r"\Dropbox\Papers\PaperPGNN\__Paper\Fig_datatransform.pdf")
    
    def gridsearch_neurons(load = 0):
        #Grid search batch size and num epochs
        neurons = [1,2,3,4,5,6,7,8]
        if load != 0:
            score, histories = [],[]
            for i in neurons:
                try:
                    hists, model, data, test_scores = NN_train_test(10000, 32, i, 'Adadelta',  test_frac = 0.25)
                    score.append(test_scores)
                    histories.append(hists)
                except:
                    pass
            all_info = {'score':score, 'History':histories}
            save_obj(all_info, 'neurons_learnRate')
        else:
            all_info = load_obj('neurons_learnRate')
            mean_test_MSE = np.zeros((len(neurons),1))
            std_test_MSE = np.zeros((len(neurons),1))
            mean_test_MAE = np.zeros((len(neurons),1))
            std_test_MAE = np.zeros((len(neurons),1))
            mean_test_R2 = np.zeros((len(neurons),1))
            std_test_R2 = np.zeros((len(neurons),1))
            mean_train_MSE = np.zeros((len(neurons),1))
            std_train_MSE = np.zeros((len(neurons),1))
            mean_train_MAE = np.zeros((len(neurons),1))
            std_train_MAE = np.zeros((len(neurons),1))
            mean_train_R2 = np.zeros((len(neurons),1))
            std_train_R2 = np.zeros((len(neurons),1))
            mean_val_MSE = np.zeros((len(neurons),1))
            std_val_MSE = np.zeros((len(neurons),1))
            mean_val_MAE = np.zeros((len(neurons),1))
            std_val_MAE = np.zeros((len(neurons),1))
            mean_val_R2 = np.zeros((len(neurons),1))
            std_val_R2 = np.zeros((len(neurons),1))
            for i in range(len(neurons)):
                mean_test_MSE[i]  = all_info['score'][i].mean()['test_MSE']
                std_test_MSE[i] = all_info['score'][i].std()['test_MSE']
                mean_test_MAE[i]  = all_info['score'][i].mean()['test_MAE']
                std_test_MAE[i] = all_info['score'][i].std()['test_MAE']
                mean_test_R2[i]  = all_info['score'][i].mean()['test_R2']
                std_test_R2[i] = all_info['score'][i].std()['test_R2']
                mean_train_MSE[i]  = all_info['score'][i].mean()['train_MSE']
                std_train_MSE[i] = all_info['score'][i].std()['train_MSE']
                mean_train_MAE[i]  = all_info['score'][i].mean()['train_MAE']
                std_train_MAE[i] = all_info['score'][i].std()['train_MAE']
                mean_train_R2[i]  = all_info['score'][i].mean()['train_R2']
                std_train_R2[i] = all_info['score'][i].std()['train_R2']
                mean_val_MSE[i]  = all_info['score'][i].mean()['val_MSE']
                std_val_MSE[i] = all_info['score'][i].std()['val_MSE']
                mean_val_MAE[i]  = all_info['score'][i].mean()['val_MAE']
                std_val_MAE[i] = all_info['score'][i].std()['val_MAE']
                mean_val_R2[i]  = all_info['score'][i].mean()['val_R2']
                std_val_R2[i] = all_info['score'][i].std()['val_R2']
            step = 0.15
            fig, ax = plt.subplots(1,1, figsize = (2.5,2.5), tight_layout = True)
            ax.scatter(np.asarray(neurons), mean_test_MSE, c = 'blue', marker="s", edgecolor = 'k', s=10, label='Test', zorder=20)
            ax.errorbar(np.asarray(neurons), mean_test_MSE, yerr = std_test_MSE.reshape(8), fmt='none', capsize = 3, capthick = 0.5, c='blue', zorder=10)
            ax.scatter(np.asarray(neurons)+step, mean_train_MSE, c = 'red', marker="s", edgecolor = 'k', s=10, label= 'Train', zorder=20)
            ax.errorbar(np.asarray(neurons)+step, mean_train_MSE, yerr = std_train_MSE.reshape(8), fmt='none', capsize = 3, capthick = 0.5, c='red', zorder=10)
            ax.scatter(np.asarray(neurons)+(2*step), mean_val_MSE, c = 'gray', marker="s", edgecolor = 'k', s=10, label= 'Validation', zorder=20)
            ax.errorbar(np.asarray(neurons)+(2*step), mean_val_MSE, yerr = std_val_MSE.reshape(8), fmt='none', capsize = 3, capthick = 0.5, c='gray', zorder=10)                        
            handles, labels = ax.get_legend_handles_labels()
            ax.legend(handles, labels, title = 'Data type', title_fontsize = 'x-small', loc='upper right', prop={'size':6})
            ax.minorticks_on()
            ax.set_xlim(0,9)
            ax.set_ylim(0, 5)
            ax.set_xticks(np.linspace(0,9,10))
            ax.grid(which='major', ls = '-', color = [0.15, 0.15, 0.15], alpha=0.15, zorder=0)
            ax.grid(which='minor', ls=':',  dashes=(1,5,1,5), color = [0.1, 0.1, 0.1], alpha=0.25, zorder=0) 
            ax.set_xlabel('No. neurons')
            ax.set_ylabel('Mean MSE')
            fig.savefig(os.environ['USERPROFILE'] + r"\Dropbox\Papers\PaperPGNN\__Paper\Fig_neurons_NN_MSE.pdf")
            
            fig, ax = plt.subplots(1,1, figsize = (2.5,2.5), tight_layout = True)
            ax.scatter(np.asarray(neurons), mean_test_MAE, c = 'blue', marker="s", edgecolor = 'k', s=10, zorder=20)
            ax.errorbar(np.asarray(neurons), mean_test_MAE, yerr = std_test_MAE.reshape(8), fmt='none', capsize = 3, capthick = 0.5, c='blue', zorder=10)
            ax.scatter(np.asarray(neurons)+step, mean_train_MAE, c = 'red', marker="s", edgecolor = 'k', s=10, label= 'Train', zorder=20)
            ax.errorbar(np.asarray(neurons)+step, mean_train_MAE, yerr = std_train_MAE.reshape(8), fmt='none', capsize = 3, capthick = 0.5, c='red', zorder=10)
            ax.scatter(np.asarray(neurons)+(2*step), mean_val_MAE, c = 'gray', marker="s", edgecolor = 'k', s=10, label= 'Validation', zorder=20)
            ax.errorbar(np.asarray(neurons)+(2*step), mean_val_MAE, yerr = std_val_MAE.reshape(8), fmt='none', capsize = 3, capthick = 0.5, c='gray', zorder=10)                        
            ax.minorticks_on()
            ax.set_xlim(0,9)
            ax.set_ylim(0, 1)
            ax.set_xticks(np.linspace(0,9,10))
            ax.grid(which='major', ls = '-', color = [0.15, 0.15, 0.15], alpha=0.15, zorder=0)
            ax.grid(which='minor', ls=':',  dashes=(1,5,1,5), color = [0.1, 0.1, 0.1], alpha=0.25, zorder=0) 
            ax.set_xlabel('No. neurons')
            ax.set_ylabel('Mean MAE')
            fig.savefig(os.environ['USERPROFILE'] + r"\Dropbox\Papers\PaperPGNN\__Paper\Fig_neurons_NN_MAE.pdf")
            
            fig, ax = plt.subplots(1,1, figsize = (2.5,2.5), tight_layout = True)
            ax.scatter(np.asarray(neurons), mean_test_R2, c = 'blue', marker="s", edgecolor = 'k', s=10, zorder=20)
            ax.errorbar(np.asarray(neurons), mean_test_R2, yerr = std_test_R2.reshape(8), fmt='none',  capsize = 3, capthick = 0.5, c='blue', zorder=10)
            ax.scatter(np.asarray(neurons)+step, mean_train_R2, c = 'red', marker="s", edgecolor = 'k', s=10, label= 'Train', zorder=20)
            ax.errorbar(np.asarray(neurons)+step, mean_train_R2, yerr = std_train_R2.reshape(8), fmt='none', capsize = 3, capthick = 0.5, c='red', zorder=10)
            ax.scatter(np.asarray(neurons)+(2*step), mean_val_R2, c = 'gray', marker="s", edgecolor = 'k', s=10, label= 'Validation', zorder=20)
            ax.errorbar(np.asarray(neurons)+(2*step), mean_val_R2, yerr = std_val_R2.reshape(8), fmt='none', capsize = 3, capthick = 0.5, c='gray', zorder=10)                                    
            ax.minorticks_on()
            ax.set_xlim(0,9)
            ax.set_ylim(0.75, 1)
            ax.set_xticks(np.linspace(0,9,10))
            ax.grid(which='major', ls = '-', color = [0.15, 0.15, 0.15], alpha=0.15, zorder=0)
            ax.grid(which='minor', ls=':',  dashes=(1,5,1,5), color = [0.1, 0.1, 0.1], alpha=0.25, zorder=0) 
            ax.set_xlabel('No. neurons')
            ax.set_ylabel(r'Mean $R^2$')
            fig.savefig(os.environ['USERPROFILE'] + r"\Dropbox\Papers\PaperPGNN\__Paper\Fig_neurons_NN_R2.pdf")
    

    def gridsearch_lamda12(load=0):
        """
        gridsearch PGNN12 with z & theta MLC
        """

        #Grid search batch size and num epochs
        lamda = np.logspace(-2,2,5)
        if load != 0:
            score, histories, sim_detail = [],[], []
            for i in lamda:
                for j in lamda:
                    try:
                        hists, model, data, test_scores = NN_train_test(100, 32, 4, 'Adadelta', lamda1 = i, lamda2 = j, test_frac = 0.25)
                        score.append(test_scores)
                        histories.append(hists)
                        sim_detail.append((i,j))
                    except:
                        pass
            all_info = {'score':score, 'History':histories, 'Sim Detail': sim_detail}
            
            save_obj(all_info, 'PGNN_12_lamda')
        else:
            #heatmap?
            lam1, lam2 = np.meshgrid(lamda, lamda)
            
            
            score = load_obj('PGNN_12_lamda')
            test_MSE = [score['score'][i]['test_MSE'].mean() for i in range(len(score['score']))]
            test_MSE = np.asarray(test_MSE).reshape((len(lam1),len(lam2)))
            score = pd.DataFrame(data = test_MSE, index = lamda, columns = lamda)
            #columns in score are lamda 2, index is lamda 1
            
            fig, ax = plt.subplots(1,1, figsize = (3.5,3.3), tight_layout = True)
            sns.heatmap(score, annot=True, vmin = 0, vmax = 10, 
                        annot_kws = {'fontsize':'x-small'},
                        cbar_kws = {'label':'RMSE'},
                        ax = ax)
            ax.set_ylabel('$\lambda_{Phy,1}$')
            ax.set_xlabel('$\lambda_{Phy,2}$')
            fig.savefig(os.environ['USERPROFILE'] + r"\Dropbox\Papers\PaperPGNN\__Paper\Fig_lamda_PGNN_12.pdf")
                    
 

            
    def stress_test_distribution_x(file_loc_string, data_kw):
        
        X_scaled, y_scaled, X_train, X_unseen, y_train, y_unseen, scaler_x, scaler2, scaler_y, X_scaled_og, y_og =load_data(1, test_frac = 0.5)
        nbins = np.histogram_bin_edges(y_scaled, bins = 40)
        nbins_z = np.histogram_bin_edges(X_scaled[:,0], bins = 18)
        nbins_theta = np.histogram_bin_edges(X_scaled[:,1], bins = 200)
        
        fs = (4.15, 2.35)
        histylim = 200
        
        fig, [ax, ax1] = plt.subplots(1,2, figsize = fs, tight_layout = True)
        X_scaled, y_scaled, X_train, X_unseen, y_train, y_unseen, scaler_x, scaler2, scaler_y, X_scaled_og, y_og =load_data(1, **{data_kw: 0.5})
        ax.hist(y_scaled, bins = nbins, alpha = 0.5, histtype = 'stepfilled', density = False, label = 'Original')
        ax.hist(y_train, bins = nbins, histtype = 'stepfilled', density = False, color = 'black', label = 'Train')
        ax.set_ylabel("Count", fontsize='x-small')
        ax.set_xlabel(r'$Y$', fontsize='x-small')
        ax.minorticks_on()
        ax.set_ylim(0, histylim)
        ax.set_yticks(np.linspace(0,histylim,3))
        ax.set_xlim(-2,2)
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles, labels, loc='upper right', prop={'size':6}) 
        
        hist = ax1.hist2d(X_train[:,0], X_train[:,1], bins = [nbins_z, nbins_theta], cmap = plt.cm.binary)
        #cbar = plt.colorbar(hist[-1], ax = ax1)
        #cbar.ax.set_ylabel("Count")
        #cbar.set_ticks(np.linspace(0,1,3))
        
        ax1.set_xlabel(r'$X_1$', fontsize='x-small')
        ax1.set_ylabel(r'$X_2$', fontsize='x-small')
        ax1.set_xlim(0, 1)
        ax1.set_ylim(0, 1)
        ax1.set_xticks(np.linspace(0,1,3))
        ax1.set_yticks(np.linspace(0,1,3))
        fig.savefig(os.environ['USERPROFILE'] + r"\Dropbox\Papers\PaperPGNN\__Paper\Fig_stresstest_distribution"+file_loc_string+".pdf")
    # stress_test_distribution_x('_random', 'test_frac')
    # stress_test_distribution_x('_y_mean', 'remove_mean')
    # stress_test_distribution_x('_y_smallest', 'remove_smallest')
    # stress_test_distribution_x('_y_largest', 'remove_largest')
    # stress_test_distribution_x('_z_mean', 'r_z_mean')
    # stress_test_distribution_x('_z_smallest', 'r_z_smallest')
    # stress_test_distribution_x('_z_largest', 'r_z_largest')
    # stress_test_distribution_x('_theta_mean', 'r_theta_mean')
    # stress_test_distribution_x('_theta_smallest', 'r_theta_smallest')
    # stress_test_distribution_x('_theta_largest', 'r_theta_largest')  
    
    def remove_x(file_loc_string, data_kw):
        tfs = np.arange(0.1,1,0.1)
    
        #grid search svr and opt params for black box model
        X_train, X_unseen, y_train, y_unseen, X_og, y_og = load_data(**{data_kw: 0.25})
        parameterz = {'epsilon':np.logspace(-3,2, 8), 'C':np.logspace(-3,2, 8)}
        svr_rbf = SVR(kernel='rbf')
        clf = GridSearchCV(svr_rbf, parameterz, n_jobs = -1)
        opt = clf.fit(X_train, y_train.reshape(len(y_train)))
        
        #Prepare data structs
        svrModels, gbrModels, bb_scores = [], [], []
        NN_hists, PGNN_hists = [],[]
        NN_scores, PGNN_scores = [],[]
        NN_tf, PGNN_tf = [],[]
        
        for tf in tfs: 
            try:    
                #Blackbox
                X_train, X_unseen, y_train, y_unseen, X_og, y_og = load_data(**{data_kw: tf}) 
                #Scaling X
                scaler = MinMaxScaler(feature_range=(0,1))
                scaler_x = scaler.fit(X_train)
                X_train_scaled = scaler_x.transform(X_train)
                X_test_scaled = scaler_x.transform(X_unseen)
                #scaling y
                scaler2 = PowerTransformer()
                scaler_y = scaler2.fit(y_train)
                y_train_scaled = scaler_y.transform(y_train)
                y_test_scaled = scaler_y.transform(y_unseen)
                cv = RepeatedKFold(n_splits = 4, n_repeats = 1)
                
                #SVR
                svr_rbf = SVR(kernel='rbf', C = opt.best_params_['C'], epsilon = opt.best_params_['epsilon'])
                svr_n_scores = cross_val_score(svr_rbf, X_train_scaled, y_train_scaled, scoring = 'neg_mean_squared_error', cv = cv, n_jobs = -1)
                svrModel = svr_rbf.fit(X_train_scaled, y_train_scaled.reshape(len(y_train_scaled),))
                svrModels.append(svrModel)      
                svr_test_pred = svrModel.predict(X_test_scaled).reshape(len(X_test_scaled),1)
                svr_MSE = MSE(y_test_scaled, svr_test_pred)[0]
                svr_MAE = MAE(y_test_scaled, svr_test_pred)[0]          
                svr_R2 = r2_score(y_test_scaled, svr_test_pred)
                
                #GBR
                reg = GradientBoostingRegressor(n_estimators = 2000)
                reg_n_scores = cross_val_score(reg, X_train_scaled, y_train_scaled, scoring = 'neg_mean_squared_error', cv = cv, n_jobs = -1)
                gbrModel = reg.fit(X_train_scaled, y_train_scaled.reshape(len(y_train_scaled),))
                gbrModels.append(gbrModel)
                gbr_test_pred = gbrModel.predict(X_test_scaled).reshape(len(X_test_scaled),1)
                gbr_MSE = MSE(y_test_scaled, gbr_test_pred)[0]
                gbr_MAE = MAE(y_test_scaled, gbr_test_pred)[0]          
                gbr_R2 = r2_score(y_test_scaled, gbr_test_pred)
                
                score = {'svr_MSE':svr_MSE, 'svr_MAE':svr_MAE, 'svr_R2':svr_R2,
                         'gbr_MSE':gbr_MSE, 'gbr_MAE':gbr_MAE, 'gbr_R2':gbr_R2}
                bb_scores.append(score)
            except:
                pass                
               
            try:        
                #NN
                hists, model, data, test_scores = NN_train_test(10000, 32, 4, 'Adadelta',  **{data_kw: tf})
                model.save('obj/remove'+ file_loc_string + '/NN' + str(int(tf*100)) +'.h5')
                NN_hists.append(hists)
                NN_scores.append(test_scores)
                
            except:
                NN_tf.append(tf)  
                
            try:
                #PGNN12
                hists, model, data, test_scores = NN_train_test(10000, 32, 4, 'Adadelta',  lamda1 = 1, lamda2 = 1, **{data_kw: tf})
                model.save('obj/remove'+ file_loc_string + '/PGNN_12_' + str(int(tf*100)) +'.h5')
                PGNN_hists.append(hists)
                PGNN_scores.append(test_scores)
                
            except:
                PGNN_tf.append(tf)                 
                

        
        test_score_df = pd.DataFrame(bb_scores)
        to_save = {'NN_hists':NN_hists, 'NN_scores':NN_scores,
                   'PGNN_hists':PGNN_hists, 'PGNN_scores':PGNN_scores,
                   'tfs':tfs,
                   'svrModels': svrModels, 'gbrModels':gbrModels,
                   'bb_scores':test_score_df,
                   'NN_tf':NN_tf, 'PGNN_tf':PGNN_tf}
        save_obj(to_save, 'remove'+ file_loc_string + '_data')
    
        
    #To run stress test ------------------------------------------------------
    #remove_x('_random', 'test_frac') 
    #remove_x('_y_mean', 'remove_mean')
    #remove_x('_y_smallest', 'remove_smallest')
    #remove_x('_y_largest', 'remove_largest')
    #remove_x('_z_mean', 'r_z_mean')
    #remove_x('_z_smallest', 'r_z_smallest')
    #remove_x('_z_largest', 'r_z_largest')
    #remove_x('_theta_mean', 'r_theta_mean')
    #remove_x('_theta_smallest', 'r_theta_smallest')
    #remove_x('_theta_largest', 'r_theta_largest')

    def remove_x_graph(file_loc_string, data_kw, axlims):
        
        rmseLim = axlims
        
        tfs = np.arange(0.1,1,0.1)
   
        all_data = load_obj('remove_'+file_loc_string+'_data')
        
        fig, ax = plt.subplots(1,1, figsize = (2.3,2.3), tight_layout = True)
        ax.grid(which='major', ls = '-', color = [0.15, 0.15, 0.15], alpha=0.15, zorder=0)
        ax.grid(which='minor', ls=':',  dashes=(1,5,1,5), color = [0.1, 0.1, 0.1], alpha=0.25, zorder=0)
        
        ax.scatter(all_data['tfs'], all_data['bb_scores']['svr_MSE'], c = 'grey', marker="s", edgecolor = 'k', s=10, label = 'SVR', zorder=20)
        ax.scatter(all_data['tfs'], all_data['bb_scores']['gbr_MSE'], c = 'yellow', marker="D", edgecolor = 'k', s=10, label = 'GBR', zorder=20)
        
        #NN
        ax.scatter(all_data['tfs'], [all_data['NN_scores'][i]['test_MSE'].mean() for i in range(len(all_data['NN_scores']))], 
                   c = 'blue', marker="o", edgecolor = 'k', s=10, label = 'NN', zorder=20)
        ax.errorbar(all_data['tfs'], [all_data['NN_scores'][i]['test_MSE'].mean() for i in range(len(all_data['NN_scores']))],
                    yerr = [all_data['NN_scores'][i]['test_MSE'].std() for i in range(len(all_data['NN_scores']))], 
                    fmt='none', capsize = 3, capthick = 0.5, c='blue', ls='none', zorder=10)          
        
        #PGNN_12
        ax.scatter(all_data['tfs'], [all_data['PGNN_scores'][i]['test_MSE'].mean() for i in range(len(all_data['PGNN_scores']))], 
                   c = 'red', marker="d", edgecolor = 'k', s=10, label = 'PGNN', zorder=20)
        ax.errorbar(all_data['tfs'], [all_data['PGNN_scores'][i]['test_MSE'].mean() for i in range(len(all_data['PGNN_scores']))],
                    yerr = [all_data['PGNN_scores'][i]['test_MSE'].std() for i in range(len(all_data['PGNN_scores']))], 
                    fmt='none', capsize = 3, capthick = 0.5, c='red', ls='none', zorder=10) 
         
        
        ax.set_yscale('log')
        ax.set_ylabel('Test MSE', fontsize='x-small')
        ax.set_xlabel('Holdout data fraction', fontsize='x-small')
        ax.set_xlim(0,1)
        ax.set_ylim(rmseLim)
        ax.minorticks_on()
   
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles, labels, loc='lower right', prop={'size':5}) 
        fig.savefig(os.environ['USERPROFILE'] + r"\Dropbox\Papers\PaperPGNN\__Paper\Fig_remove_"+file_loc_string+".pdf")

    # remove_x_graph('y_smallest', 'remove_smallest', (10**-4, 6*10**0)) 
    # remove_x_graph('z_largest', 'r_z_largest', (10**-4, 6*10**0))
    # remove_x_graph('theta_largest', 'r_theta_largest', (10**-4, 6*10**0))
    
    # remove_x_graph('y_largest', 'remove_largest', (10**-4, 6*10**0))
    # remove_x_graph('z_smallest', 'r_z_smallest', (10**-4, 6*10**0))
    # remove_x_graph('theta_smallest', 'r_theta_smallest', (10**-4, 6*10**0))
    
    # remove_x_graph('y_mean', 'remove_mean', (10**-4, 6*10**0))       
    # remove_x_graph('z_mean', 'r_z_mean', (10**-4, 6*10**0))
    # remove_x_graph('theta_mean', 'r_theta_mean', (10**-4, 6*10**0))
    
    # remove_x_graph('random', 'test_frac', (10**-4, 6*10**0))


    def remove_x_20_80graphs(file_loc_string, data_kw, axlims):
        scatter20Lim, scatter80Lim = axlims
        
        #unseen evaluation
        import tensorflow as tf
        NN_20 = tf.keras.models.load_model('obj/remove'+file_loc_string+'/NN20.h5', compile = False)
        PGNN12_20 = tf.keras.models.load_model('obj/remove'+file_loc_string+'/PGNN_12_20.h5', custom_objects={'loss':combined_loss}, compile = False)
        X_scaled, y_scaled_20, X_train, X_unseen_20, y_train, y_unseen_20, scaler_x, scaler2, scaler_y, X_scaled_og, y_og =load_data(1, **{data_kw: 0.2})
        NN_80 = tf.keras.models.load_model('obj/remove'+file_loc_string+'/NN80.h5', compile = False)
        PGNN12_80 = tf.keras.models.load_model('obj/remove'+file_loc_string+'/PGNN_12_80.h5', custom_objects={'loss':combined_loss}, compile = False)
        X_scaled, y_scaled_80, X_train, X_unseen_80, y_train, y_unseen_80, scaler_x, scaler2, scaler_y, X_scaled_og, y_og =load_data(1, **{data_kw: 0.8})
        
        fig_scatter20, ax = plt.subplots(1,1, figsize=(2.5,2.5), tight_layout = True)
        alp = 0.5
        ax.scatter(y_unseen_20, NN_20.predict(X_unseen_20), alpha = alp, c = 'blue', marker="o", s=5, label = 'NN', zorder=20)
        ax.scatter(y_unseen_20, PGNN12_20.predict(X_unseen_20), alpha = alp, c = 'red', marker="d",  s=5,label = 'PGNN-12', zorder=20)
        text_NN = "NN $= {:.3f}$".format(r2_score(y_unseen_20,NN_20.predict(X_unseen_20) ) )
        text_PGNN12 = "PGNN $= {:.3f}$".format(r2_score(y_unseen_20,PGNN12_20.predict(X_unseen_20) ) )
        ax.text(0.05, 0.9, "$R^2$", fontsize = 'xx-small', transform=ax.transAxes)
        ax.text(0.05, 0.8, text_NN, fontsize = 'xx-small', transform=ax.transAxes)
        ax.text(0.05, 0.7, text_PGNN12, fontsize = 'xx-small', transform=ax.transAxes)
        ax.set_ylabel('Predicted response')
        ax.set_xlabel('Actual response')
        ax.set_ylim(scatter20Lim)
        ax.set_xlim(scatter20Lim)
        ax.minorticks_on()
        ax.grid(which='major', ls = '-', color = [0.15, 0.15, 0.15], alpha=0.15)
        ax.grid(which='minor', ls=':',  dashes=(1,5,1,5), color = [0.1, 0.1, 0.1], alpha=0.25) 
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles, labels, loc='lower right', prop={'size':6}) 
        fig_scatter20.savefig(os.environ['USERPROFILE'] + r"\Dropbox\Papers\PaperPGNN\__Paper\Fig_remove_"+file_loc_string+"_20.pdf")
        
        fig_scatter80, ax = plt.subplots(1,1, figsize=(2.5,2.5), tight_layout = True)
        ax.scatter(y_unseen_80, NN_80.predict(X_unseen_80), alpha = alp, c = 'blue', marker="o", s=5, label = 'NN', zorder=20)
        ax.scatter(y_unseen_80, PGNN12_80.predict(X_unseen_80), alpha = alp, c = 'red', marker="d",  s=5,label = 'PGNN-12', zorder=20)
        text_NN = "NN $= {:.3f}$".format(r2_score(y_unseen_80,NN_80.predict(X_unseen_80) ) )
        text_PGNN2 = "PGNN $= {:.3f}$".format(r2_score(y_unseen_80,PGNN12_80.predict(X_unseen_80) ) )
        ax.text(0.05, 0.9, "$R^2$", fontsize = 'xx-small', transform=ax.transAxes)
        ax.text(0.05, 0.8, text_NN, fontsize = 'xx-small', transform=ax.transAxes)
        ax.text(0.05, 0.7, text_PGNN12, fontsize = 'xx-small', transform=ax.transAxes)
        ax.set_ylabel('Predicted response')
        ax.set_xlabel('Actual response')
        ax.set_ylim(scatter80Lim)
        ax.set_xlim(scatter80Lim)
        ax.minorticks_on()
        ax.grid(which='major', ls = '-', color = [0.15, 0.15, 0.15], alpha=0.15)
        ax.grid(which='minor', ls=':',  dashes=(1,5,1,5), color = [0.1, 0.1, 0.1], alpha=0.25) 
        fig_scatter80.savefig(os.environ['USERPROFILE'] + r"\Dropbox\Papers\PaperPGNN\__Paper\Fig_remove_"+file_loc_string+"_80.pdf")
        
    def finalboxplot(dh):
        extrap_strings = ['y_smallest', 'z_largest', 'theta_largest',
        				  'y_largest', 'z_smallest', 'theta_smallest']		  
        interp_strings = ['y_mean', 'z_mean', 'theta_mean']				  
        
				
        
        NNs_extr, PGNNs_extr, svrs_extr, gbrs_extr = [],[],[],[]
        for i in extrap_strings:
        					
            all_data = load_obj('remove_'+i+'_data')
            #All scores up to threshold
            NNs_extr.append(np.asarray(all_data['NNtest_scores'])[0:dh,:,0].flatten())
            PGNNs_extr.append(np.asarray(all_data['PGNN_12_test_scores'])[0:dh,:,0].flatten())
            svrs_extr.append(np.asarray(all_data['svr_test_rmse'])[0:dh].flatten())
            gbrs_extr.append(np.asarray(all_data['reg_test_rmse'])[0:dh].flatten())
            
            # #All scores above threshold
            # NNs_extr.append(np.asarray(all_data['NNtest_scores'])[dh::,:,0].flatten())
            # PGNNs_extr.append(np.asarray(all_data['PGNN_12_test_scores'])[dh::,:,0].flatten())
            # svrs_extr.append(np.asarray(all_data['svr_test_rmse'])[dh::].flatten())
            # gbrs_extr.append(np.asarray(all_data['reg_test_rmse'])[dh::].flatten())            
            
        NNs_extr   = np.asarray(NNs_extr).flatten()
        PGNNs_extr = np.asarray(PGNNs_extr).flatten()
        svrs_extr  = np.asarray(svrs_extr).flatten()
        gbrs_extr  = np.asarray(gbrs_extr).flatten()
        
        # #Taking logs
        # NNs_extr = np.log10(NNs_extr)
        # PGNNs_extr = np.log10(PGNNs_extr)
        # svrs_extr = np.log10(svrs_extr)
        # gbrs_extr = np.log10(gbrs_extr)
        
        NNs_intr, PGNNs_intr, svrs_intr, gbrs_intr = [],[],[],[]
        for i in interp_strings:
        					
        	all_data = load_obj('remove_'+i+'_data')
            #All scores
        	NNs_intr.append(np.asarray(all_data['NNtest_scores'])[0:dh,:,0].flatten())
        	PGNNs_intr.append(np.asarray(all_data['PGNN_12_test_scores'])[0:dh,:,0].flatten())
            #Min scores
        	# NNs_intr.append(np.asarray(all_data['NNtest_scores'])[0:dh,:,0].min(1).flatten())
        	# PGNNs_intr.append(np.asarray(all_data['PGNN_12_test_scores'])[0:dh,:,0].min(1).flatten())            
        	svrs_intr.append(np.asarray(all_data['svr_test_rmse'])[0:dh].flatten())
        	gbrs_intr.append(np.asarray(all_data['reg_test_rmse'])[0:dh].flatten())
        NNs_intr   = np.asarray(NNs_intr).flatten()
        PGNNs_intr = np.asarray(PGNNs_intr).flatten()
        svrs_intr  = np.asarray(svrs_intr).flatten()
        gbrs_intr  = np.asarray(gbrs_intr).flatten()
        
        # #Taking logs
        # NNs_intr = np.log10(NNs_intr)
        # PGNNs_intr = np.log10(PGNNs_intr)
        # svrs_intr = np.log10(svrs_intr)
        # gbrs_intr = np.log10(gbrs_intr)        
        
        
        
        # NNs_overall   = np.concatenate([NNs_intr, NNs_extr])
        # PGNNs_overall = np.concatenate([NNs_intr, PGNNs_extr])
        # svrs_overall  = np.concatenate([svrs_intr, svrs_extr])
        # gbrs_overall  = np.concatenate([gbrs_intr, gbrs_extr])
        
        
        #Create dataframe
        networks = ['NN', 'PGNN', 'SVR', 'GBR']
        assessments = ['Extrapolate', 'Interpolate']
        data = [NNs_extr,   NNs_intr, 
                PGNNs_extr, PGNNs_intr,
                svrs_extr,  svrs_intr,
                gbrs_extr,  gbrs_intr]
        
        df = pd.DataFrame(columns = ['Score', 'Network', 'Assessment'])
        z = 0
        for i in networks:
            for j in assessments:
                    d = {'Score': data[z], "Network": i, "Assessment":j}
                    new = pd.DataFrame(data = d)      
                    df = pd.concat([df, new])
                    z +=1
        df = df.dropna()
        
        
        g = sns.catplot(x='Assessment', y='Score', hue='Network', data = df, kind = 'box', 
                        height = 3, aspect=1,
                        palette =['blue','red', 'grey', 'yellow'], legend=True,
                        saturation=0.5, fliersize=0.5)
        g.set(yscale="log", ylabel = 'RMSE')
        #g.set(ylabel = 'log(RMSE)')
        g.despine(right=False, top=False)
        #g.savefig("__Paper\Fig_pubgraph.pdf")
        
        #graphs below dh threshold
        #g.savefig("potential_thesis_figures\Fig_aggregated_logRMSE_"+str(dh*10)+".pdf")
        #g.savefig("potential_thesis_figures\Fig_aggregated_RMSE_"+str(dh*10)+".pdf")
        #graphs above dh threshold
        #g.savefig("potential_thesis_figures\Fig_aggregated_RMSE_above"+str(dh*10)+".pdf")
        #g.savefig("potential_thesis_figures\Fig_aggregated_logRMSE_above"+str(dh*10)+".pdf")
        return df
    
    def pubtabledata():
        mean_extr = np.zeros((9,4))
        mean_intr = np.zeros((9,4))
        med_extr = np.zeros((9,4))
        med_intr = np.zeros((9,4))
        
        dfs = []
        for i in range(1,10):
            dfs.append(finalboxplot(i))
            
        for i in range(len(mean_extr)):      
            mean_extr[i,0] = dfs[i][(dfs[i]['Network'] == 'NN') & (dfs[i]['Assessment'] == 'Extrapolate')].mean()
            mean_extr[i,1] = dfs[i][(dfs[i]['Network'] == 'PGNN') & (dfs[i]['Assessment'] == 'Extrapolate')].mean() 
            mean_extr[i,2] = dfs[i][(dfs[i]['Network'] == 'SVR') & (dfs[i]['Assessment'] == 'Extrapolate')].mean() 
            mean_extr[i,3] = dfs[i][(dfs[i]['Network'] == 'GBR') & (dfs[i]['Assessment'] == 'Extrapolate')].mean() 
            
            mean_intr[i,0] = dfs[i][(dfs[i]['Network'] == 'NN') & (dfs[i]['Assessment'] == 'Interpolate')].mean() 
            mean_intr[i,1] = dfs[i][(dfs[i]['Network'] == 'PGNN') & (dfs[i]['Assessment'] == 'Interpolate')].mean()       
            mean_intr[i,2] = dfs[i][(dfs[i]['Network'] == 'SVR') & (dfs[i]['Assessment'] == 'Interpolate')].mean() 
            mean_intr[i,3] = dfs[i][(dfs[i]['Network'] == 'GBR') & (dfs[i]['Assessment'] == 'Interpolate')].mean()  
            
            med_extr[i,0] = dfs[i][(dfs[i]['Network'] == 'NN') & (dfs[i]['Assessment'] == 'Extrapolate')].median()
            med_extr[i,1] = dfs[i][(dfs[i]['Network'] == 'PGNN') & (dfs[i]['Assessment'] == 'Extrapolate')].median()
            med_extr[i,2] = dfs[i][(dfs[i]['Network'] == 'SVR') & (dfs[i]['Assessment'] == 'Extrapolate')].median()
            med_extr[i,3] = dfs[i][(dfs[i]['Network'] == 'GBR') & (dfs[i]['Assessment'] == 'Extrapolate')].median()
            
            med_intr[i,0] = dfs[i][(dfs[i]['Network'] == 'NN') & (dfs[i]['Assessment'] == 'Interpolate')].median() 
            med_intr[i,1] = dfs[i][(dfs[i]['Network'] == 'PGNN') & (dfs[i]['Assessment'] == 'Interpolate')].median() 
            med_intr[i,2] = dfs[i][(dfs[i]['Network'] == 'SVR') & (dfs[i]['Assessment'] == 'Interpolate')].median() 
            med_intr[i,3] = dfs[i][(dfs[i]['Network'] == 'GBR') & (dfs[i]['Assessment'] == 'Interpolate')].median() 
            
        mean_extr_premium = mean_extr[:,0] - mean_extr[:,1]
        mean_intr_premium = mean_intr[:,0] - mean_intr[:,1]
        med_extr_premium = med_extr[:,0] - med_extr[:,1]
        med_intr_premium = med_intr[:,0] - med_intr[:,1]   
        
        
        
        #for table if x10**2
        # mean_extr_tab = np.round(mean_extr * 100, 2)
        # mean_intr_tab = np.round(mean_intr * 100, 2)
        # med_extr_tab = np.round(med_extr * 100, 2)
        # med_intr_tab = np.round(med_intr * 100, 2)
        #for table if notx10**2
        mean_extr_tab = mean_extr
        mean_intr_tab = mean_intr
        med_extr_tab = med_extr
        med_intr_tab = med_intr

        
        fig, [ax, ax1] = plt.subplots(1,2, figsize = (5, 2.5), tight_layout = True)
        df = [10,20,30,40,50,60,70,80,90]
        ax.plot(df, mean_extr_tab[:,0], c = 'blue', label = 'NN - extr.')
        ax.plot(df, mean_extr_tab[:,1], c = 'red', label = 'PGNN - extr.')
        ax.plot(df, mean_intr_tab[:,0], ls = '--', c = 'blue', label = 'NN - inter.')
        ax.plot(df, mean_intr_tab[:,1], ls = '--', c = 'red', label = 'PGNN - inter.')
        ax.set_ylabel('Mean aggregate RMSE')
        ax.set_xlabel('Data holdout threshold (%)')
        #ax.set_ylim((0,60))
        ax.set_yscale('log')
        ax.set_ylim((10**-3,10**0))
        ax.set_xlim((0,100))
        ax.minorticks_on()
        ax.grid(which='major', ls = '-', color = [0.15, 0.15, 0.15], alpha=0.15)
        ax.grid(which='minor', ls=':',  dashes=(1,5,1,5), color = [0.1, 0.1, 0.1], alpha=0.25) 
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles, labels, loc='lower right', prop={'size':5}) 
        
        ax1.plot(df, med_extr_tab[:,0], c = 'blue', label = 'NN - extr.')
        ax1.plot(df, med_extr_tab[:,1], c = 'red', label = 'PGNN - extr.')
        ax1.plot(df, med_intr_tab[:,0], ls = '--', c = 'blue', label = 'NN - inter.')
        ax1.plot(df, med_intr_tab[:,1], ls = '--', c = 'red', label = 'PGNN - inter.')
        ax1.set_ylabel('Median aggregate RMSE')
        ax1.set_xlabel('Data holdout threshold (%)')
        #ax1.set_ylim((0,60))
        ax1.set_yscale('log')
        ax1.set_ylim((10**-3,10**0))
        ax1.set_xlim((0,100))
        ax1.minorticks_on()
        ax1.grid(which='major', ls = '-', color = [0.15, 0.15, 0.15], alpha=0.15)
        ax1.grid(which='minor', ls=':',  dashes=(1,5,1,5), color = [0.1, 0.1, 0.1], alpha=0.25) 
        fig.savefig(os.environ['USERPROFILE'] + r"\Dropbox\Papers\PaperPGNN\__Paper\Fig_aggregateRMSEsummary.pdf")


#Pickling functions
def save_obj(obj, name ):
    with open('obj/'+ name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
#save_obj(lsoa_dist, "lsoa_data")
def load_obj(name ):
    with open('obj/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)



# for i in range(18):
#     fig, [ax0, ax1] = plt.subplots(1,2, tight_layout = True, figsize = (6, 2.5))
#     ax0.plot(X[0+i*200:200+i*200,1], y[0+i*200:200+i*200])
#     ax0.set_title("Z = " + str(round(X[i*200,0], 3)))
#     ax0.set_ylabel('y unscaled')
#     ax0.set_xlabel('X unscaled')
#     ax1.plot(X_scaled_og[0+i*200:200+i*200,1], y_scaled[0+i*200:200+i*200])
#     ax1.set_ylabel('y scaled')
#     ax1.set_xlabel('x scaled')