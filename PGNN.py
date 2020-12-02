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

# def mse_loss_without_phys(params):
#     """
#     Additional metric that just returns MSE. Use when the physics constraint is activated so that test RMSE results are comparable.
#     """
#     #zimpdiff, lam, thetaimpdiff, lam2, thetaimpratio, lam3 = params
#     def loss(y_true,y_pred):
#             return mean_squared_error(y_true, y_pred)
#     return loss



def load_data(scaling_input, test_frac = None, 
              remove_mean = None, remove_smallest = None, remove_largest = None,
              r_z_mean = None, r_z_smallest = None, r_z_largest = None, 
              r_theta_mean = None, r_theta_smallest = None, r_theta_largest = None):
    # Load features (Xc) and target values (Y) 
    filename = os.environ['USERPROFILE'] + r"\Dropbox\Papers\PaperPGNN\datasets\spherical.csv"
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
        
        # scaler_y2 = scaler.fit(y_scaled)
        # y_scaled = scaler_y2.transform(y_scaled)
    else:
        X_scaled, y_scaled = X, y
        
    X_scaled_og = X_scaled
   

    if remove_mean != None:
        limit = int(len(y_og) * remove_mean * 0.5)
        y_og_sort_inds = np.argsort(y_og.reshape(3600)) #sort low to high
        inds_above_mean = np.where(y_og[y_og_sort_inds] > y_og.mean())[0][0:limit]
        inds_below_mean = np.where(y_og[y_og_sort_inds] < y_og.mean())[0][-limit::]
        remove_inds = np.concatenate((inds_above_mean, inds_below_mean))        
        X_unseen, y_unseen = X_scaled[y_og_sort_inds][remove_inds], y_scaled[y_og_sort_inds][remove_inds]
        X_train, y_train = np.delete(X_scaled[y_og_sort_inds], remove_inds, 0),  np.delete(y_scaled[y_og_sort_inds], remove_inds)
        
    elif remove_smallest != None:
        inds = np.argsort(y_og.reshape(3600))
        limit = int(len(y_og) * remove_smallest)
        X_unseen, y_unseen = X_scaled[inds][0:limit], y_scaled[inds][0:limit]
        X_train, y_train = X_scaled[inds][limit::], y_scaled[inds][limit::]
        
    elif remove_largest != None:
        inds = np.flip(np.argsort(y_og.reshape(3600)))
        limit = int(len(y_og) * remove_largest)
        X_unseen, y_unseen = X_scaled[inds][0:limit], y_scaled[inds][0:limit]
        X_train, y_train = X_scaled[inds][limit::], y_scaled[inds][limit::]
        
    elif r_z_mean != None:
        limit = int(len(X_scaled_og[:,0]) * r_z_mean * 0.5)
        og_sort_inds = np.argsort(X_scaled_og[:,0]) #sort low to high
        inds_above_mean = np.where(X_scaled_og[:,0][og_sort_inds] > X_scaled_og[:,0].mean())[0][0:limit]
        inds_below_mean = np.where(X_scaled_og[:,0][og_sort_inds] < X_scaled_og[:,0].mean())[0][-limit::]
        remove_inds = np.concatenate((inds_above_mean, inds_below_mean))        
        X_unseen, y_unseen = X_scaled[og_sort_inds][remove_inds], y_scaled[og_sort_inds][remove_inds]
        X_train, y_train = np.delete(X_scaled[og_sort_inds], remove_inds, 0),  np.delete(y_scaled[og_sort_inds], remove_inds)
    
    elif r_z_smallest != None:
        inds = np.argsort(X_scaled_og[:,0].reshape(3600))
        limit = int(len(X_scaled_og) * r_z_smallest)
        X_unseen, y_unseen = X_scaled[inds][0:limit], y_scaled[inds][0:limit]
        X_train, y_train = X_scaled[inds][limit::], y_scaled[inds][limit::]
        
    elif r_z_largest != None:
        inds = np.flip(np.argsort(X_scaled_og[:,0].reshape(3600)))
        limit = int(len(X_scaled_og) * r_z_largest)
        X_unseen, y_unseen = X_scaled[inds][0:limit], y_scaled[inds][0:limit]
        X_train, y_train = X_scaled[inds][limit::], y_scaled[inds][limit::]
    
    elif r_theta_mean != None:
        limit = int(len(X_scaled_og[:,1]) * r_theta_mean * 0.5)
        og_sort_inds = np.argsort(X_scaled_og[:,1]) #sort low to high
        inds_above_mean = np.where(X_scaled_og[:,1][og_sort_inds] > X_scaled_og[:,1].mean())[0][0:limit]
        inds_below_mean = np.where(X_scaled_og[:,1][og_sort_inds] < X_scaled_og[:,1].mean())[0][-limit::]
        remove_inds = np.concatenate((inds_above_mean, inds_below_mean))        
        X_unseen, y_unseen = X_scaled[og_sort_inds][remove_inds], y_scaled[og_sort_inds][remove_inds]
        X_train, y_train = np.delete(X_scaled[og_sort_inds], remove_inds, 0),  np.delete(y_scaled[og_sort_inds], remove_inds)
    
    elif r_theta_smallest != None:
        inds = np.argsort(X_scaled_og[:,1].reshape(3600))
        limit = int(len(X_scaled_og) * r_theta_smallest)
        X_unseen, y_unseen = X_scaled[inds][0:limit], y_scaled[inds][0:limit]
        X_train, y_train = X_scaled[inds][limit::], y_scaled[inds][limit::]
    
    elif r_theta_largest != None:
        inds = np.flip(np.argsort(X_scaled_og[:,1].reshape(3600)))
        limit = int(len(X_scaled_og) * r_theta_largest)
        X_unseen, y_unseen = X_scaled[inds][0:limit], y_scaled[inds][0:limit]
        X_train, y_train = X_scaled[inds][limit::], y_scaled[inds][limit::]
      
    else:
        #Split data 
        X_train, X_unseen, y_train, y_unseen = train_test_split(X_scaled, y_scaled, test_size=test_frac, shuffle = True, random_state=32)
    
    return X_scaled, y_scaled, X_train, X_unseen, y_train, y_unseen, scaler_x, scaler2, scaler_y, X_scaled_og, y_og

def load_loss(X_scaled_og, model, lamda, lamda2, lamda3):
    # Defining data for physics-based regularization, Z Condition
    zX1 =  X_scaled_og[0:-200,:]  # X at Z i for every pair of consecutive Z values
    zX2 =  X_scaled_og[200::,:]# X at Z i + 1 for every pair of consecutive Z values
    zin1 = K.constant(value=zX1) # input at Z i
    zin2 = K.constant(value=zX2) # input at Z i + 1
    lam = K.constant(value=lamda) # regularization hyper-parameter
    zout1 = model(zin1) # model output at Z i
    zout2 = model(zin2) # model output at Z i + 1
    zimpdiff = zout2 - zout1 # difference in impulse estimates at every pair of consecutive z values    
    
    # Defining data for physics-based regularization, Z Condition
    thetaX1 =  X_scaled_og[np.argsort(X_scaled_og[:,1])][0:-18,:]  # X at theta i for every pair of consecutive theta values
    thetaX2 =  X_scaled_og[np.argsort(X_scaled_og[:,1])][18::,:]   # X at theta i + 1 for every pair of consecutive theta values
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
    #non_phy_loss = mse_loss_without_phys([zimpdiff, lam, thetaimpdiff, lam2, thetaimpratio, lam3])
    
    return totloss



def NN_train_test(epochs, batch, opt_str, learn_rate = None, dropout = None, 
                  lamda1 = None, lamda2 = None,  
                  **kwarg):
     
    #set lamdas=0 for pgnn0
    if lamda1 == None:
        lamda = 0 # Physics-based regularization constant - Z
    else:
        lamda = lamda1
    
    if lamda2 == None:
        lamda2 = 0 
    else:
        lamda2 = lamda2  #Physics-based regularization constant - theta monotonic
     
    lamda3 = 0 #theta smoothness    
    
    # Hyper-parameters of the training process
    n_layers = 1
    
    batch_size = batch
    num_epochs = epochs
    
    if dropout == None:
        drop_frac = 0
    else:
        drop_frac = dropout
    n_nodes = 200
    
    patience_val = int(0.3 * num_epochs)
    scaling_input = 1
     
    
    X_scaled, y_scaled, X_train, X_unseen, y_train, y_unseen, scaler_x, scaler2, scaler_y, X_scaled_og, y_og = load_data(scaling_input, **kwarg)
    
    # Creating the model
    model = Sequential()     
    for layer in np.arange(n_layers):
        if layer == 0:
            model.add(Dense(n_nodes, input_shape=(np.shape(X_scaled)[1],), activation='relu'))
            model.add(Dropout(drop_frac))
        else:
            model.add(Dense(n_nodes, activation='relu'))
            model.add(Dropout(drop_frac))
    model.add(Dense(1, activation='linear'))    
    
    totloss = load_loss(X_scaled_og, model, lamda, lamda2, lamda3)
    
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


    kf = KFold(n_splits=4, shuffle = True)
    hist_df, test_scores, datasets = [], [], []
    
    if lamda1 == None and lamda2 == None:
        early_stopping = EarlyStopping(monitor='val_loss', patience=patience_val, verbose=1)
    else:
        early_stopping = EarlyStopping(monitor='val_mean_squared_error', patience=patience_val, verbose=1)
    
    for train_index, test_index in kf.split(X_train, y=y_train):
        
        
        history = model.fit(X_train[train_index], 
                            y_train[train_index],
                            batch_size=batch_size,
                            epochs=num_epochs,
                            verbose = 1,
                            validation_data=(X_train[test_index],
                                            y_train[test_index]),
                            validation_freq=1,
                            callbacks=[early_stopping])
        
        test_score = model.evaluate(X_unseen, y_unseen, verbose=0)
        test_scores.append(test_score)
                
        history = pd.DataFrame(history.history)
        hist_df.append(history)  

        datasets.append({'X_Scaled':X_scaled, 'y_scaled':y_scaled, 'X_train':X_train, 
                'X_unseen':X_unseen, 'y_train':y_train, 'y_unseen':y_unseen, 
                'scaler_x':scaler_x, 'scaler2':scaler2, 'scaler_y':scaler_y, 
                'X_scaled_og':X_scaled_og, 'y_og':y_og})   
        
    
    return hist_df, model, datasets, test_scores




if __name__ == '__main__':	
    def datatransformgraphs():
        X_scaled, y_scaled, X_train, X_unseen, y_train, y_unseen, scaler_x, scaler2, scaler_y, X_scaled_og, y_og = load_data(1, 0, 0, 0.01)
        #Data transformation
        fig, [ax, ax1] = plt.subplots(1,2, figsize = (3.3,1.8), sharey = True, tight_layout = True)
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
    
    
    
    def gridsearch_epoch_bs(load = 0):
        
        #Grid search batch size and num epochs
        epochs = [10, 50, 100]
        batch_size = [10, 20, 40, 60, 80, 100]
        if load != 0:
            score_RMSE, histories = [],[]
            for i in epochs:
                for j in batch_size:
                    hists, model, data, test_scores = NN_train_test(i, j, 'Adam')
                    score_RMSE.append(test_scores)
                    histories.append(hists)
            all_info = {'RMSE':score_RMSE, 'History':histories}
            save_obj(all_info, 'Epoch_batch_Score_RMSE')
        else:
            all_info = load_obj('Epoch_batch_Score_RMSE')
            RMSE = all_info['RMSE']
            RMSE_10, RMSE_50, RMSE_100 = np.stack(RMSE[0:6]).T, np.stack(RMSE[6:12]).T, np.stack(RMSE[12::]).T
            RMSE_10, RMSE_50, RMSE_100 = RMSE_10**0.5, RMSE_50**0.5, RMSE_100**0.5
        
        step = 0.5
        fig, ax = plt.subplots(1,1, figsize = (2.5,2.5), tight_layout = True)
        ax.scatter(np.asarray(batch_size)-step, RMSE_10.mean(0), c = 'blue', marker="s", edgecolor = 'k', s=10, label = '10', zorder=20)
        err = np.stack((RMSE_10.min(0).reshape(1, 6), RMSE_10.max(0).reshape(1, 6)), axis = 1).reshape(2,6)
        err = abs(err - RMSE_10.mean(0))
        ax.errorbar(np.asarray(batch_size)-step, RMSE_10.mean(0), yerr = err, capsize = 3, capthick = 0.5, c='blue', zorder=10)
        
        
        ax.scatter(np.asarray(batch_size)-step, RMSE_50.mean(0), c = 'red', marker="D", edgecolor = 'k', s=10,  label = '50', zorder=20)
        err = np.stack((RMSE_50.min(0).reshape(1, 6), RMSE_50.max(0).reshape(1, 6)), axis = 1).reshape(2,6)
        err = abs(err - RMSE_50.mean(0))
        ax.errorbar(np.asarray(batch_size)-step, RMSE_50.mean(0), yerr = err, capsize = 3, capthick = 0.5, c='red', zorder=10)
        
        ax.scatter(np.asarray(batch_size)-step, RMSE_100.mean(0), c = 'gray', marker="o", edgecolor = 'k', s=10,  label = '100', zorder=20)
        err = np.stack((RMSE_100.min(0).reshape(1, 6), RMSE_100.max(0).reshape(1, 6)), axis = 1).reshape(2,6)
        err = abs(err - RMSE_100.mean(0))
        ax.errorbar(np.asarray(batch_size)-step, RMSE_100.mean(0), yerr = err, capsize = 3,capthick = 0.5, c='black', zorder=10)
        
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles, labels, title = 'No. epochs', title_fontsize = 'x-small', loc='upper left', prop={'size':6})
        ax.minorticks_on()
        ax.set_xlim(0,100)
        ax.set_ylim(0,0.2)
        ax.set_xticks(np.linspace(0,100,5))
        ax.set_yticks(np.linspace(0,0.2, 5))
        ax.grid(which='major', ls = '-', color = [0.15, 0.15, 0.15], alpha=0.15, zorder=0)
        ax.grid(which='minor', ls=':',  dashes=(1,5,1,5), color = [0.1, 0.1, 0.1], alpha=0.25, zorder=0) 
        ax.set_xlabel('Batch Size')
        ax.set_ylabel('Test RMSE')
        fig.savefig(os.environ['USERPROFILE'] + r"\Dropbox\Papers\PaperPGNN\__Paper\Fig_epoch_batchsize_NN.pdf")
        
        
        
        
    def gridsearch_opt(load=0):
        optimizer_names = ['SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam']
        if load != 0:
            score_RMSE, histories = [],[]
            for i in optimizer_names:
                hists, model, data, test_scores = NN_train_test(50, 20, i)
                score_RMSE.append(test_scores)
                histories.append(hists)
            all_info = {'RMSE':score_RMSE, 'History':histories}
            save_obj(all_info, 'Opt_Score_RMSE')
        else:
            score_RMSE = load_obj('Opt_Score_RMSE')
            RMSE = score_RMSE['RMSE']
            RMSE = (np.stack(RMSE).T)**0.5
            score = pd.DataFrame(data = RMSE, columns = optimizer_names)
            
        import seaborn as sns
        fig, ax = plt.subplots(1,1, figsize = (2.5,2.5), tight_layout = True)
        l = sns.pointplot(data = score, markers = 's', capsize=.2, errwidth = 0.5, color='black', join = False, ax = ax)
        plt.setp(l.get_xticklabels(), rotation=30)
        ax.minorticks_on()
        lim = 0.14
        ax.set_ylim(0,0.14)
        ax.set_yticks(np.linspace(0, lim, 7))
        ax.grid(which='major', ls = '-', color = [0.15, 0.15, 0.15], alpha=0.15)
        ax.grid(which='minor', ls=':',  dashes=(1,5,1,5), color = [0.1, 0.1, 0.1], alpha=0.25) 
        ax.set_xlabel('Optimizer')
        ax.set_ylabel('Test RMSE')
        fig.savefig(os.environ['USERPROFILE'] + r"\Dropbox\Papers\PaperPGNN\__Paper\Fig_opt_NN.pdf")


    def gridsearch_lr(load =0):
        learn_rate = [0.001, 0.01, 0.1, 0.2]
        if load != 0:
            score_RMSE, histories = [],[]
            for i in learn_rate:
                hists, model, data, test_scores = NN_train_test(50, 20, 'Nadam', learn_rate = i)
                score_RMSE.append(test_scores)
                histories.append(hists)
            all_info = {'RMSE':score_RMSE, 'History':histories}
            save_obj(all_info, 'Learnrate_Score_RMSE')
        else:
            score_RMSE = load_obj('Learnrate_Score_RMSE')
            RMSE = score_RMSE['RMSE']
            RMSE = (np.stack(RMSE).T)**0.5
            score = pd.DataFrame(data = RMSE, columns = learn_rate)
            
        fig, ax = plt.subplots(1,1, figsize = (2.5,2.5), tight_layout = True)
        err = np.stack((score.min().values.reshape(1, 4), score.max().values.reshape(1, 4)), axis = 1).reshape(2,4)
        err = abs(err - score.mean().values)
        ax.scatter(learn_rate, score.mean(), s=10, c='k')
        ax.errorbar(learn_rate, score.mean(), yerr = err, capsize = 3, capthick = 0.5, c='k')
        ax.set_xscale('log')
        ax.set_ylim(0,0.25)
        ax.minorticks_on()
        ax.grid(which='major', ls = '-', color = [0.15, 0.15, 0.15], alpha=0.15)
        ax.grid(which='minor', ls=':',  dashes=(1,5,1,5), color = [0.1, 0.1, 0.1], alpha=0.25) 
        ax.set_xlabel('Learn rate')
        ax.set_ylabel('Test RMSE')
        fig.savefig(os.environ['USERPROFILE'] + r"\Dropbox\Papers\PaperPGNN\__Paper\Fig_lr_NN.pdf")         
            
    def gridsearch_dropout(load = 0):
        dropout_rate = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        if load != 0:
            score_RMSE, histories = [],[]
            for i in dropout_rate:
                hists, model, data, test_scores = NN_train_test(50, 20, 'Nadam', learn_rate = 0.001, dropout = i)
                score_RMSE.append(test_scores)
                histories.append(hists)
            all_info = {'RMSE':score_RMSE, 'History':histories}
            save_obj(all_info, 'dropout_Score_RMSE')
                
        else:
            score_RMSE = load_obj('dropout_Score_RMSE')
            RMSE = score_RMSE['RMSE']
            RMSE = (np.stack(RMSE).T)**0.5
            score = pd.DataFrame(data = RMSE, columns = dropout_rate)
        
        fig, ax = plt.subplots(1,1, figsize = (2.5,2.5), tight_layout = True)
        err = np.stack((score.min().values.reshape(1, len(score.T)), score.max().values.reshape(1, len(score.T))), axis = 1).reshape(2,len(score.T))
        err = abs(err - score.mean().values)
        ax.scatter(dropout_rate, score.mean(), s=10, c='k')
        ax.errorbar(dropout_rate, score.mean(), yerr = err, capsize = 3, capthick = 0.5, c='k')
        ax.set_ylim(0,0.30)
        #ax.set_xlim(0,1)
        ax.minorticks_on()
        ax.grid(which='major', ls = '-', color = [0.15, 0.15, 0.15], alpha=0.15)
        ax.grid(which='minor', ls=':',  dashes=(1,5,1,5), color = [0.1, 0.1, 0.1], alpha=0.25) 
        ax.set_xlabel('Dropout')
        ax.set_ylabel('Test RMSE')                  
        fig.savefig(os.environ['USERPROFILE'] + r"\Dropbox\Papers\PaperPGNN\__Paper\Fig_dropout_NN.pdf")

    def performance_NN(load = 0):
        if load != 0:
            hists, model, data, test_scores = NN_train_test(50, 20, 'Adam', learn_rate = 0.001)
            model.save('obj/NNmodel')
            NN = {'hist':hists, 'data':data, 'test_scores':test_scores}
            save_obj(NN, 'NN')  
        else:
            NN = load_obj('NN')
            fig, ax = plt.subplots(1, 4, figsize=(6,1.8), tight_layout = True)
            fax = ax.ravel()
            for i in range(0,len(NN['hist'])):
                fax[i].plot(NN['hist'][i]['val_loss'], 'k--', label = 'Validation')
                fax[i].plot(NN['hist'][i]['loss'], 'k', label = 'Training')
                fax[i].minorticks_on()
                fax[i].grid(which='major', ls = '-', color = [0.15, 0.15, 0.15], alpha=0.15)
                fax[i].grid(which='minor', ls=':',  dashes=(1,5,1,5), color = [0.1, 0.1, 0.1], alpha=0.25)
                fax[i].set_xlim(0,50)
                fax[i].set_xticks([0,25,50])
                fax[i].set_xlabel('Epoch')
                 
                if i == 0:     
                    fax[i].set_ylabel('Loss') 
                    fax[i].set_yticks([0, 0.2, 0.4, 0.6])
                    handles, labels = fax[i].get_legend_handles_labels()
                    fax[i].legend(handles, labels, title_fontsize = 'x-small', loc='upper right', prop={'size':5})     
                else:
                    fax[i].set_yticks([0, 0.02,0.04,0.06])
                
            fax[0].set_ylim(0, 0.6)
            fax[1].set_ylim(0, 0.06)
            fax[2].set_ylim(0, 0.06)
            fax[3].set_ylim(0, 0.06)
            fig.savefig(os.environ['USERPROFILE'] + r"\Dropbox\Papers\PaperPGNN\__Paper\Fig_training_NN.pdf")

            import tensorflow as tf
            load_model = tf.keras.models.load_model('obj/NNmodel.h5', compile = False)
            

            fig, ax = plt.subplots(1,1, figsize=(2,2), tight_layout = True)
            ax.scatter(NN['data'][0]['y_unseen'], load_model.predict(NN['data'][0]['X_unseen']), s=10., color='black')
            text = "$R^2 = {:.3f}$".format(r2_score(NN['data'][0]['y_unseen'],load_model.predict(NN['data'][0]['X_unseen'])))
            ax.text(0.2, 0.8, text, fontsize = 'small', transform=ax.transAxes)
            ax.set_ylabel('Predicted response')
            ax.set_xlabel('Actual response')
            ax.set_xlim(-2,2)
            ax.set_ylim(-2,2)
            ax.minorticks_on()
            ax.grid(which='major', ls = '-', color = [0.15, 0.15, 0.15], alpha=0.15)
            ax.grid(which='minor', ls=':',  dashes=(1,5,1,5), color = [0.1, 0.1, 0.1], alpha=0.25)           
            fig.savefig(os.environ['USERPROFILE'] + r"\Dropbox\Papers\PaperPGNN\__Paper\Fig_unseenperformance_NN.pdf")


    def gridsearch_lamda12(load=0):
        """
        gridsearch PGNN12 with z & theta MLC
        """

        #Grid search batch size and num epochs
        lamda = np.logspace(-2,2,5)
        if load != 0:
            score_RMSE, histories, sim_detail = [],[], []
            for i in lamda:
                for j in lamda:
                    try:
                        hists, model, data, test_scores = NN_train_test(50, 20, 'Adam', learn_rate = 0.001,  lamda1 = i, lamda2 = j)
                        score_RMSE.append(test_scores)
                        histories.append(hists)
                        sim_detail.append((i,j))
                    except:
                        pass
            all_info = {'RMSE':score_RMSE, 'History':histories, 'Sim Detail': sim_detail}
            
            save_obj(all_info, 'PGNN_12_lamda')
        else:
            #heatmap?
            lam1, lam2 = np.meshgrid(lamda, lamda)
            
            
            score_RMSE = load_obj('PGNN_12_lamda')
            RMSE = score_RMSE['RMSE']
            RMSE = [np.stack(RMSE[i])[:,-1] for i in range(len(RMSE))]
            RMSE_final = np.asarray([RMSE[i][-1] for i in range(len(RMSE))]).reshape((len(lam1), len(lam2)))
            score = pd.DataFrame(data = RMSE_final, index = lamda, columns = lamda)
            #columns in score are lamda 2, index is lamda 1
            
            fig, ax = plt.subplots(1,1, figsize = (3.5,3.3), tight_layout = True)
            sns.heatmap(score, annot=True, annot_kws = {'fontsize':'x-small'},ax = ax)
            ax.set_ylabel('$\lambda_{Phy,1}$')
            ax.set_xlabel('$\lambda_{Phy,2}$')
            fig.savefig(os.environ['USERPROFILE'] + r"\Dropbox\Papers\PaperPGNN\__Paper\Fig_lamda_PGNN_12.pdf")
                    



    def performance_PGNN(load = 0):
        if load != 0:
            hists, model, data, test_scores = NN_train_test(50, 20, 'Adam', learn_rate = 0.001, lamda1 = np.logspace(-2,2,5)[-1], lamda2 = np.logspace(-2,2,5)[1])
            model.save('obj/PGNN_12_model.h5')
            PGNN = {'hist':hists, 'data':data, 'test_scores':test_scores}
            save_obj(PGNN, 'PGNN_12')  
        else:
            PGNN = load_obj('PGNN_12')
        
        fig, ax = plt.subplots(1, 4, figsize=(6,1.8), tight_layout = True)
        fax = ax.ravel()
        for i in range(0,len(PGNN['hist'])):
            fax[i].plot(PGNN['hist'][i]['val_loss'], 'k--', label = 'Validation')
            fax[i].plot(PGNN['hist'][i]['loss'], 'k', label = 'Training')
            fax[i].minorticks_on()
            fax[i].grid(which='major', ls = '-', color = [0.15, 0.15, 0.15], alpha=0.15)
            fax[i].grid(which='minor', ls=':',  dashes=(1,5,1,5), color = [0.1, 0.1, 0.1], alpha=0.25) 
            fax[i].set_xlim(0,50)
            fax[i].set_xticks([0,25,50])
            fax[i].set_xlabel('Epoch')
            if i == 0:                    
                handles, labels = fax[i].get_legend_handles_labels()
                fax[i].set_yticks([0, 0.2, 0.4, 0.6])
                fax[i].set_ylabel('Loss') 
                fax[i].legend(handles, labels, title_fontsize = 'x-small', loc='upper right', prop={'size':5})     
            else:
                ax[i].set_yticks([0, 0.02,0.04,0.06])
        fax[0].set_ylim(0, 0.6)
        fax[1].set_ylim(0, 0.06)
        fax[2].set_ylim(0, 0.06)
        fax[3].set_ylim(0, 0.06)
        fig.savefig(os.environ['USERPROFILE'] + r"\Dropbox\Papers\PaperPGNN\__Paper\Fig_training_PGNN_12.pdf")

        import tensorflow as tf
        load_model = tf.keras.models.load_model('obj/PGNN_12_model.h5', 
                                custom_objects={'loss':combined_loss}, compile = False)
        

        fig, ax = plt.subplots(1,1, figsize=(2,2), tight_layout = True)
        ax.scatter(PGNN['data'][0]['y_unseen'], load_model.predict(PGNN['data'][0]['X_unseen']), s=10., color='black')
        text = "$R^2 = {:.3f}$".format(r2_score(PGNN['data'][0]['y_unseen'],load_model.predict(PGNN['data'][0]['X_unseen'])))
        ax.text(0.2, 0.8, text, fontsize = 'small', transform=ax.transAxes)
        ax.set_ylabel('Predicted response')
        ax.set_xlabel('Actual response')
        ax.set_ylim(-2,2)
        ax.set_xlim(-2,2)
        ax.minorticks_on()
        ax.grid(which='major', ls = '-', color = [0.15, 0.15, 0.15], alpha=0.15)
        ax.grid(which='minor', ls=':',  dashes=(1,5,1,5), color = [0.1, 0.1, 0.1], alpha=0.25)           
        fig.savefig(os.environ['USERPROFILE'] + r"\Dropbox\Papers\PaperPGNN\__Paper\Fig_unseenperformance_PGNN_12.pdf")

    

            
    def stress_test_distribution_x(file_loc_string, data_kw):
        
        X_scaled, y_scaled, X_train, X_unseen, y_train, y_unseen, scaler_x, scaler2, scaler_y, X_scaled_og, y_og =load_data(1, test_frac = 0.5)
        nbins = np.histogram_bin_edges(y_scaled, bins = 40)
        nbins_z = np.histogram_bin_edges(X_scaled[:,0], bins = 18)
        nbins_theta = np.histogram_bin_edges(X_scaled[:,1], bins = 200)
        
        fs = (3.3, 1.8)
        histylim = 200
        
        fig, [ax, ax1] = plt.subplots(1,2, figsize = fs, tight_layout = True)
        X_scaled, y_scaled, X_train, X_unseen, y_train, y_unseen, scaler_x, scaler2, scaler_y, X_scaled_og, y_og =load_data(1, **{data_kw: 0.5})
        ax.hist(y_scaled, bins = nbins, alpha = 0.5, histtype = 'stepfilled', density = False, label = 'Original')
        ax.hist(y_train, bins = nbins, histtype = 'stepfilled', density = False, color = 'black', label = 'Train')
        ax.set_ylabel("Count")
        ax.set_xlabel("Y")
        ax.minorticks_on()
        ax.set_ylim(0, histylim)
        ax.set_yticks(np.linspace(0,histylim,3))
        ax.set_xlim(-2,2)
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles, labels, loc='upper right', prop={'size':6}) 
        
        hist = ax1.hist2d(X_train[:,0], X_train[:,1], bins = [nbins_z, nbins_theta], cmap = plt.cm.magma)
        cbar = plt.colorbar(hist[-1], ax = ax1)
        cbar.ax.set_ylabel("Count")
        cbar.set_ticks(np.linspace(0,1,3))
        
        ax1.set_xlabel('Z')
        ax1.set_ylabel(r'$\theta$')
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
        X_scaled, y_scaled, X_train, X_unseen, y_train, y_unseen, scaler_x, scaler2, scaler_y, X_scaled_og, y_og =load_data(1, test_frac = 0.01)
        parameterz = {'epsilon':np.logspace(-3,2, 8), 'C':np.logspace(-3,2, 8)}
        svr_rbf = SVR(kernel='rbf')
        clf = GridSearchCV(svr_rbf, parameterz, n_jobs = -1)
        opt = clf.fit(X_scaled, y_scaled.reshape(3600))
        
        #Prepare data structs
        svrModels, gbrModels = [], []
        svr_val_rmse, svr_test_rmse = [],[]
        reg_val_rmse, reg_test_rmse = [],[]
        NNhist, NNtest_scores = [], []
        PGNN_1_hist, PGNN_1_test_scores = [],[]
        PGNN_2_hist, PGNN_2_test_scores = [],[]
        PGNN_12_hist, PGNN_12_test_scores = [],[]
        
        for tf in tfs: 
            try:    
                #Blackbox
                X_scaled, y_scaled, X_train, X_unseen, y_train, y_unseen, scaler_x, scaler2, scaler_y, X_scaled_og, y_og =load_data(1, **{data_kw: tf})   
                cv = RepeatedKFold(n_splits = 4, n_repeats = 1)
                svr_rbf = SVR(kernel='rbf', C = opt.best_params_['C'], epsilon = opt.best_params_['epsilon'])
                svr_n_scores = cross_val_score(svr_rbf, X_train, y_train.reshape(len(y_train)), scoring = 'neg_mean_squared_error', cv = cv, n_jobs = -1)
                svr_val_rmse.append(abs(svr_n_scores) ** 0.5)
                svrModel = svr_rbf.fit(X_train, y_train.reshape(len(y_train)))
                svrModels.append(svrModel)
                error = svrModel.predict(X_unseen).reshape(len(X_unseen),1) - y_unseen
                error = error**2
                svr_test_rmse.append(np.mean(error)**0.5)
                reg = GradientBoostingRegressor(n_estimators = 2000)
                reg_n_scores = cross_val_score(reg, X_train, y_train.reshape(len(y_train)), scoring = 'neg_mean_squared_error', cv = cv, n_jobs = -1)
                reg_n_scores_rmse = abs(reg_n_scores) ** 0.5
                reg_val_rmse.append(abs(reg_n_scores) ** 0.5)
                gbrModel = reg.fit(X_train, y_train.reshape(len(y_train)))
                gbrModels.append(gbrModel)
                error = gbrModel.predict(X_unseen).reshape(len(X_unseen),1) - y_unseen
                error = error**2
                reg_test_rmse.append(np.mean(error)**0.5)      
                
                #NN
                hists, model, data, test_scores = NN_train_test(50, 20, 'Adam', learn_rate = 0.001, **{data_kw: tf})
                model.save('obj/remove'+ file_loc_string + '/NN' + str(int(tf*100)) +'.h5')
                NNhist.append(hists)
                NNtest_scores.append(test_scores)
                
                #PGNN1
                hists, model, data, test_scores = NN_train_test(50, 20, 'Adam', learn_rate = 0.001,  lamda1 = np.logspace(-2,2,10)[2], **{data_kw: tf})
                model.save('obj/remove'+ file_loc_string + '/PGNN_1_' + str(int(tf*100)) +'.h5')
                PGNN_1_hist.append(hists)
                PGNN_1_test_scores.append(test_scores)
                
                #PGNN2
                hists, model, data, test_scores = NN_train_test(50, 20, 'Adam', learn_rate = 0.001,  lamda2 = np.logspace(-2,2,10)[2], **{data_kw: tf})
                model.save('obj/remove'+ file_loc_string + '/PGNN_2_' + str(int(tf*100)) +'.h5')
                PGNN_2_hist.append(hists)
                PGNN_2_test_scores.append(test_scores)
    
                #PGNN12
                hists, model, data, test_scores = NN_train_test(50, 20, 'Adam', learn_rate = 0.001,  lamda1 = np.logspace(-2,2,5)[-1], lamda2 = np.logspace(-2,2,5)[1], **{data_kw: tf})
                model.save('obj/remove'+ file_loc_string + '/PGNN_12_' + str(int(tf*100)) +'.h5')
                PGNN_12_hist.append(hists)
                PGNN_12_test_scores.append(test_scores)
            except:
                pass
        
        
        to_save = {'svr_val_rmse':svr_val_rmse, 'svr_test_rmse':svr_test_rmse, 
                   'reg_val_rmse':reg_val_rmse, 'reg_test_rmse':reg_test_rmse,
                   'NNhist':NNhist, 'NNtest_scores':NNtest_scores,
                   'PGNN_1_hist':PGNN_1_hist, 'PGNN_1_test_scores':PGNN_1_test_scores,
                   'PGNN_2_hist':PGNN_2_hist, 'PGNN_2_test_scores':PGNN_2_test_scores,
                   'PGNN_12_hist':PGNN_12_hist, 'PGNN_12_test_scores':PGNN_12_test_scores,
                   'tfs':tfs,
                   'svrModels': svrModels, 'gbrModels':gbrModels}
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

    def remove_x_rmse_graph(file_loc_string, data_kw, axlims):
        
        rmseLim = axlims
        
        tfs = np.arange(0.1,1,0.1)
   
        all_data = load_obj('remove_'+file_loc_string+'_data')
        all_data['NNtest_scores'] = np.asarray(all_data['NNtest_scores'])
        all_data['PGNN_1_test_scores']= np.asarray(all_data['PGNN_1_test_scores'])
        all_data['PGNN_2_test_scores']= np.asarray(all_data['PGNN_2_test_scores'])
        all_data['PGNN_12_test_scores']= np.asarray(all_data['PGNN_12_test_scores'])
        
        fig, ax = plt.subplots(1,1, figsize = (2.3,2.3), tight_layout = True)
        ax.grid(which='major', ls = '-', color = [0.15, 0.15, 0.15], alpha=0.15, zorder=0)
        ax.grid(which='minor', ls=':',  dashes=(1,5,1,5), color = [0.1, 0.1, 0.1], alpha=0.25, zorder=0)
        
        ax.scatter(all_data['tfs'], all_data['svr_test_rmse'], c = 'grey', marker="s", edgecolor = 'k', s=10, label = 'SVR', zorder=20)
        ax.scatter(all_data['tfs'], all_data['reg_test_rmse'], c = 'yellow', marker="D", edgecolor = 'k', s=10, label = 'GBR', zorder=20)
        
        #NN
        err = np.stack((all_data['NNtest_scores'][:,:,0].min(1), all_data['NNtest_scores'][:,:,0].max(1)), axis = 1).T
        err = abs(err - all_data['NNtest_scores'][:,:,0].mean(1))
        ax.errorbar(all_data['tfs'], all_data['NNtest_scores'][:,:,0].mean(1), yerr = err, capsize = 3, capthick = 0.5, c='blue', ls='none', zorder=10)
        ax.scatter(all_data['tfs'], all_data['NNtest_scores'][:,:,0].mean(1), c = 'blue', marker="o", edgecolor = 'k', s=10, label = 'NN', zorder=20)
                  
        offset = (all_data['tfs'][1]- all_data['tfs'][0])* 0.2

        #PGNN_12
        err = np.stack((all_data['PGNN_12_test_scores'][:,:,0].min(1), all_data['PGNN_12_test_scores'][:,:,0].max(1)), axis = 1).T
        err = abs(err - all_data['PGNN_12_test_scores'][:,:,0].mean(1))
        ax.errorbar(all_data['tfs']+ 1*offset, all_data['PGNN_12_test_scores'][:,:,0].mean(1), yerr = err, capsize = 3, capthick = 0.5, c='red', ls='none', zorder=10)
        ax.scatter(all_data['tfs']+1*offset, all_data['PGNN_12_test_scores'][:,:,0].mean(1), c = 'red', marker="d", edgecolor = 'k', s=10,label = 'PGNN', zorder=20)
        
        
        ax.set_yscale('log')
        ax.set_ylabel('Test RMSE', fontsize='x-small')
        ax.set_xlabel('Holdout data fraction', fontsize='x-small')
        ax.set_xlim(0,1)
        ax.set_ylim(rmseLim)
        ax.minorticks_on()
   
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles, labels, loc='lower right', prop={'size':5}) 
        fig.savefig(os.environ['USERPROFILE'] + r"\Dropbox\Papers\PaperPGNN\__Paper\Fig_remove_"+file_loc_string+".pdf")

    # remove_x_rmse_graph('y_smallest', 'remove_smallest', (10**-3, 6*10**0)) 
    # remove_x_rmse_graph('z_largest', 'r_z_largest', (10**-3, 6*10**0))
    # remove_x_rmse_graph('theta_largest', 'r_theta_largest', (10**-3, 6*10**0))
    
    # remove_x_rmse_graph('y_largest', 'remove_largest', (10**-3, 6*10**0))
    # remove_x_rmse_graph('z_smallest', 'r_z_smallest', (10**-3, 6*10**0))
    # remove_x_rmse_graph('theta_smallest', 'r_theta_smallest', (10**-3, 6*10**0))
    
    # remove_x_rmse_graph('y_mean', 'remove_mean', (10**-3, 6*10**0))       
    # remove_x_rmse_graph('z_mean', 'r_z_mean', (10**-3, 6*10**0))
    # remove_x_rmse_graph('theta_mean', 'r_theta_mean', (10**-3, 6*10**0))
    
    # remove_x_rmse_graph('random', 'test_frac', (10**-3, 6*10**-2))


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
            #All scores
            NNs_extr.append(np.asarray(all_data['NNtest_scores'])[0:dh,:,0].flatten())
            PGNNs_extr.append(np.asarray(all_data['PGNN_12_test_scores'])[0:dh,:,0].flatten())
            # #Min scores
            # NNs_extr.append(np.asarray(all_data['NNtest_scores'])[0:dh,:,0].min(1).flatten())
            # PGNNs_extr.append(np.asarray(all_data['PGNN_12_test_scores'])[0:dh,:,0].min(1).flatten())  
            svrs_extr.append(np.asarray(all_data['svr_test_rmse'])[0:dh].flatten())
            gbrs_extr.append(np.asarray(all_data['reg_test_rmse'])[0:dh].flatten())
            
            
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
        #g.savefig("potential_thesis_figures\Fig_aggregated_logRMSE_"+str(dh*10)+".pdf")
        #g.savefig("potential_thesis_figures\Fig_aggregated_RMSE_"+str(dh*10)+".pdf")
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
    
    
    
    #for table
    mean_extr_tab = np.round(mean_extr * 100, 2)
    mean_intr_tab = np.round(mean_intr * 100, 2)
    med_extr_tab = np.round(med_extr * 100, 2)
    med_intr_tab = np.round(med_intr * 100, 2)
    



#Pickling functions
def save_obj(obj, name ):
    with open('obj/'+ name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
#save_obj(lsoa_dist, "lsoa_data")
def load_obj(name ):
    with open('obj/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)
