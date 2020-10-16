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

import multiprocessing

params = {'font.family':'serif',
        'axes.labelsize':'small',
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



def load_data(scaling_input, remove_last, remove_first, test_frac, remove_largest = None):
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
    
    
    #Testing extrapolation
    if remove_last != 0:
        X_scaled, y_scaled = X_scaled[:int(-200*remove_last),:], y_scaled[:int(-200*remove_last),:]
    else:
        pass
    
    if remove_first != 0:
        X_scaled, y_scaled = X_scaled[int(200*remove_first)::,:], y_scaled[int(200*remove_first)::,:]
    else:
        pass

   
    if remove_largest != None:
        inds = np.flip(np.argsort(y_og.reshape(3600)))
        limit = int(len(y_og) * remove_largest)
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



def NN_train_test(epochs, batch, opt_str, learn_rate = None, dropout = None, lamda1 = None, removefirst = None, removelast = None, holdout = None):
     
    #set lamdas=0 for pgnn0
    if lamda1 == None:
        lamda = 0 # Physics-based regularization constant - Z
    else:
        lamda = lamda1
    lamda2 = 0  #Physics-based regularization constant - theta monotonic
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
    
    #Data
    if holdout == None:
        test_frac = 0.10
    else:
        test_frac = holdout
    
    patience_val = int(0.3 * num_epochs)

    #load data
    scaling_input = 1
    
    if removefirst == None:
        remove_first = 0
    else:
        remove_first = removefirst
            
    if removelast == None:
        remove_last = 0
    else:
        remove_last = removelast
    
    X_scaled, y_scaled, X_train, X_unseen, y_train, y_unseen, scaler_x, scaler2, scaler_y, X_scaled_og, y_og = load_data(scaling_input, remove_last, remove_first, test_frac)
    
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
    
    if lamda1 == None:
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
    
    if lamda1 == None:
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
        
        fig, ax = plt.subplots(1,1, figsize = (2.5,2.5), tight_layout = True)
        ax.scatter(batch_size, RMSE_10.mean(0), s=10, c='blue', label = '10')
        err = np.stack((RMSE_10.min(0).reshape(1, 6), RMSE_10.max(0).reshape(1, 6)), axis = 1).reshape(2,6)
        err = abs(err - RMSE_10.mean(0))
        ax.errorbar(batch_size, RMSE_10.mean(0), yerr = err, capsize = 3, capthick = 0.5, c='blue')
        
        
        ax.scatter(batch_size, RMSE_50.mean(0), s=10, c='red', label = '50')
        err = np.stack((RMSE_50.min(0).reshape(1, 6), RMSE_50.max(0).reshape(1, 6)), axis = 1).reshape(2,6)
        err = abs(err - RMSE_50.mean(0))
        ax.errorbar(batch_size, RMSE_50.mean(0), yerr = err, capsize = 3, capthick = 0.5, c='red')
        
        ax.scatter(batch_size, RMSE_100.mean(0), s=10, c='black', label = '100')
        err = np.stack((RMSE_100.min(0).reshape(1, 6), RMSE_100.max(0).reshape(1, 6)), axis = 1).reshape(2,6)
        err = abs(err - RMSE_100.mean(0))
        ax.errorbar(batch_size, RMSE_100.mean(0), yerr = err, capsize = 3,capthick = 0.5, c='black')
        
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles, labels, title = 'No. epochs', title_fontsize = 'x-small', loc='upper left', prop={'size':6})
        ax.minorticks_on()
        ax.set_xlim(0,120)
        ax.set_ylim(0,0.2)
        ax.grid(which='major', ls = '-', color = [0.15, 0.15, 0.15], alpha=0.15)
        ax.grid(which='minor', ls=':',  dashes=(1,5,1,5), color = [0.1, 0.1, 0.1], alpha=0.25) 
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
        ax.set_ylim(0,0.16)
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

    def performance(load = 0):
        if load != 0:
            hists, model, data, test_scores = NN_train_test(50, 20, 'Adam', learn_rate = 0.001)
            model.save('obj/NNmodel')
            NN = {'hist':hists, 'data':data, 'test_scores':test_scores}
            save_obj(NN, 'NN')  
        else:
            NN = load_obj('NN')
            fig, ax = plt.subplots(2, 2, figsize=(5,5), tight_layout = True)
            fax = ax.ravel()
            for i in range(0,len(NN['hist'])):
                fax[i].plot(NN['hist'][i]['val_loss'], 'k--', label = 'Validation set')
                fax[i].plot(NN['hist'][i]['loss'], 'k', label = 'Training set')
                fax[i].minorticks_on()
                fax[i].grid(which='major', ls = '-', color = [0.15, 0.15, 0.15], alpha=0.15)
                fax[i].grid(which='minor', ls=':',  dashes=(1,5,1,5), color = [0.1, 0.1, 0.1], alpha=0.25) 
                fax[i].set_xlabel('Epoch')
                fax[i].set_ylabel('Loss')  
                if i == 0:                    
                    handles, labels = fax[i].get_legend_handles_labels()
                    fax[i].legend(handles, labels, title_fontsize = 'x-small', loc='upper right', prop={'size':6})     
                else:
                    pass
            fig.savefig(os.environ['USERPROFILE'] + r"\Dropbox\Papers\PaperPGNN\__Paper\Fig_training_NN.pdf")

            import tensorflow as tf
            load_model = tf.keras.models.load_model('obj/NNmodel.h5', compile = False)
            

            fig, ax = plt.subplots(1,1, figsize=(2.5,2.5), tight_layout = True)
            ax.scatter(NN['data'][0]['y_unseen'], load_model.predict(NN['data'][0]['X_unseen']), s=10., color='black')
            text = "$R^2 = {:.3f}$".format(r2_score(NN['data'][0]['y_unseen'],load_model.predict(NN['data'][0]['X_unseen'])))
            ax.text(0.2, 0.8, text, fontsize = 'small', transform=ax.transAxes)
            ax.set_ylabel('Predicted response')
            ax.set_xlabel('Actual response')
            #ax.set_xlim(0,1)
            #ax.set_ylim(0,1)
            ax.minorticks_on()
            ax.grid(which='major', ls = '-', color = [0.15, 0.15, 0.15], alpha=0.15)
            ax.grid(which='minor', ls=':',  dashes=(1,5,1,5), color = [0.1, 0.1, 0.1], alpha=0.25)           
            fig.savefig(os.environ['USERPROFILE'] + r"\Dropbox\Papers\PaperPGNN\__Paper\Fig_unseenperformance_NN.pdf")


    def gridsearch_lamda(load=0):
        
        #Grid search batch size and num epochs
        lamda = np.logspace(-2,1,10)
        if load != 0:
            score_RMSE, histories = [],[]
            for i in lamda:
                    hists, model, data, test_scores = NN_train_test(50, 20, 'Adam', learn_rate = 0.001,  lamda1 = i)
                    score_RMSE.append(test_scores)
                    histories.append(hists)
            all_info = {'RMSE':score_RMSE, 'History':histories}
            
            save_obj(all_info, 'PCNN_lamda1')
        else:
            score_RMSE = load_obj('PCNN_lamda1')
            RMSE = score_RMSE['RMSE']
            RMSE = [np.stack(RMSE[i])[:,-1] for i in range(len(RMSE))] 
            score = pd.DataFrame(np.stack(RMSE).T, columns = lamda)
        
            
        fig, ax = plt.subplots(1,1, figsize = (2.5,2.5), tight_layout = True)
        err = np.stack((score.min().values.reshape(1, 10), score.max().values.reshape(1, 10)), axis = 1).reshape(2,10)
        err = abs(err - score.mean().values)
        ax.scatter(lamda, score.mean(), s=10, c='k')
        ax.errorbar(lamda, score.mean(), yerr = err, capsize = 3, capthick = 0.5, c='k')
        ax.set_xscale('log')
        #ax.set_ylim(0,0.25)
        ax.minorticks_on()
        ax.grid(which='major', ls = '-', color = [0.15, 0.15, 0.15], alpha=0.15)
        ax.grid(which='minor', ls=':',  dashes=(1,5,1,5), color = [0.1, 0.1, 0.1], alpha=0.25) 
        ax.set_xlabel('Lambda')
        ax.set_ylabel('Test RMSE')
        fig.savefig(os.environ['USERPROFILE'] + r"\Dropbox\Papers\PaperPGNN\__Paper\Fig_lamda_PGNN.pdf")

    def PGNNperformance(load = 1):
        if load != 0:
            hists, model, data, test_scores = NN_train_test(50, 20, 'Adam', learn_rate = 0.001, lamda1 = np.logspace(-2,1,10)[2])
            model.save('obj/PGNNmodel.h5')
            PGNN = {'hist':hists, 'data':data, 'test_scores':test_scores}
            save_obj(PGNN, 'PGNN')  
        else:
            PGNN = load_obj('PGNN')
        
        fig, ax = plt.subplots(2, 2, figsize=(5,5), tight_layout = True)
        fax = ax.ravel()
        for i in range(0,len(PGNN['hist'])):
            fax[i].plot(PGNN['hist'][i]['val_loss'], 'k--', label = 'Validation set')
            fax[i].plot(PGNN['hist'][i]['loss'], 'k', label = 'Training set')
            fax[i].minorticks_on()
            fax[i].grid(which='major', ls = '-', color = [0.15, 0.15, 0.15], alpha=0.15)
            fax[i].grid(which='minor', ls=':',  dashes=(1,5,1,5), color = [0.1, 0.1, 0.1], alpha=0.25) 
            fax[i].set_xlabel('Epoch')
            fax[i].set_ylabel('Loss')  
            if i == 0:                    
                handles, labels = fax[i].get_legend_handles_labels()
                fax[i].legend(handles, labels, title_fontsize = 'x-small', loc='upper right', prop={'size':6})     
            else:
                pass
        fig.savefig(os.environ['USERPROFILE'] + r"\Dropbox\Papers\PaperPGNN\__Paper\Fig_training_PGNN.pdf")

        import tensorflow as tf
        load_model = tf.keras.models.load_model('obj/PGNNmodel.h5', 
                                custom_objects={'loss':combined_loss}, compile = False)
        

        fig, ax = plt.subplots(1,1, figsize=(2.5,2.5), tight_layout = True)
        ax.scatter(PGNN['data'][0]['y_unseen'], load_model.predict(PGNN['data'][0]['X_unseen']), s=10., color='black')
        text = "$R^2 = {:.3f}$".format(r2_score(PGNN['data'][0]['y_unseen'],load_model.predict(PGNN['data'][0]['X_unseen'])))
        ax.text(0.2, 0.8, text, fontsize = 'small', transform=ax.transAxes)
        ax.set_ylabel('Predicted response')
        ax.set_xlabel('Actual response')
        ax.minorticks_on()
        ax.grid(which='major', ls = '-', color = [0.15, 0.15, 0.15], alpha=0.15)
        ax.grid(which='minor', ls=':',  dashes=(1,5,1,5), color = [0.1, 0.1, 0.1], alpha=0.25)           
        fig.savefig(os.environ['USERPROFILE'] + r"\Dropbox\Papers\PaperPGNN\__Paper\Fig_unseenperformance_PGNN.pdf")

    def extrapolate_networks(load = 0):
        rf, rl = 4, 4
        if load != 0:
            hists, model, data, test_scores = NN_train_test(50, 20, 'Adam', learn_rate = 0.001,  lamda1 = 0, removefirst = rf, removelast = rl)
            model.save('obj/NNmodel_extrapolate.h5')
            NN = {'hist':hists, 'data':data, 'test_scores':test_scores}
            save_obj(NN, 'NN_extrapolate')   
            
            hists, model, data, test_scores = NN_train_test(50, 20, 'Adam', learn_rate = 0.001, lamda1 = np.logspace(-2,1,10)[6], removefirst = rf, removelast = rl)
            model.save('obj/PGNNmodel_extrapolate.h5')
            PGNN = {'hist':hists, 'data':data, 'test_scores':test_scores}
            save_obj(PGNN, 'PGNN_extrapolate')          
        else:
            NN = load_obj('NN_extrapolate')
            PGNN = load_obj('PGNN_extrapolate')
            
            import tensorflow as tf
            load_NN = tf.keras.models.load_model('obj/NNmodel_extrapolate.h5', compile = False)
            load_PGNN = tf.keras.models.load_model('obj/PGNNmodel_extrapolate.h5', 
                                                    custom_objects={'loss':combined_loss}, compile = False)
        
            #Dataset plots 1
            X_scaled, y_scaled, X_train, X_unseen, y_train, y_unseen, scaler_x, scaler2, scaler_y, X_scaled_og, y_og = load_data(1, remove_last = rl, remove_first = rf, test_frac=0.05)
            fig, ax = plt.subplots(3, 3, figsize=(6,6), tight_layout = True)
            fax = ax.ravel()
            for i in range(0,9):
                exp = 200 * i
                pred_NN = load_NN.predict(X_scaled_og[np.arange(0,200,1)+exp,:]).reshape(200,1)
                pred_PGNN = load_PGNN.predict(X_scaled_og[np.arange(0,200,1)+exp,:]).reshape(200,1)
                theta = np.linspace(0,80,200).reshape(200,1)                
                fax[i].plot(theta, y_og[np.arange(0,200,1)+exp], 'k', label = 'CFD')                 
                fax[i].plot(theta, scaler_y.inverse_transform(pred_NN), 'r', label = 'NN')
                fax[i].plot(theta, scaler_y.inverse_transform(pred_PGNN), 'r--', label = 'PGNN')
                fax[i].set_title("Z ="+str(np.round(scaler_x.inverse_transform(X_scaled_og[exp].reshape(1,2))[0][0],3)), fontsize='x-small')
                fax[i].set_ylabel('Peak specific impulse (MPa.ms)', fontsize='x-small')
                fax[i].set_xlabel('Theta', fontsize='x-small')
                fax[i].set_xlim(0,80)
                fax[i].minorticks_on()
                fax[i].grid(which='major', ls = '-', color = [0.15, 0.15, 0.15], alpha=0.15)
                fax[i].grid(which='minor', ls=':',  dashes=(1,5,1,5), color = [0.1, 0.1, 0.1], alpha=0.25)   
            handles, labels = fax[0].get_legend_handles_labels()
            fax[0].legend(handles, labels, loc='upper right', prop={'size':6})
            fig.savefig(os.environ['USERPROFILE'] + r"\Dropbox\Papers\PaperPGNN\__Paper\Fig_extrapolate_1.pdf")
            
            fig, ax = plt.subplots(3, 3, figsize=(6,6), tight_layout = True)
            fax = ax.ravel()            
            for i in range(0,9):
                exp = 200 * (i+9)
                pred_NN = load_NN.predict(X_scaled_og[np.arange(0,200,1)+exp,:]).reshape(200,1)
                pred_PGNN = load_PGNN.predict(X_scaled_og[np.arange(0,200,1)+exp,:]).reshape(200,1)
                theta = np.linspace(0,80,200).reshape(200,1)                
                fax[i].plot(theta, y_og[np.arange(0,200,1)+exp], 'k', label = 'CFD')                 
                fax[i].plot(theta, scaler_y.inverse_transform(pred_NN), 'r', label = 'NN')
                fax[i].plot(theta, scaler_y.inverse_transform(pred_PGNN), 'r--', label = 'PGNN')
                fax[i].set_title("Z ="+str(np.round(scaler_x.inverse_transform(X_scaled_og[exp].reshape(1,2))[0][0],3)), fontsize='x-small')
                fax[i].set_ylabel('Peak specific impulse (MPa.ms)', fontsize='x-small')
                fax[i].set_xlabel('Theta', fontsize='x-small')
                fax[i].set_xlim(0,80)
                fax[i].minorticks_on()
                fax[i].grid(which='major', ls = '-', color = [0.15, 0.15, 0.15], alpha=0.15)
                fax[i].grid(which='minor', ls=':',  dashes=(1,5,1,5), color = [0.1, 0.1, 0.1], alpha=0.25)   
            handles, labels = fax[0].get_legend_handles_labels()
            fax[0].legend(handles, labels, loc='upper right', prop={'size':6})
            fig.savefig(os.environ['USERPROFILE'] + r"\Dropbox\Papers\PaperPGNN\__Paper\Fig_extrapolate_2.pdf")

    def holdout_networks_uniform(load = 0):
        tfs = np.arange(0.1,0.9,0.1)
        tfs = [0.9, 0.95, 0.98]
        if load != 0 :
            
            X_scaled, y_scaled, X_train, X_unseen, y_train, y_unseen, scaler_x, scaler2, scaler_y, X_scaled_og, y_og =load_data(1, 0,0, 0.10)
            #grid search svr
            parameterz = {'epsilon':np.logspace(-3,2, 8), 'C':np.logspace(-3,2, 8)}
            svr_rbf = SVR(kernel='rbf')
            clf = GridSearchCV(svr_rbf, parameterz, n_jobs = -1)
            opt = clf.fit(X_scaled, y_scaled.reshape(3600))
            
            svr_val_rmse, svr_test_rmse = [],[]
            reg_val_rmse, reg_test_rmse = [],[]
            NNhist, NNtest_scores = [], []
            PGNNhist, PGNNtest_scores = [],[]
            for tf in tfs: 
                X_scaled, y_scaled, X_train, X_unseen, y_train, y_unseen, scaler_x, scaler2, scaler_y, X_scaled_og, y_og =load_data(1, 0,0, tf)               
                
                cv = RepeatedKFold(n_splits = 4, n_repeats = 1)
                
                svr_rbf = SVR(kernel='rbf', C = opt.best_params_['C'], epsilon = opt.best_params_['epsilon'])
                svr_n_scores = cross_val_score(svr_rbf, X_train, y_train.reshape(len(y_train)), scoring = 'neg_mean_squared_error', cv = cv, n_jobs = -1)
                svr_val_rmse.append(abs(svr_n_scores) ** 0.5)
                svr_rbf.fit(X_train, y_train.reshape(len(y_train)))
                error = svr_rbf.predict(X_unseen).reshape(len(X_unseen),1) - y_unseen
                error = error**2
                svr_test_rmse.append(np.mean(error)**0.5)
                    
                
                reg = GradientBoostingRegressor(n_estimators = 2000)
                reg_n_scores = cross_val_score(reg, X_train, y_train.reshape(len(y_train)), scoring = 'neg_mean_squared_error', cv = cv, n_jobs = -1)
                reg_n_scores_rmse = abs(reg_n_scores) ** 0.5
                reg_val_rmse.append(abs(reg_n_scores) ** 0.5)
                reg.fit(X_train, y_train.reshape(len(y_train)))
                error = reg.predict(X_unseen).reshape(len(X_unseen),1) - y_unseen
                error = error**2
                reg_test_rmse.append(np.mean(error)**0.5)      
            
                hists, model, data, test_scores = NN_train_test(50, 20, 'Adam', learn_rate = 0.001)
                NNhist.append(hists)
                NNtest_scores.append(test_scores)
    
                hists, model, data, test_scores = NN_train_test(50, 20, 'Adam', learn_rate = 0.001,  lamda1 = np.logspace(-2,1,10)[6])
                PGNNhist.append(hists)
                PGNNtest_scores.append(test_scores)
            
            
            to_save = {'svr_val_rmse':svr_val_rmse, 'svr_test_rmse':svr_test_rmse, 
                       'reg_val_rmse':reg_val_rmse, 'reg_test_rmse':reg_test_rmse,
                       'NNhist':NNhist, 'NNtest_scores':NNtest_scores,
                       'PGNNhist':PGNNhist, 'PGNNtest_scores':PGNNtest_scores}
            save_obj(to_save, 'HoldoutData')
        
        else:
            all_data = load_obj('HoldoutData')
            all_data['NNtest_scores'] = np.asarray(all_data['NNtest_scores'])
            all_data['PGNNtest_scores_additional_mse'] = np.asarray(all_data['PGNNtest_scores'])[:,:,-1]
            all_data['PGNNtest_scores']= np.asarray(all_data['PGNNtest_scores'])[:,:,0]
            
            fig, ax = plt.subplots(1,1, figsize = (2.5,2.5), tight_layout = True)
            ax.scatter(all_data['tfs'], all_data['svr_test_rmse'], s=10, c='blue', label = 'SVR')
            ax.scatter(all_data['tfs'], all_data['reg_test_rmse'], s=10, c='red', label = 'GBR')
            ax.scatter(all_data['tfs'], all_data['NNtest_scores'][:,-1], s=10, c='black',label = 'NN')
            ax.scatter(all_data['tfs'], all_data['PGNNtest_scores'][:,-2], s=10, c='green',label = 'PGNN')
            ax.set_ylabel('Test RMSE', fontsize='x-small')
            ax.set_xlabel('Holdout data fraction', fontsize='x-small')
            #ax.set_xlim(0,80)
            ax.minorticks_on()
            ax.grid(which='major', ls = '-', color = [0.15, 0.15, 0.15], alpha=0.15)
            ax.grid(which='minor', ls=':',  dashes=(1,5,1,5), color = [0.1, 0.1, 0.1], alpha=0.25)   
            handles, labels = ax.get_legend_handles_labels()
            ax.legend(handles, labels, loc='upper left', prop={'size':6}) 
            fig.savefig(os.environ['USERPROFILE'] + r"\Dropbox\Papers\PaperPGNN\__Paper\Fig_holdout_fraction_performance.pdf")
            
    def holdout_networks_nonuniform(load = 0):
        tfs = np.arange(0.05,0.4,0.05)
        if load != 0 :
            
            X_scaled, y_scaled, X_train, X_unseen, y_train, y_unseen, scaler_x, scaler2, scaler_y, X_scaled_og, y_og =load_data(1, 0,0, 0.10)
            #grid search svr
            parameterz = {'epsilon':np.logspace(-3,2, 8), 'C':np.logspace(-3,2, 8)}
            svr_rbf = SVR(kernel='rbf')
            clf = GridSearchCV(svr_rbf, parameterz, n_jobs = -1)
            opt = clf.fit(X_scaled, y_scaled.reshape(3600))
            
            svr_val_rmse, svr_test_rmse = [],[]
            reg_val_rmse, reg_test_rmse = [],[]
            NNhist, NNtest_scores = [], []
            PGNNhist, PGNNtest_scores = [],[]
            for tf in tfs: 
                X_scaled, y_scaled, X_train, X_unseen, y_train, y_unseen, scaler_x, scaler2, scaler_y, X_scaled_og, y_og =load_data(1, 0,0, 0.01, remove_largest = tf)               
                
                cv = RepeatedKFold(n_splits = 4, n_repeats = 1)
                
                svr_rbf = SVR(kernel='rbf', C = opt.best_params_['C'], epsilon = opt.best_params_['epsilon'])
                svr_n_scores = cross_val_score(svr_rbf, X_train, y_train.reshape(len(y_train)), scoring = 'neg_mean_squared_error', cv = cv, n_jobs = -1)
                svr_val_rmse.append(abs(svr_n_scores) ** 0.5)
                svr_rbf.fit(X_train, y_train.reshape(len(y_train)))
                error = svr_rbf.predict(X_unseen).reshape(len(X_unseen),1) - y_unseen
                error = error**2
                svr_test_rmse.append(np.mean(error)**0.5)
                    
                
                reg = GradientBoostingRegressor(n_estimators = 2000)
                reg_n_scores = cross_val_score(reg, X_train, y_train.reshape(len(y_train)), scoring = 'neg_mean_squared_error', cv = cv, n_jobs = -1)
                reg_n_scores_rmse = abs(reg_n_scores) ** 0.5
                reg_val_rmse.append(abs(reg_n_scores) ** 0.5)
                reg.fit(X_train, y_train.reshape(len(y_train)))
                error = reg.predict(X_unseen).reshape(len(X_unseen),1) - y_unseen
                error = error**2
                reg_test_rmse.append(np.mean(error)**0.5)      
            
                hists, model, data, test_scores = NN_train_test(50, 20, 'Adam', learn_rate = 0.001)
                NNhist.append(hists)
                NNtest_scores.append(test_scores)
    
                hists, model, data, test_scores = NN_train_test(50, 20, 'Adam', learn_rate = 0.001,  lamda1 = np.logspace(-2,1,10)[6])
                PGNNhist.append(hists)
                PGNNtest_scores.append(test_scores)
            
            
            to_save = {'svr_val_rmse':svr_val_rmse, 'svr_test_rmse':svr_test_rmse, 
                       'reg_val_rmse':reg_val_rmse, 'reg_test_rmse':reg_test_rmse,
                       'NNhist':NNhist, 'NNtest_scores':NNtest_scores,
                       'PGNNhist':PGNNhist, 'PGNNtest_scores':PGNNtest_scores,
                       'tfs':tfs}
            save_obj(to_save, 'HoldoutData_nonuniform')
        
        else:
            all_data = load_obj('HoldoutData_nonuniform')
            all_data['NNtest_scores'] = np.asarray(all_data['NNtest_scores'])
            all_data['PGNNtest_scores_additional_mse'] = np.asarray(all_data['PGNNtest_scores'])[:,:,-1]
            all_data['PGNNtest_scores']= np.asarray(all_data['PGNNtest_scores'])[:,:,0]
            
            fig, ax = plt.subplots(1,1, figsize = (2.5,2.5), tight_layout = True)
            ax.scatter(all_data['tfs'], all_data['svr_test_rmse'], s=10, c='blue', label = 'SVR')
            ax.scatter(all_data['tfs'], all_data['reg_test_rmse'], s=10, c='red', label = 'GBR')
            ax.scatter(all_data['tfs'], all_data['NNtest_scores'][:,-1], s=10, c='black',label = 'NN')
            ax.scatter(all_data['tfs'], all_data['PGNNtest_scores'][:,-2], s=10, c='green',label = 'PGNN')
            ax.set_ylabel('Test RMSE', fontsize='x-small')
            ax.set_xlabel('Holdout data fraction', fontsize='x-small')
            #ax.set_xlim(0,80)
            ax.minorticks_on()
            ax.grid(which='major', ls = '-', color = [0.15, 0.15, 0.15], alpha=0.15)
            ax.grid(which='minor', ls=':',  dashes=(1,5,1,5), color = [0.1, 0.1, 0.1], alpha=0.25)   
            handles, labels = ax.get_legend_handles_labels()
            ax.legend(handles, labels, loc='upper left', prop={'size':6}) 
            fig.savefig(os.environ['USERPROFILE'] + r"\Dropbox\Papers\PaperPGNN\__Paper\Fig_holdout_fraction_performance_nonuniform.pdf")        
          


"""

if plotting != 0:
    

    
    
    
    #validation against completely unseen data
    fn_val_highZ = os.environ['USERPROFILE'] + r"\Dropbox\Papers\PaperPGNN\datasets\spherical_val_highZ.csv"
    val_highZ = pd.read_csv(fn_val_highZ, header = None)
    val_highZ = val_highZ.values
    fn_val_lowZ = os.environ['USERPROFILE'] + r"\Dropbox\Papers\PaperPGNN\datasets\spherical_val_lowZ.csv"
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
    
    #peak I vs Z
    fig, ax = plt.subplots(2, 3, figsize=(8,6))
    fax = ax.ravel()
    angles = [0, 20, 30, 40, 60, 80]
    for i in range(len(angles)):
        n = 50 #resolution of predictions
        sampleZ = np.linspace(0, 1, n).reshape(n,1)
        if angles[i] == 0:
            idtheta = 0
            
        elif angles[i] == 80:
            idtheta = 199
        else:
            idtheta = np.where(np.linspace(0,80,200) > angles[i])[0][0]
        sampleZ = np.concatenate((sampleZ, np.zeros((n , 1)) + X_scaled_og[idtheta,1]), axis = 1) 
        
        fax[i].plot(sampleZ[:,0], 
                    scaler_y.inverse_transform(model.predict(sampleZ).reshape(len(sampleZ),1)),
                    'k', label='NN')
        fax[i].scatter(X_scaled_og[np.where(X[:,1]==X[idtheta,1]),0], y_og[np.where(X[:,1]==X[idtheta,1]),0], marker = 's',facecolors = 'white', edgecolors='k', s = 10., label = 'CFD data')
        fax[i].minorticks_on()
        fax[i].grid(which='minor', ls=':', dashes=(1,5,1,5), color = [0.1, 0.1, 0.1], alpha=0.25)
        fax[i].grid(which='major', ls = '-', color = [0.15, 0.15, 0.15], alpha=0.15)
        fax[i].set_title(str(angles[i]) + "degrees")
        fax[i].set_xlabel("Normalised Z")
        fax[i].set_ylabel("specific impulse (MPa.ms)")                 
        plt.tight_layout()

#Exploratory Contour plot
z = np.linspace(X.min(0)[0], X.max(0)[0], 200)
theta = np.linspace(0,80,200)
[z1, theta1] = np.meshgrid(z,theta)
z = z1.flatten('F')
z = z.reshape(len(z), 1)
theta = theta1.flatten('F')
theta = theta.reshape(len(theta), 1)
to_pred = scaler_x.transform(np.concatenate((z, theta), axis = 1))
pred = scaler_y.inverse_transform(model.predict(to_pred).reshape(len(to_pred),1))
pred = pred.reshape(np.shape(z1), order = 'F')
fig0, ax = plt.subplots(1,1, figsize=(5,5))
CS = ax.contourf(theta1, z1, pred, levels = 10, cmap = plt.cm.magma_r) # levels = np.linspace(0,25,50)
#ax.clabel(CS, inline=1, fontsize=10)
cbar = fig0.colorbar(CS, format='%.1f') #ticks = np.linspace(0,25,6)
cbar.ax.set_ylabel('Scaled specific impulse '+r'$(MPa.ms/kg^{1/3}$)', fontsize = 'x-small')
ax.set_ylabel('Scaled distance, Z ' + r'$(m/kg^{1/3}$)')
ax.set_xlabel('Angle of incidence (degrees)')
plt.tight_layout()


#Comparison Contour 
z = np.linspace(X.min(0)[0], X.max(0)[0], 18)
theta = np.linspace(0,80,200)
[z1, theta1] = np.meshgrid(z,theta)
z = z1.flatten('F')
z = z.reshape(len(z), 1)
theta = theta1.flatten('F')
theta = theta.reshape(len(theta), 1)
to_pred = scaler_x.transform(np.concatenate((z, theta), axis = 1))
pred = scaler_y.inverse_transform(model.predict(to_pred).reshape(len(to_pred),1))
pred = pred.reshape(np.shape(z1), order = 'F')
residuals = pred-y.reshape(np.shape(pred))
                           
fig0, ax = plt.subplots(1,1, figsize=(5,5))
CS = ax.contourf(theta1, z1, residuals, levels = 10, cmap = plt.cm.magma_r) # levels = np.linspace(0,25,50)
#ax.clabel(CS, inline=1, fontsize=10)
cbar = fig0.colorbar(CS, format='%.1f') #ticks = np.linspace(0,25,6)
cbar.ax.set_ylabel('Scaled specific impulse '+r'$(MPa.ms/kg^{1/3}$)', fontsize = 'x-small')
ax.set_ylabel('Scaled distance, Z ' + r'$(m/kg^{1/3}$)')
ax.set_xlabel('Angle of incidence (degrees)')
plt.tight_layout()

#Histogram of residuals
fig, ax = plt.subplots(1,1, figsize=(5,5), tight_layout = True)
ax.hist(residuals.flatten('F'), bins = 20, density = True)
ax.yaxis.set_major_formatter(PercentFormatter(xmax=1))

#Histogram of residuals
fig, ax = plt.subplots(1,1, figsize=(5,5), tight_layout = True)
ax.hist(res[0].flatten('F'), bins = 20, density = True)
ax.hist(res[1].flatten('F'), bins = 20, density = True)
ax.yaxis.set_major_formatter(PercentFormatter(xmax=1))



"""






#Pickling functions
def save_obj(obj, name ):
    with open('obj/'+ name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
#save_obj(lsoa_dist, "lsoa_data")
def load_obj(name ):
    with open('obj/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)
