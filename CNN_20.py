##----------------------------------------------------------------------------      
## SLEEP APNEA DETECTION: COMPREHENSIVE ANALYSIS OF MACHINE LEARNING AND DEEP LEARNING METHODS
                                    
                                        ## DATE: 12-6-2021
                                        ## MODEL: VGG16
##-----------------------------------------------------------------------------
#IMPORT LIBRARIES    
import pickle
import numpy as np
import os
from keras.callbacks import LearningRateScheduler,EarlyStopping
from keras.layers import Dense,Flatten,MaxPooling2D,Conv2D
from keras.regularizers import l2
from scipy.interpolate import splev, splrep
from sklearn.model_selection import train_test_split
import tensorflow as tf
from keras.models import Sequential
from sklearn.model_selection import StratifiedKFold
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
base_dir = r'''D:\apnea-ecg-database-1.0.0'''
#------------------------------------------------------------------------------
ir = 1/360 # Interpolation steps 
max_freq = 0.5 # Max frequency of heart rate
#------------------------------------------------------------------------------
# normalize with Min-Max method:
scaler = lambda arr: (arr - np.min(arr)) / (np.max(arr) - np.min(arr))
#------------------------------------------------------------------------------

def load_data():
    tm = np.arange(0, max_freq , step= ir)
    # Load data: we used 1min_ahead for forecasting sleep apnea in 1-min ahead
    with open(os.path.join(base_dir, "apnea-prediction_1min_ahead.pkl"), 'rb') as f: # read preprocessing result
        apnea_ecg = pickle.load(f)

    x_train = []
    o_train, y_train = apnea_ecg["O"], apnea_ecg["Y"]
    for i in range(len(o_train)):
        (rri_tm, rri_signal), (am_tm, am_signal) = o_train[i]
		# Interpolating with K=3
        rri_interp_signal = splev(tm, splrep(rri_tm, scaler(rri_signal), k=3), ext=1) 
        ampl_interp_signal = splev(tm, splrep(am_tm, scaler(am_signal), k=3), ext=1)
        x_train.append([rri_interp_signal, ampl_interp_signal])
    x_train = np.expand_dims(x_train,1)
    k=0 
    x_final=[]
    y_train2=[]
    
    while k<len(y_train):

      X_fin=np.concatenate((x_train[k-19,:,:,:],x_train[k-18,:,:,:],x_train[k-17,:,:,:],
                            x_train[k-16,:,:,:],x_train[k-15,:,:,:],x_train[k-14,:,:,:],
                            x_train[k-13,:,:,:],x_train[k-12,:,:,:],
                            x_train[k-11,:,:,:],x_train[k-10,:,:,:]x_train[k-9,:,:,:],x_train[k-8,:,:,:],x_train[k-7,:,:,:],
                            x_train[k-6,:,:,:],x_train[k-5,:,:,:],x_train[k-4,:,:,:],
                            x_train[k-3,:,:,:],x_train[k-2,:,:,:],
                            x_train[k-1,:,:,:],x_train[k,:,:,:]),axis=0)
      y1=y_train[k]
      k+=1
      x_final.append(X_fin)
      y_train2.append(y1)
    x_final=np.array(x_final, dtype="float32").transpose((0, 3, 1, 2))


    return x_final, y_train2, groups_train
#------------------------------------------------------------------------------
# CREAT DEEP LEARNING MODEL
def create_model(weight=1e-3):
    model= Sequential()
    model.add(Conv2D(32, kernel_size=(5,1), strides=1, padding="valid", activation="relu", kernel_initializer="he_normal",
                kernel_regularizer=l2(weight), bias_regularizer=l2(weight),input_shape=(180,20,2)))
    model.add(MaxPooling2D(pool_size=(3,1)))

    model.add(Conv2D(64, kernel_size=(5,1), strides=1, padding="valid", activation="relu", kernel_initializer="he_normal",
                kernel_regularizer=l2(weight), bias_regularizer=l2(weight)))
    model.add(MaxPooling2D(pool_size=(3,1)))

    model.add(Flatten())
    
    model.add(Dense(152, activation="relu"))
    model.add(Dense(12, activation="relu"))
    model.add(Dense(2, activation="softmax"))
    return model

#------------------------------------------------------------------------------
# Define learning rate schedule for preventing overfitting in deep learning methods:
def lr_schedule(epoch, lr):
    if epoch > 30 and \
            (epoch - 1) % 10 == 0:
        lr *= 0.1
    print("Learning rate: ", lr)
    return lr
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
# Compile and evaluate model: 
if __name__ == "__main__":
    # loading Data:
    X, Y = load_data()
    # we have labels(Y) in a binary way 0 for normal and 1 for apnea patients
    # we want to classify data into 2-class so we changed y in a categorical way:
    Y = tf.keras.utils.to_categorical(Y, num_classes=2)
    # we used k-fold cross-validation for more reliable experiments: 
    kfold = StratifiedKFold(n_splits=10, shuffle=True,random_state=7)
    cvscores = []
    ACC=[]
    SN=[]
    SP=[]
    F2=[]
    # separate train& test and then compile model
    for train, test in kfold.split(X, Y.argmax(1)):
     model = create_model()

     
     model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=['accuracy'])
     # define callback for early stopping:
     lr_scheduler = LearningRateScheduler(lr_schedule)
     callback1 = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
     
     #10% of Data used for validation:
     X1,x_val,Y1,y_val=train_test_split(X[train],Y[train],test_size=0.10)
     
     history = model.fit(X1, Y1, batch_size=128, epochs=100, validation_data=(x_val, y_val),
                        callbacks=[callback1,lr_scheduler])
     
     loss, accuracy = model.evaluate(X[test], Y[test]) 

     y_score = model.predict(X[test])
     
     y_predict= np.argmax(y_score, axis=-1)
     y_training = np.argmax(Y[test], axis=-1)
     # Confusion matrix:
     from sklearn.metrics import confusion_matrix
     from sklearn.metrics import f1_score
     C = confusion_matrix(y_training, y_predict, labels=(1, 0))
     TP, TN, FP, FN = C[0, 0], C[1, 1], C[1, 0], C[0, 1]
     acc, sn, sp = 1. * (TP + TN) / (TP + TN + FP + FN), 1. * TP / (TP + FN), 1. * TN / (TN + FP)
     f2=f1_score(y_training, y_predict)
     
     ACC.append(acc * 100)
     SN.append(sn * 100)
     SP.append(sp * 100)
     F2.append(f2 * 100)