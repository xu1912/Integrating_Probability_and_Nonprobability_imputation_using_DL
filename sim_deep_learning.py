from sklearn.preprocessing import StandardScaler
import pandas as pd
from keras.models import Model, Sequential, load_model
from keras.layers import *
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import applications, optimizers
import numpy as np

# Make sure the path is specific to test 3, 4, 5; relu 3, 4, 5 !!!!!
data_path = "xxx/Documents/SamplingDL/"

ns=200
bias_array = np.zeros( (ns, 1) )
res_array_mean = np.zeros( (5,2))
nd=50
nfv=['relu', 'selu']
true_mean=5.017318
  
data_path2 = "xxxL/dat4_n200_pps/"
    
for nf in nfv:
    #if k==1000 and nd!=100:
    #    continue
    print( nf)
    
    fn_res = data_path + "Test3_pps_3L_"+nf+"_m4.txt"
    
    model = Sequential()
    model.add(Dense(nd, input_dim=20, activation=nf))
    model.add(Dense(nd, activation=nf))
    model.add(Dense(1, activation='linear', kernel_initializer='glorot_normal'))
    learning_rate = .1
    opt = optimizers.SGD(learning_rate = learning_rate)
    #opt = optimizers.Adam(learning_rate = learning_rate)
    # Compile model
    model.compile(loss='mean_squared_error', optimizer=opt, metrics=['mse'])
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=15)
    mc = ModelCheckpoint('best_model.h5', monitor='val_loss', mode='min', verbose=1, save_best_only=True)
    
    for j in range(ns):
        i = j + 1
        full_path = data_path2 + "dtB_" + str(i) + ".csv"
        dat2 = pd.read_csv(full_path)
        # Data needs to be scaled to a small range like 0 to 1 for the neural
        # network to work well.
        scaler = StandardScaler()
        new_columns=dat2.columns.values
        new_columns[0]="syB"
        dat2.columns=new_columns
        X = dat2.drop('syB', axis=1).values
        Y = dat2[['syB']].values
        scaler.fit(Y)
        Y_scaled = scaler.transform(Y)
        # Y
        # scaler.inverse_transform(Y_scaled)
        # Scale both the training inputs and outputs
        # scaled_training = scaler.fit_transform(training_data_df)
        # scaled_testing = scaler.transform(test_data_df)
        # Define the model
        # ADD / SUBTRACT number of layers (try 3, 4, 5 output included)
        # Change learning rate each time (range: .1, .01, .001, .0001, .00001)
        # Train the model
        model.fit(
            X,
            Y_scaled,
            # batch_size = 32,
            validation_split = .25,
            epochs = 200,
            shuffle = True,
            callbacks = [es, mc],
            verbose = 2
        )
        # Call model
        saved_model = load_model('best_model.h5')
        # Extract Y hats from saved_model
        full_A_path = data_path2 + "dtA_" + str(i) + ".csv"
        test_data = pd.read_csv(full_A_path)
        Xt = test_data.values[:,1:21]
        Yp = saved_model.predict(Xt)
        Yp_unscaled = scaler.inverse_transform(Yp)
        bias_array[j,0]=Yp_unscaled.mean()-(true_mean)
    
    
    bias_array.mean()
    print(np.mean(bias_array,axis=0))
    print(np.std(bias_array,axis=0))
    #np.savetxt(fn_res, bias_array)
    res_array_mean[0,0]=np.mean(bias_array,axis=0)[0]
    res_array_mean[0,1]=np.std(bias_array,axis=0)[0]
    
    model = Sequential()
    model.add(Dense(nd, input_dim=20, activation=nf))
    model.add(Dense(nd, activation=nf))
    model.add(Dense(1, activation='linear', kernel_initializer='glorot_normal'))
    learning_rate = .01
    opt = optimizers.SGD(learning_rate = learning_rate)
    #opt = optimizers.Adam(learning_rate = learning_rate)
    # Compile model
    model.compile(loss='mean_squared_error', optimizer=opt, metrics=['mse'])
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=15)
    mc = ModelCheckpoint('best_model.h5', monitor='val_loss', mode='min', verbose=1, save_best_only=True)
    
    for j in range(ns):
        i = j + 1
        full_path = data_path2 + "dtB_" + str(i) + ".csv"
        dat2 = pd.read_csv(full_path)
        # Data needs to be scaled to a small range like 0 to 1 for the neural
        # network to work well.
        scaler = StandardScaler()
        new_columns=dat2.columns.values
        new_columns[0]="syB"
        dat2.columns=new_columns
        X = dat2.drop('syB', axis=1).values
        Y = dat2[['syB']].values
        scaler.fit(Y)
        Y_scaled = scaler.transform(Y)
        # Y
        # scaler.inverse_transform(Y_scaled)
        # Scale both the training inputs and outputs
        # scaled_training = scaler.fit_transform(training_data_df)
        # scaled_testing = scaler.transform(test_data_df)
        # Define the model
        # ADD / SUBTRACT number of layers (try 3, 4, 5 output included)
        # Change learning rate each time (range: .1, .01, .001, .0001, .00001)
        # Train the model
        model.fit(
            X,
            Y_scaled,
            # batch_size = 32,
            validation_split = .25,
            epochs = 200,
            shuffle = True,
            callbacks = [es, mc],
            verbose = 2
        )
        # Call model
        saved_model = load_model('best_model.h5')
        # Extract Y hats from saved_model
        full_A_path = data_path2 + "dtA_" + str(i) + ".csv"
        test_data = pd.read_csv(full_A_path)
        Xt = test_data.values[:,1:21]
        Yp = saved_model.predict(Xt)
        Yp_unscaled = scaler.inverse_transform(Yp)
        bias_array[j,0]=Yp_unscaled.mean()-(true_mean)
    
    bias_array.mean()
    print(np.mean(bias_array,axis=0))
    print(np.std(bias_array,axis=0))
    #np.savetxt(fn_res, bias_array)
    res_array_mean[1,0]=np.mean(bias_array,axis=0)[0]
    res_array_mean[1,1]=np.std(bias_array,axis=0)[0]
    
    
    model = Sequential()
    model.add(Dense(nd, input_dim=20, activation=nf))
    model.add(Dense(nd, activation=nf))
    model.add(Dense(1, activation='linear', kernel_initializer='glorot_normal'))
    learning_rate = .001
    opt = optimizers.SGD(learning_rate = learning_rate)
    #opt = optimizers.Adam(learning_rate = learning_rate)
    # Compile model
    model.compile(loss='mean_squared_error', optimizer=opt, metrics=['mse'])
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=15)
    mc = ModelCheckpoint('best_model.h5', monitor='val_loss', mode='min', verbose=1, save_best_only=True)
    
    for j in range(ns):
        i = j + 1
        full_path = data_path2 + "dtB_" + str(i) + ".csv"
        dat2 = pd.read_csv(full_path)
        # Data needs to be scaled to a small range like 0 to 1 for the neural
        # network to work well.
        scaler = StandardScaler()
        new_columns=dat2.columns.values
        new_columns[0]="syB"
        dat2.columns=new_columns
        X = dat2.drop('syB', axis=1).values
        Y = dat2[['syB']].values
        scaler.fit(Y)
        Y_scaled = scaler.transform(Y)
        # Y
        # scaler.inverse_transform(Y_scaled)
        # Scale both the training inputs and outputs
        # scaled_training = scaler.fit_transform(training_data_df)
        # scaled_testing = scaler.transform(test_data_df)
        # Define the model
        # ADD / SUBTRACT number of layers (try 3, 4, 5 output included)
        # Change learning rate each time (range: .1, .01, .001, .0001, .00001)
        # Train the model
        model.fit(
            X,
            Y_scaled,
            # batch_size = 32,
            validation_split = .25,
            epochs = 200,
            shuffle = True,
            callbacks = [es, mc],
            verbose = 2
        )
        # Call model
        saved_model = load_model('best_model.h5')
        # Extract Y hats from saved_model
        full_A_path = data_path2 + "dtA_" + str(i) + ".csv"
        test_data = pd.read_csv(full_A_path)
        Xt = test_data.values[:,1:21]
        Yp = saved_model.predict(Xt)
        Yp_unscaled = scaler.inverse_transform(Yp)
        bias_array[j,0]=Yp_unscaled.mean()-(true_mean)
    
    
    bias_array.mean()
    print(np.mean(bias_array,axis=0))
    print(np.std(bias_array,axis=0))
    res_array_mean[2,0]=np.mean(bias_array,axis=0)[0]
    res_array_mean[2,1]=np.std(bias_array,axis=0)[0]
    
    
    model = Sequential()
    model.add(Dense(nd, input_dim=20, activation=nf))
    model.add(Dense(nd, activation=nf))
    model.add(Dense(1, activation='linear', kernel_initializer='glorot_normal'))
    learning_rate = .0001
    opt = optimizers.SGD(learning_rate = learning_rate)
    #opt = optimizers.Adam(learning_rate = learning_rate)
    # Compile model
    model.compile(loss='mean_squared_error', optimizer=opt, metrics=['mse'])
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=15)
    mc = ModelCheckpoint('best_model.h5', monitor='val_loss', mode='min', verbose=1, save_best_only=True)
    
    for j in range(ns):
        i = j + 1
        full_path = data_path2 + "dtB_" + str(i) + ".csv"
        dat2 = pd.read_csv(full_path)
        # Data needs to be scaled to a small range like 0 to 1 for the neural
        # network to work well.
        scaler = StandardScaler()
        new_columns=dat2.columns.values
        new_columns[0]="syB"
        dat2.columns=new_columns
        X = dat2.drop('syB', axis=1).values
        Y = dat2[['syB']].values
        scaler.fit(Y)
        Y_scaled = scaler.transform(Y)
        # Y
        # scaler.inverse_transform(Y_scaled)
        # Scale both the training inputs and outputs
        # scaled_training = scaler.fit_transform(training_data_df)
        # scaled_testing = scaler.transform(test_data_df)
        # Define the model
        # ADD / SUBTRACT number of layers (try 3, 4, 5 output included)
        # Change learning rate each time (range: .1, .01, .001, .0001, .00001)
        # Train the model
        model.fit(
            X,
            Y_scaled,
            # batch_size = 32,
            validation_split = .25,
            epochs = 200,
            shuffle = True,
            callbacks = [es, mc],
            verbose = 2
        )
        # Call model
        saved_model = load_model('best_model.h5')
        # Extract Y hats from saved_model
        full_A_path = data_path2 + "dtA_" + str(i) + ".csv"
        test_data = pd.read_csv(full_A_path)
        Xt = test_data.values[:,1:21]
        Yp = saved_model.predict(Xt)
        Yp_unscaled = scaler.inverse_transform(Yp)
        bias_array[j,0]=Yp_unscaled.mean()-(true_mean)
    
    
    bias_array.mean()
    print(np.mean(bias_array,axis=0))
    print(np.std(bias_array,axis=0))
    res_array_mean[3,0]=np.mean(bias_array,axis=0)[0]
    res_array_mean[3,1]=np.std(bias_array,axis=0)[0]
    
    
    model = Sequential()
    model.add(Dense(nd, input_dim=20, activation=nf))
    model.add(Dense(nd, activation=nf))
    model.add(Dense(1, activation='linear', kernel_initializer='glorot_normal'))
    learning_rate = .00001
    opt = optimizers.SGD(learning_rate = learning_rate)
    #opt = optimizers.Adam(learning_rate = learning_rate)
    # Compile model
    model.compile(loss='mean_squared_error', optimizer=opt, metrics=['mse'])
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=15)
    mc = ModelCheckpoint('best_model.h5', monitor='val_loss', mode='min', verbose=1, save_best_only=True)
    
    for j in range(ns):
        i = j + 1
        full_path = data_path2 + "dtB_" + str(i) + ".csv"
        dat2 = pd.read_csv(full_path)
        # Data needs to be scaled to a small range like 0 to 1 for the neural
        # network to work well.
        scaler = StandardScaler()
        new_columns=dat2.columns.values
        new_columns[0]="syB"
        dat2.columns=new_columns
        X = dat2.drop('syB', axis=1).values
        Y = dat2[['syB']].values
        scaler.fit(Y)
        Y_scaled = scaler.transform(Y)
        # Y
        # scaler.inverse_transform(Y_scaled)
        # Scale both the training inputs and outputs
        # scaled_training = scaler.fit_transform(training_data_df)
        # scaled_testing = scaler.transform(test_data_df)
        # Define the model
        # ADD / SUBTRACT number of layers (try 3, 4, 5 output included)
        # Change learning rate each time (range: .1, .01, .001, .0001, .00001)
        # Train the model
        model.fit(
            X,
            Y_scaled,
            # batch_size = 32,
            validation_split = .25,
            epochs = 200,
            shuffle = True,
            callbacks = [es, mc],
            verbose = 2
        )
        # Call model
        saved_model = load_model('best_model.h5')
        # Extract Y hats from saved_model
        full_A_path = data_path2 + "dtA_" + str(i) + ".csv"
        test_data = pd.read_csv(full_A_path)
        Xt = test_data.values[:,1:21]
        Yp = saved_model.predict(Xt)
        Yp_unscaled = scaler.inverse_transform(Yp)
        bias_array[j,0]=Yp_unscaled.mean()-(true_mean)
    
    
    bias_array.mean()
    print(np.mean(bias_array,axis=0))
    print(np.std(bias_array,axis=0))
    res_array_mean[4,0]=np.mean(bias_array,axis=0)[0]
    res_array_mean[4,1]=np.std(bias_array,axis=0)[0]
    
    
    np.savetxt(fn_res, res_array_mean)
    
    
    ####4L
    fn_res = data_path + "Test3_pps_4L_"+nf+"_m4.txt"
    
    model = Sequential()
    model.add(Dense(nd, input_dim=20, activation=nf))
    model.add(Dense(nd, activation=nf))
    model.add(Dense(nd, activation=nf))
    model.add(Dense(1, activation='linear', kernel_initializer='glorot_normal'))
    learning_rate = .1
    opt = optimizers.SGD(learning_rate = learning_rate)
    #opt = optimizers.Adam(learning_rate = learning_rate)
    # Compile model
    model.compile(loss='mean_squared_error', optimizer=opt, metrics=['mse'])
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=15)
    mc = ModelCheckpoint('best_model.h5', monitor='val_loss', mode='min', verbose=1, save_best_only=True)
    
    for j in range(ns):
        i = j + 1
        full_path = data_path2 + "dtB_" + str(i) + ".csv"
        dat2 = pd.read_csv(full_path)
        # Data needs to be scaled to a small range like 0 to 1 for the neural
        # network to work well.
        scaler = StandardScaler()
        new_columns=dat2.columns.values
        new_columns[0]="syB"
        dat2.columns=new_columns
        X = dat2.drop('syB', axis=1).values
        Y = dat2[['syB']].values
        scaler.fit(Y)
        Y_scaled = scaler.transform(Y)
        # Y
        # scaler.inverse_transform(Y_scaled)
        # Scale both the training inputs and outputs
        # scaled_training = scaler.fit_transform(training_data_df)
        # scaled_testing = scaler.transform(test_data_df)
        # Define the model
        # ADD / SUBTRACT number of layers (try 3, 4, 5 output included)
        # Change learning rate each time (range: .1, .01, .001, .0001, .00001)
        # Train the model
        model.fit(
            X,
            Y_scaled,
            # batch_size = 32,
            validation_split = .25,
            epochs = 200,
            shuffle = True,
            callbacks = [es, mc],
            verbose = 2
        )
        # Call model
        saved_model = load_model('best_model.h5')
        # Extract Y hats from saved_model
        full_A_path = data_path2 + "dtA_" + str(i) + ".csv"
        test_data = pd.read_csv(full_A_path)
        Xt = test_data.values[:,1:21]
        Yp = saved_model.predict(Xt)
        Yp_unscaled = scaler.inverse_transform(Yp)
        bias_array[j,0]=Yp_unscaled.mean()-(true_mean)
    
    
    bias_array.mean()
    print(np.mean(bias_array,axis=0))
    print(np.std(bias_array,axis=0))
    #np.savetxt(fn_res, bias_array)
    res_array_mean[0,0]=np.mean(bias_array,axis=0)[0]
    res_array_mean[0,1]=np.std(bias_array,axis=0)[0]
    
    model = Sequential()
    model.add(Dense(nd, input_dim=20, activation=nf))
    model.add(Dense(nd, activation=nf))
    model.add(Dense(nd, activation=nf))
    model.add(Dense(1, activation='linear', kernel_initializer='glorot_normal'))
    learning_rate = .01
    opt = optimizers.SGD(learning_rate = learning_rate)
    #opt = optimizers.Adam(learning_rate = learning_rate)
    # Compile model
    model.compile(loss='mean_squared_error', optimizer=opt, metrics=['mse'])
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=15)
    mc = ModelCheckpoint('best_model.h5', monitor='val_loss', mode='min', verbose=1, save_best_only=True)
    
    for j in range(ns):
        i = j + 1
        full_path = data_path2 + "dtB_" + str(i) + ".csv"
        dat2 = pd.read_csv(full_path)
        # Data needs to be scaled to a small range like 0 to 1 for the neural
        # network to work well.
        scaler = StandardScaler()
        new_columns=dat2.columns.values
        new_columns[0]="syB"
        dat2.columns=new_columns
        X = dat2.drop('syB', axis=1).values
        Y = dat2[['syB']].values
        scaler.fit(Y)
        Y_scaled = scaler.transform(Y)
        # Y
        # scaler.inverse_transform(Y_scaled)
        # Scale both the training inputs and outputs
        # scaled_training = scaler.fit_transform(training_data_df)
        # scaled_testing = scaler.transform(test_data_df)
        # Define the model
        # ADD / SUBTRACT number of layers (try 3, 4, 5 output included)
        # Change learning rate each time (range: .1, .01, .001, .0001, .00001)
        # Train the model
        model.fit(
            X,
            Y_scaled,
            # batch_size = 32,
            validation_split = .25,
            epochs = 200,
            shuffle = True,
            callbacks = [es, mc],
            verbose = 2
        )
        # Call model
        saved_model = load_model('best_model.h5')
        # Extract Y hats from saved_model
        full_A_path = data_path2 + "dtA_" + str(i) + ".csv"
        test_data = pd.read_csv(full_A_path)
        Xt = test_data.values[:,1:21]
        Yp = saved_model.predict(Xt)
        Yp_unscaled = scaler.inverse_transform(Yp)
        bias_array[j,0]=Yp_unscaled.mean()-(true_mean)
    
    
    bias_array.mean()
    print(np.mean(bias_array,axis=0))
    print(np.std(bias_array,axis=0))
    #np.savetxt(fn_res, bias_array)
    res_array_mean[1,0]=np.mean(bias_array,axis=0)[0]
    res_array_mean[1,1]=np.std(bias_array,axis=0)[0]
    
    
    model = Sequential()
    model.add(Dense(nd, input_dim=20, activation=nf))
    model.add(Dense(nd, activation=nf))
    model.add(Dense(nd, activation=nf))
    model.add(Dense(1, activation='linear', kernel_initializer='glorot_normal'))
    learning_rate = .001
    opt = optimizers.SGD(learning_rate = learning_rate)
    #opt = optimizers.Adam(learning_rate = learning_rate)
    # Compile model
    model.compile(loss='mean_squared_error', optimizer=opt, metrics=['mse'])
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=15)
    mc = ModelCheckpoint('best_model.h5', monitor='val_loss', mode='min', verbose=1, save_best_only=True)
    
    for j in range(ns):
        i = j + 1
        full_path = data_path2 + "dtB_" + str(i) + ".csv"
        dat2 = pd.read_csv(full_path)
        # Data needs to be scaled to a small range like 0 to 1 for the neural
        # network to work well.
        scaler = StandardScaler()
        new_columns=dat2.columns.values
        new_columns[0]="syB"
        dat2.columns=new_columns
        X = dat2.drop('syB', axis=1).values
        Y = dat2[['syB']].values
        scaler.fit(Y)
        Y_scaled = scaler.transform(Y)
        # Y
        # scaler.inverse_transform(Y_scaled)
        # Scale both the training inputs and outputs
        # scaled_training = scaler.fit_transform(training_data_df)
        # scaled_testing = scaler.transform(test_data_df)
        # Define the model
        # ADD / SUBTRACT number of layers (try 3, 4, 5 output included)
        # Change learning rate each time (range: .1, .01, .001, .0001, .00001)
        # Train the model
        model.fit(
            X,
            Y_scaled,
            # batch_size = 32,
            validation_split = .25,
            epochs = 200,
            shuffle = True,
            callbacks = [es, mc],
            verbose = 2
        )
        # Call model
        saved_model = load_model('best_model.h5')
        # Extract Y hats from saved_model
        full_A_path = data_path2 + "dtA_" + str(i) + ".csv"
        test_data = pd.read_csv(full_A_path)
        Xt = test_data.values[:,1:21]
        Yp = saved_model.predict(Xt)
        Yp_unscaled = scaler.inverse_transform(Yp)
        bias_array[j,0]=Yp_unscaled.mean()-(true_mean)
    
    
    bias_array.mean()
    print(np.mean(bias_array,axis=0))
    print(np.std(bias_array,axis=0))
    res_array_mean[2,0]=np.mean(bias_array,axis=0)[0]
    res_array_mean[2,1]=np.std(bias_array,axis=0)[0]
    
    
    model = Sequential()
    model.add(Dense(nd, input_dim=20, activation=nf))
    model.add(Dense(nd, activation=nf))
    model.add(Dense(nd, activation=nf))
    model.add(Dense(1, activation='linear', kernel_initializer='glorot_normal'))
    learning_rate = .0001
    opt = optimizers.SGD(learning_rate = learning_rate)
    #opt = optimizers.Adam(learning_rate = learning_rate)
    # Compile model
    model.compile(loss='mean_squared_error', optimizer=opt, metrics=['mse'])
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=15)
    mc = ModelCheckpoint('best_model.h5', monitor='val_loss', mode='min', verbose=1, save_best_only=True)
    
    for j in range(ns):
        i = j + 1
        full_path = data_path2 + "dtB_" + str(i) + ".csv"
        dat2 = pd.read_csv(full_path)
        # Data needs to be scaled to a small range like 0 to 1 for the neural
        # network to work well.
        scaler = StandardScaler()
        new_columns=dat2.columns.values
        new_columns[0]="syB"
        dat2.columns=new_columns
        X = dat2.drop('syB', axis=1).values
        Y = dat2[['syB']].values
        scaler.fit(Y)
        Y_scaled = scaler.transform(Y)
        # Y
        # scaler.inverse_transform(Y_scaled)
        # Scale both the training inputs and outputs
        # scaled_training = scaler.fit_transform(training_data_df)
        # scaled_testing = scaler.transform(test_data_df)
        # Define the model
        # ADD / SUBTRACT number of layers (try 3, 4, 5 output included)
        # Change learning rate each time (range: .1, .01, .001, .0001, .00001)
        # Train the model
        model.fit(
            X,
            Y_scaled,
            # batch_size = 32,
            validation_split = .25,
            epochs = 200,
            shuffle = True,
            callbacks = [es, mc],
            verbose = 2
        )
        # Call model
        saved_model = load_model('best_model.h5')
        # Extract Y hats from saved_model
        full_A_path = data_path2 + "dtA_" + str(i) + ".csv"
        test_data = pd.read_csv(full_A_path)
        Xt = test_data.values[:,1:21]
        Yp = saved_model.predict(Xt)
        Yp_unscaled = scaler.inverse_transform(Yp)
        bias_array[j,0]=Yp_unscaled.mean()-(true_mean)
    
    
    bias_array.mean()
    print(np.mean(bias_array,axis=0))
    print(np.std(bias_array,axis=0))
    res_array_mean[3,0]=np.mean(bias_array,axis=0)[0]
    res_array_mean[3,1]=np.std(bias_array,axis=0)[0]
    
    
    model = Sequential()
    model.add(Dense(nd, input_dim=20, activation=nf))
    model.add(Dense(nd, activation=nf))
    model.add(Dense(nd, activation=nf))
    model.add(Dense(1, activation='linear', kernel_initializer='glorot_normal'))
    learning_rate = .00001
    opt = optimizers.SGD(learning_rate = learning_rate)
    #opt = optimizers.Adam(learning_rate = learning_rate)
    # Compile model
    model.compile(loss='mean_squared_error', optimizer=opt, metrics=['mse'])
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=15)
    mc = ModelCheckpoint('best_model.h5', monitor='val_loss', mode='min', verbose=1, save_best_only=True)
    
    for j in range(ns):
        i = j + 1
        full_path = data_path2 + "dtB_" + str(i) + ".csv"
        dat2 = pd.read_csv(full_path)
        # Data needs to be scaled to a small range like 0 to 1 for the neural
        # network to work well.
        scaler = StandardScaler()
        new_columns=dat2.columns.values
        new_columns[0]="syB"
        dat2.columns=new_columns
        X = dat2.drop('syB', axis=1).values
        Y = dat2[['syB']].values
        scaler.fit(Y)
        Y_scaled = scaler.transform(Y)
        # Y
        # scaler.inverse_transform(Y_scaled)
        # Scale both the training inputs and outputs
        # scaled_training = scaler.fit_transform(training_data_df)
        # scaled_testing = scaler.transform(test_data_df)
        # Define the model
        # ADD / SUBTRACT number of layers (try 3, 4, 5 output included)
        # Change learning rate each time (range: .1, .01, .001, .0001, .00001)
        # Train the model
        model.fit(
            X,
            Y_scaled,
            # batch_size = 32,
            validation_split = .25,
            epochs = 200,
            shuffle = True,
            callbacks = [es, mc],
            verbose = 2
        )
        # Call model
        saved_model = load_model('best_model.h5')
        # Extract Y hats from saved_model
        full_A_path = data_path2 + "dtA_" + str(i) + ".csv"
        test_data = pd.read_csv(full_A_path)
        Xt = test_data.values[:,1:21]
        Yp = saved_model.predict(Xt)
        Yp_unscaled = scaler.inverse_transform(Yp)
        bias_array[j,0]=Yp_unscaled.mean()-(true_mean)
    
    
    bias_array.mean()
    print(np.mean(bias_array,axis=0))
    print(np.std(bias_array,axis=0))
    res_array_mean[4,0]=np.mean(bias_array,axis=0)[0]
    res_array_mean[4,1]=np.std(bias_array,axis=0)[0]
    
    np.savetxt(fn_res, res_array_mean)
    
    
    ####5L
    fn_res = data_path + "Test3_pps_5L_"+nf+"_m4.txt"

    
    model = Sequential()
    model.add(Dense(nd, input_dim=20, activation=nf))
    model.add(Dense(nd, activation=nf))
    model.add(Dense(nd, activation=nf))
    model.add(Dense(nd, activation=nf))
    model.add(Dense(1, activation='linear', kernel_initializer='glorot_normal'))
    learning_rate = .1
    opt = optimizers.SGD(learning_rate = learning_rate)
    #opt = optimizers.Adam(learning_rate = learning_rate)
    # Compile model
    model.compile(loss='mean_squared_error', optimizer=opt, metrics=['mse'])
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=15)
    mc = ModelCheckpoint('best_model.h5', monitor='val_loss', mode='min', verbose=1, save_best_only=True)
    
    for j in range(ns):
        i = j + 1
        full_path = data_path2 + "dtB_" + str(i) + ".csv"
        dat2 = pd.read_csv(full_path)
        # Data needs to be scaled to a small range like 0 to 1 for the neural
        # network to work well.
        scaler = StandardScaler()
        new_columns=dat2.columns.values
        new_columns[0]="syB"
        dat2.columns=new_columns
        X = dat2.drop('syB', axis=1).values
        Y = dat2[['syB']].values
        scaler.fit(Y)
        Y_scaled = scaler.transform(Y)
        # Y
        # scaler.inverse_transform(Y_scaled)
        # Scale both the training inputs and outputs
        # scaled_training = scaler.fit_transform(training_data_df)
        # scaled_testing = scaler.transform(test_data_df)
        # Define the model
        # ADD / SUBTRACT number of layers (try 3, 4, 5 output included)
        # Change learning rate each time (range: .1, .01, .001, .0001, .00001)
        # Train the model
        model.fit(
            X,
            Y_scaled,
            # batch_size = 32,
            validation_split = .25,
            epochs = 200,
            shuffle = True,
            callbacks = [es, mc],
            verbose = 2
        )
        # Call model
        saved_model = load_model('best_model.h5')
        # Extract Y hats from saved_model
        full_A_path = data_path2 + "dtA_" + str(i) + ".csv"
        test_data = pd.read_csv(full_A_path)
        Xt = test_data.values[:,1:21]
        Yp = saved_model.predict(Xt)
        Yp_unscaled = scaler.inverse_transform(Yp)
        bias_array[j,0]=Yp_unscaled.mean()-(true_mean)
    
    
    bias_array.mean()
    print(np.mean(bias_array,axis=0))
    print(np.std(bias_array,axis=0))
    #np.savetxt(fn_res, bias_array)
    res_array_mean[0,0]=np.mean(bias_array,axis=0)[0]
    res_array_mean[0,1]=np.std(bias_array,axis=0)[0]
    
    model = Sequential()
    model.add(Dense(nd, input_dim=20, activation=nf))
    model.add(Dense(nd, activation=nf))
    model.add(Dense(nd, activation=nf))
    model.add(Dense(nd, activation=nf))
    model.add(Dense(1, activation='linear', kernel_initializer='glorot_normal'))
    learning_rate = .01
    opt = optimizers.SGD(learning_rate = learning_rate)
    #opt = optimizers.Adam(learning_rate = learning_rate)
    # Compile model
    model.compile(loss='mean_squared_error', optimizer=opt, metrics=['mse'])
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=15)
    mc = ModelCheckpoint('best_model.h5', monitor='val_loss', mode='min', verbose=1, save_best_only=True)
    
    for j in range(ns):
        i = j + 1
        full_path = data_path2 + "dtB_" + str(i) + ".csv"
        dat2 = pd.read_csv(full_path)
        # Data needs to be scaled to a small range like 0 to 1 for the neural
        # network to work well.
        scaler = StandardScaler()
        new_columns=dat2.columns.values
        new_columns[0]="syB"
        dat2.columns=new_columns
        X = dat2.drop('syB', axis=1).values
        Y = dat2[['syB']].values
        scaler.fit(Y)
        Y_scaled = scaler.transform(Y)
        # Y
        # scaler.inverse_transform(Y_scaled)
        # Scale both the training inputs and outputs
        # scaled_training = scaler.fit_transform(training_data_df)
        # scaled_testing = scaler.transform(test_data_df)
        # Define the model
        # ADD / SUBTRACT number of layers (try 3, 4, 5 output included)
        # Change learning rate each time (range: .1, .01, .001, .0001, .00001)
        # Train the model
        model.fit(
            X,
            Y_scaled,
            # batch_size = 32,
            validation_split = .25,
            epochs = 200,
            shuffle = True,
            callbacks = [es, mc],
            verbose = 2
        )
        # Call model
        saved_model = load_model('best_model.h5')
        # Extract Y hats from saved_model
        full_A_path = data_path2 + "dtA_" + str(i) + ".csv"
        test_data = pd.read_csv(full_A_path)
        Xt = test_data.values[:,1:21]
        Yp = saved_model.predict(Xt)
        Yp_unscaled = scaler.inverse_transform(Yp)
        bias_array[j,0]=Yp_unscaled.mean()-(true_mean)
    
    
    bias_array.mean()
    print(np.mean(bias_array,axis=0))
    print(np.std(bias_array,axis=0))
    #np.savetxt(fn_res, bias_array)
    res_array_mean[1,0]=np.mean(bias_array,axis=0)[0]
    res_array_mean[1,1]=np.std(bias_array,axis=0)[0]
    
    
    model = Sequential()
    model.add(Dense(nd, input_dim=20, activation=nf))
    model.add(Dense(nd, activation=nf))
    model.add(Dense(nd, activation=nf))
    model.add(Dense(nd, activation=nf))
    model.add(Dense(1, activation='linear', kernel_initializer='glorot_normal'))
    learning_rate = .001
    opt = optimizers.SGD(learning_rate = learning_rate)
    #opt = optimizers.Adam(learning_rate = learning_rate)
    # Compile model
    model.compile(loss='mean_squared_error', optimizer=opt, metrics=['mse'])
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=15)
    mc = ModelCheckpoint('best_model.h5', monitor='val_loss', mode='min', verbose=1, save_best_only=True)
    
    for j in range(ns):
        i = j + 1
        full_path = data_path2 + "dtB_" + str(i) + ".csv"
        dat2 = pd.read_csv(full_path)
        # Data needs to be scaled to a small range like 0 to 1 for the neural
        # network to work well.
        scaler = StandardScaler()
        new_columns=dat2.columns.values
        new_columns[0]="syB"
        dat2.columns=new_columns
        X = dat2.drop('syB', axis=1).values
        Y = dat2[['syB']].values
        scaler.fit(Y)
        Y_scaled = scaler.transform(Y)
        # Y
        # scaler.inverse_transform(Y_scaled)
        # Scale both the training inputs and outputs
        # scaled_training = scaler.fit_transform(training_data_df)
        # scaled_testing = scaler.transform(test_data_df)
        # Define the model
        # ADD / SUBTRACT number of layers (try 3, 4, 5 output included)
        # Change learning rate each time (range: .1, .01, .001, .0001, .00001)
        # Train the model
        model.fit(
            X,
            Y_scaled,
            # batch_size = 32,
            validation_split = .25,
            epochs = 200,
            shuffle = True,
            callbacks = [es, mc],
            verbose = 2
        )
        # Call model
        saved_model = load_model('best_model.h5')
        # Extract Y hats from saved_model
        full_A_path = data_path2 + "dtA_" + str(i) + ".csv"
        test_data = pd.read_csv(full_A_path)
        Xt = test_data.values[:,1:21]
        Yp = saved_model.predict(Xt)
        Yp_unscaled = scaler.inverse_transform(Yp)
        bias_array[j,0]=Yp_unscaled.mean()-(true_mean)
    
    
    bias_array.mean()
    print(np.mean(bias_array,axis=0))
    print(np.std(bias_array,axis=0))
    res_array_mean[2,0]=np.mean(bias_array,axis=0)[0]
    res_array_mean[2,1]=np.std(bias_array,axis=0)[0]
    
    
    model = Sequential()
    model.add(Dense(nd, input_dim=20, activation=nf))
    model.add(Dense(nd, activation=nf))
    model.add(Dense(nd, activation=nf))
    model.add(Dense(nd, activation=nf))
    model.add(Dense(1, activation='linear', kernel_initializer='glorot_normal'))
    learning_rate = .0001
    opt = optimizers.SGD(learning_rate = learning_rate)
    #opt = optimizers.Adam(learning_rate = learning_rate)
    # Compile model
    model.compile(loss='mean_squared_error', optimizer=opt, metrics=['mse'])
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=15)
    mc = ModelCheckpoint('best_model.h5', monitor='val_loss', mode='min', verbose=1, save_best_only=True)
    
    for j in range(ns):
        i = j + 1
        full_path = data_path2 + "dtB_" + str(i) + ".csv"
        dat2 = pd.read_csv(full_path)
        # Data needs to be scaled to a small range like 0 to 1 for the neural
        # network to work well.
        scaler = StandardScaler()
        new_columns=dat2.columns.values
        new_columns[0]="syB"
        dat2.columns=new_columns
        X = dat2.drop('syB', axis=1).values
        Y = dat2[['syB']].values
        scaler.fit(Y)
        Y_scaled = scaler.transform(Y)
        # Y
        # scaler.inverse_transform(Y_scaled)
        # Scale both the training inputs and outputs
        # scaled_training = scaler.fit_transform(training_data_df)
        # scaled_testing = scaler.transform(test_data_df)
        # Define the model
        # ADD / SUBTRACT number of layers (try 3, 4, 5 output included)
        # Change learning rate each time (range: .1, .01, .001, .0001, .00001)
        # Train the model
        model.fit(
            X,
            Y_scaled,
            # batch_size = 32,
            validation_split = .25,
            epochs = 200,
            shuffle = True,
            callbacks = [es, mc],
            verbose = 2
        )
        # Call model
        saved_model = load_model('best_model.h5')
        # Extract Y hats from saved_model
        full_A_path = data_path2 + "dtA_" + str(i) + ".csv"
        test_data = pd.read_csv(full_A_path)
        Xt = test_data.values[:,1:21]
        Yp = saved_model.predict(Xt)
        Yp_unscaled = scaler.inverse_transform(Yp)
        bias_array[j,0]=Yp_unscaled.mean()-(true_mean)
    
    
    bias_array.mean()
    print(np.mean(bias_array,axis=0))
    print(np.std(bias_array,axis=0))
    res_array_mean[3,0]=np.mean(bias_array,axis=0)[0]
    res_array_mean[3,1]=np.std(bias_array,axis=0)[0]
    
    
    model = Sequential()
    model.add(Dense(nd, input_dim=20, activation=nf))
    model.add(Dense(nd, activation=nf))
    model.add(Dense(nd, activation=nf))
    model.add(Dense(nd, activation=nf))
    model.add(Dense(1, activation='linear', kernel_initializer='glorot_normal'))
    learning_rate = .00001
    opt = optimizers.SGD(learning_rate = learning_rate)
    #opt = optimizers.Adam(learning_rate = learning_rate)
    # Compile model
    model.compile(loss='mean_squared_error', optimizer=opt, metrics=['mse'])
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=15)
    mc = ModelCheckpoint('best_model.h5', monitor='val_loss', mode='min', verbose=1, save_best_only=True)
    
    for j in range(ns):
        i = j + 1
        full_path = data_path2 + "dtB_" + str(i) + ".csv"
        dat2 = pd.read_csv(full_path)
        # Data needs to be scaled to a small range like 0 to 1 for the neural
        # network to work well.
        scaler = StandardScaler()
        new_columns=dat2.columns.values
        new_columns[0]="syB"
        dat2.columns=new_columns
        X = dat2.drop('syB', axis=1).values
        Y = dat2[['syB']].values
        scaler.fit(Y)
        Y_scaled = scaler.transform(Y)
        # Y
        # scaler.inverse_transform(Y_scaled)
        # Scale both the training inputs and outputs
        # scaled_training = scaler.fit_transform(training_data_df)
        # scaled_testing = scaler.transform(test_data_df)
        # Define the model
        # ADD / SUBTRACT number of layers (try 3, 4, 5 output included)
        # Change learning rate each time (range: .1, .01, .001, .0001, .00001)
        # Train the model
        model.fit(
            X,
            Y_scaled,
            # batch_size = 32,
            validation_split = .25,
            epochs = 200,
            shuffle = True,
            callbacks = [es, mc],
            verbose = 2
        )
        # Call model
        saved_model = load_model('best_model.h5')
        # Extract Y hats from saved_model
        full_A_path = data_path2 + "dtA_" + str(i) + ".csv"
        test_data = pd.read_csv(full_A_path)
        Xt = test_data.values[:,1:21]
        Yp = saved_model.predict(Xt)
        Yp_unscaled = scaler.inverse_transform(Yp)
        bias_array[j,0]=Yp_unscaled.mean()-(true_mean)
    
    
    bias_array.mean()
    print(np.mean(bias_array,axis=0))
    print(np.std(bias_array,axis=0))
    res_array_mean[4,0]=np.mean(bias_array,axis=0)[0]
    res_array_mean[4,1]=np.std(bias_array,axis=0)[0]
    
    np.savetxt(fn_res, res_array_mean)
