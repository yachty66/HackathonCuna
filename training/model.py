from sklearn.preprocessing import StandardScaler
import torch
from torch import dropout, nn
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.metrics import mean_squared_error as mse


def loadData():
    dfX = pd.read_csv("data/preprocessed_dataX.csv")
    dfY = pd.read_csv("data/preprocessed_dataY.csv")

    dfX = dfX.drop(dfX.columns[0], axis=1)

    x_train, x_test, y_train, y_test = train_test_split(
        dfX, dfY, test_size=0.2, random_state=0
    )
    
    scalerX = StandardScaler().fit(x_train)
    scalerY = StandardScaler().fit(y_train)
    x_train = scalerX.transform(x_train)
    y_train = scalerY.transform(y_train)
    x_test = scalerX.transform(x_test)
    y_test = scalerY.transform(y_test)

    x_train, y_train, x_test, y_test = (
        torch.from_numpy(x_train.values),
        torch.from_numpy(y_train.values),
        torch.from_numpy(x_test.values),
        torch.from_numpy(y_test.values),
    )
    return x_train, y_train, x_test, y_test


def multiLinearRegression():
    dfX = pd.read_csv("data/preprocessed_dataX_merged_eval.csv")
    dfY = pd.read_csv("data/preprocessed_dataY_merged_eval.csv")
    
    #drop randomly 5 columns from dataset and print name of this columns 
    

    dfX = dfX.drop(dfX.columns[0], axis=1)
    
    #get number of columns
    #select randomly 5 numbers from 0 to number of columns (without duplicates)

    #drop columns with this numbers and print name of this column
    x_train, x_test, y_train, y_test = train_test_split(
        dfX, dfY, test_size=0.2, random_state=0
    )
    scalerX = StandardScaler().fit(x_train)
    scalerY = StandardScaler().fit(y_train)
    x_train = scalerX.transform(x_train)
    y_train = scalerY.transform(y_train)
    x_test = scalerX.transform(x_test)

    x_train = torch.from_numpy(x_train).float()
    y_train = torch.from_numpy(y_train).float()
    x_test = torch.from_numpy(x_test).float()


    '''
    implement
    dense2 = Dense(l, activation = 'relu')(dense1)
    dense3 = Dense(4*l, activation = 'relu')(dense2)
    dense4 = Dense(16*l, activation = 'relu')(dense3)
    dense5 = Dense(16*l, activation = 'relu')(dense4)
    dense6 = Dense(4*l, activation = 'relu')(dense5)
    dense7 = Dense(l, activation = 'relu')(dense6)
    dense8 = Dense(3, activation = 'relu')(dense7)

    output = Dense(outputSize, activation = 'linear')(dense8)
    '''
    #print dataset as pd df 
    print(x_train)
    
    #create model with dense layers with relu activation function
    model = nn.Sequential(
        nn.Linear(21, 3)
        
        
        #nn.ReLU(3),
        #create dense layer    
    )
    
    
    #model = nn.Sequential(nn.Linear(21, 3))
    
    #adam, rmsprop, adamw
    

    criterion = nn.MSELoss()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    # x_train, y_train, x_test, y_test = loadData()

    test_model = model(x_test.float())
    print(test_model)

    # train model with x data and save model
    for epoch in range(300):
        y_pred = model(x_train.float())
        # see source for PyTorch RSME https://stackoverflow.com/a/61991258
        loss = torch.sqrt(criterion(y_pred, y_train.float()))
        print("epoch: ", epoch, "loss: ", loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    test_model = model(x_test.float()).detach().numpy()

    yInverse = scalerY.inverse_transform(test_model)
    
    rsme = np.sqrt(np.mean(np.power((y_test - yInverse), 2)))
    rsme = np.sqrt(mse(yInverse, y_test))
    print(rsme)


# call model
if __name__ == "__main__":
    multiLinearRegression()

    # loadData()