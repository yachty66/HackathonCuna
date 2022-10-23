from sklearn.preprocessing import StandardScaler
import torch
from torch import nn
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
    # scale data with standard scaler
    """x_train = (x_train - x_train.mean()) / x_train.std()
    y_train = (y_train - y_train.mean()) / y_train.std()
    x_test = (x_test - x_test.mean()) / x_test.std()
    y_test = (y_test - y_test.mean()) / y_test.std()"""
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


def LSTM():
    dfX = pd.read_csv("data/preprocessed_dataX_merged_eval.csv")
    dfY = pd.read_csv("data/preprocessed_dataY_merged_eval.csv")

    dfX = dfX.drop(dfX.columns[0], axis=1)

    x_train, x_test, y_train, y_test = train_test_split(
        dfX, dfY, test_size=0.2, random_state=0
    )

    '''scalerX = StandardScaler().fit(x_train)
    scalerY = StandardScaler().fit(y_train)
    x_train = scalerX.transform(x_train)
    y_train = scalerY.transform(y_train)
    x_test = scalerX.transform(x_test)'''
    # y_test = scalerY.transform(y_test)

    # convert data to torch tensors
    #make x_train fit itnt t
    x_train = torch.randn(2, 3, 20)
    #y_train = torch.from_numpy(y_train).float()
    #x_test = torch.from_numpy(x_test).float()

    # create model
    model = nn.Sequential(
        nn.LSTM(input_size=20, hidden_size=50, num_layers=2),
        nn.Linear(50, 1),
    )
    
    # define loss function
    loss_fn = nn.MSELoss()
    # define optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    # train model
    num_epochs = 100
    for t in range(num_epochs):
        y_pred = model(x_train)
        loss = loss_fn(y_pred, y_train)
        if t % 10 == 9:
            print(t, loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    # evaluate model
    with torch.no_grad():
        y_pred = model(x_test)
        loss = loss_fn(y_pred, y_test)
        print(loss.item())
        print(mse(y_pred, y_test))
        print(y_pred)
        print(y_test)
    return y_pred, y_test


if __name__ == "__main__":
    LSTM()
