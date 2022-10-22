from sklearn.preprocessing import StandardScaler
import torch
from torch import nn
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd


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
    dfX = pd.read_csv("data/preprocessed_dataX.csv")
    dfY = pd.read_csv("data/preprocessed_dataY.csv")

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
    '''x_train = torch.from_numpy(x_train).float()
    y_train = torch.from_numpy(y_train).float()
    x_test = torch.from_numpy(x_test).float()
    y_test = torch.from_numpy(y_test).float()'''

    """x_train = torch.detach().numpy()(x_train).float()
    y_train = torch.detach().numpy()(y_train).float()
    x_test = torch.detach().numpy()(x_test).float()
    y_test = torch.detach().numpy()(y_test).float()"""
    """x_train, y_train, x_test, y_test = (
        torch.from_numpy(x_train.values),
        torch.from_numpy(y_train.values),
        torch.from_numpy(x_test.values),
        torch.from_numpy(y_test.values),
    )"""
    
    x_train, y_train, x_test, y_test = (
        torch.from_numpy(x_train.values),
        torch.from_numpy(y_train.values),
        torch.from_numpy(x_test.values),
        torch.from_numpy(y_test.values),
    )

    # scale data with standard scaler

    # add layer for 21 x variables and 3 y variables
    model = nn.Sequential(nn.Linear(21, 3))

    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    # x_train, y_train, x_test, y_test = loadData()

    #test_model = model(x_test.float())
    #print(test_model)

    # train model with x data and save model
    for epoch in range(100):
        y_pred = model(x_train.float())
        # see source for PyTorch RSME https://stackoverflow.com/a/61991258
        loss = torch.sqrt(criterion(y_pred, y_train.float()))
        print("epoch: ", epoch, "loss: ", loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    #test_model = model(x_test.float()).detach().numpy()
    # print(y_test)
    #yInverse = scalerY.inverse_transform(test_model)

    # print first 2 rows from yInverse
    #print(yInverse[:2])

    # print(yInverse)
    # print first 2 rows from y_test
    #print(y_test[:2])

    # print(y_test)

    """for epoch in range(100):
        y_pred = model(x_train.float())
        loss = torch.sqrt(criterion(y_pred, y_train.float()))
        print(f"Epoch: {epoch+1}/100 | Loss: {loss.item():.4f}")
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()"""

    # test the trained model with test data
    # test_model = model(x_test.float())

    # use test data to predict values
    # y_pred = model(x_test.float())
    ##rescale data
    # scale data back
    # y_pred = y_pred * y_test.std() + y_test.mean()
    # print(y_pred)


# call model
if __name__ == "__main__":
    print(multiLinearRegression())

    # loadData()
