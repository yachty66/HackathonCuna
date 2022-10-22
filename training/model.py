import torch
from torch import nn
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
from yaml import load


def loadData():

    dfX = pd.read_csv("data/preprocessed_dataX.csv")
    dfY = pd.read_csv("data/preprocessed_dataY.csv")

    dfX = dfX.drop(dfX.columns[0], axis=1)

    x_train, x_test, y_train, y_test = train_test_split(
        dfX, dfY, test_size=0.2, random_state=0
    )
    # scale data with standard scaler
    x_train = (x_train - x_train.mean()) / x_train.std()
    y_train = (y_train - y_train.mean()) / y_train.std()

    x_train, y_train, x_test, y_test = (
        torch.from_numpy(x_train.values),
        torch.from_numpy(y_train.values),
        torch.from_numpy(x_test.values),
        torch.from_numpy(y_test.values),
    )

    return x_train, y_train, x_test, y_test


def multiLinearRegression():
    # scale data with standard scaler

    # add layer for 21 x variables and 3 y variables
    model = nn.Sequential(nn.Linear(21, 3))

    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    x_train, y_train, x_test, y_test = loadData()

    for epoch in range(100):
        y_pred = model(x_train.float())
        loss = torch.sqrt(criterion(y_pred, y_train.float()))
        print(f"Epoch: {epoch+1}/100 | Loss: {loss.item():.4f}")
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    # use test data to predict values
    y_pred = model(x_test.float())
    # print(y_pred)

    # print(y_test)


# call model
if __name__ == "__main__":
    multiLinearRegression()
    # loadData()
