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


def multipleLinearRegression():
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
    x_train = torch.from_numpy(x_train).float()
    y_train = torch.from_numpy(y_train).float()
    x_test = torch.from_numpy(x_test).float()
    model = nn.Linear(21, 3)
    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    epochs = 1000
    for epoch in range(epochs):
        y_pred = model(x_train)
        loss = criterion(y_pred, y_train)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        if (epoch + 1) % 50 == 0:
            print(f"epoch: {epoch+1}, loss = {loss.item():.4f}")
    y_pred = model(x_test)
    y_pred = y_pred.float().detach().numpy()
    y_test = y_test.to_numpy()
    print("MSE: ", mse(y_test, y_pred))
    print("RMSE: ", np.sqrt(mse(y_test, y_pred)))
    
    
if __name__ == "__main__":
    multipleLinearRegression()
    