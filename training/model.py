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


def multiLinearRegression():
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

    #model = nn.Sequential(nn.Linear(21, 3))
    #add another layer with neurons to the model
    #model = nn.Sequential(nn.Linear(21, 3), nn.Linear(3, 3))
    #model.add_module("relu", nn.ReLU())
    
    #create a model with three layers
    model = nn.Sequential(nn.Linear(21, 3), nn.Linear(3, 3), nn.Linear(3, 3))

    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    for epoch in range(100):
        y_pred = model(x_train.float())
        loss = torch.sqrt(criterion(y_pred, y_train.float()))
        print("epoch: ", epoch, "loss: ", loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    test_model = model(x_test.float()).detach().numpy()

    yInverse = scalerY.inverse_transform(test_model)

    #print(yInverse[:2])
    #print(y_test[:2])
    
    rsme = np.sqrt(np.mean(np.power((y_test - yInverse), 2)))
    rsme = np.sqrt(mse(yInverse, y_test))
    print(rsme)
    
    

    


if __name__ == "__main__":
    print(multiLinearRegression())
