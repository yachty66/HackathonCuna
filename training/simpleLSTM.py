
import torch
from torch import nn
import numpy as np




def LSTM():
    #dfX = pd.read_csv("data/preprocessed_dataX_merged_eval.csv")
    #dfY = pd.read_csv("data/preprocessed_dataY_merged_eval.csv")

    #dfX = dfX.drop(dfX.columns[0], axis=1)

    #x_train, x_test, y_train, y_test = train_test_split(
    #    dfX, dfY, test_size=0.2, random_state=0
    #)

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

    #y_test = torch.from_numpy(y_test).float()

    # create LSTM
    lstm = nn.LSTM(input_size=20, hidden_size=50, num_layers=1, batch_first=True)

    # initialize hidden state
    h0 = torch.randn(1, 2, 50)
    # initialize cell state
    c0 = torch.randn(1, 2, 50)

    # forward propagate LSTM
    out, (hn, cn) = lstm(x_train, (h0, c0))
    

    # make predictions


    # calculate loss
    #loss = criterion(y_pred, y_test)

    # backpropagate
    #loss.backward()

    # update parameters
    #optimizer.step()

    # print loss
    #if epoch % 100 == 0:
    #    print(f"epoch: {epoch}, loss = {loss.item():.5f}")

    # print predictions
    #print(f"predicted output: {y_pred}, \n actual output: {y_test}")

    #print(f"loss: {np.sqrt(mse(scalerY.inverse_transform(y_pred.detach().numpy()), scalerY.inverse_transform(y_test.detach().numpy())))}")


if __name__ == "__main__":
    LSTM()