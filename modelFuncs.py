from torch.nn import Linear
from torch.nn import ReLU
from torch.nn import Module
from torch.nn import Dropout as torch_do
from torch.nn.init import kaiming_uniform_
from torch.nn.init import xavier_uniform_
from torch.nn import Softmax
from torch.optim import SGD
from torch.nn import CrossEntropyLoss
from torch import save as torch_save
from torch import load as torch_load

import numpy as np
from sklearn.metrics import accuracy_score

#SUBMIT_CHECK
def create_hidstate_dict(hid_state_cnt_vec, init_mode_vec = None, act_vec=None, verbose=0):
    hid_state_cnt = len(hid_state_cnt_vec)
    hidStatesDict = {}
    for i in range(hid_state_cnt):
        hid_state_id_str = str(i+1).zfill(2)
        hid_state_name = "hidStateDict_" + hid_state_id_str
        dim_str = str(hid_state_cnt_vec[i])
        dim_int = int(hid_state_cnt_vec[i])
        try:#if init_mode_vec is not None and len(init_mode_vec)>=i:
            initMode = init_mode_vec[i]
        except:#else:
            initMode = "kaiming_uniform_"
        try:#if act_vec is not None and len(act_vec)>=i:
            actStr = act_vec[i]
        except:#else:
            actStr = "relu"
        if verbose>0:
            print(hid_state_name, ' = {"dimOut": "', dim_str, '", "initMode": "', initMode ,'", "act": "', actStr,'"}')
        hidStatesDict[hid_state_id_str] = {"dimOut": dim_int, "initMode": initMode, "act": actStr}
    return hidStatesDict

class MLP_Dict(Module):
    # define model elements
    def __init__(self, dim_of_input, dict_hidStates, classCount, dropout_value=None):

        super(MLP_Dict, self).__init__()

        # first fetch the hidden state variables as keys
        keys = [key for key, value in dict_hidStates.items()]
        print(keys)
        keysSorted = np.sort(keys)

        self.dropout_value = dropout_value
        self.dim_of_input = dim_of_input
        self.classCount = classCount
        self.keysSorted = keysSorted
        self.dict_hidStates = dict_hidStates
        self.initialize_net()

    def initialize_net(self):
        i = 0
        dim_in = self.dim_of_input
        for k in self.keysSorted:
            i = i + 1
            actFun = self.dict_hidStates[k]['act']
            dim_out = self.dict_hidStates[k]['dimOut']
            initMode = self.dict_hidStates[k]['initMode']
            print(i, k, initMode, dim_out, actFun)

            print("  self.{:s} = Linear({:d}, {:d})".format("hidden" + str(i), dim_in, dim_out))
            setattr(self, "hidden" + str(i), Linear(dim_in, dim_out))

            print("  kaiming_uniform_(self.{:s}.weight,nonlinearity={:s})".format("hidden" + str(i), actFun))
            kaiming_uniform_(getattr(self, "hidden" + str(i)).weight, nonlinearity=actFun)

            print("  self.{:s} = ReLU()".format("act" + str(i)))
            setattr(self, "act" + str(i), ReLU())

            if self.dropout_value is not None:
                print("  self.{:s} = Dropout({:2.1f})".format("dropout" + str(i), self.dropout_value))
                setattr(self, "dropout" + str(i), torch_do(p=self.dropout_value))

            dim_in = dim_out

        print("  self.finalLayer = Linear({:d},{:d})".format(dim_in, self.classCount))
        self.finalLayer = Linear(dim_in, self.classCount)
        xavier_uniform_(getattr(self, "hidden" + str(i)).weight)

    def forward(self, X):
        for i in range(len(self.keysSorted)):
            fhid = getattr(self, "hidden" + str(i+1))
            fact = getattr(self, "act" + str(i+1))
            X = fhid(X)
            X = fact(X)
        X = self.finalLayer(X)
        return X

    def load_model(self, model_file_full_path):
        print("loading parameters from : ", model_file_full_path)
        modelParams = torch_load(model_file_full_path)
        print("load_state_dict - ")
        self.load_state_dict(modelParams)
        self.eval()

    def train_model(self, train_dl, epochCnt=500):
        # define the optimization
        criterion = CrossEntropyLoss()
        optimizer = SGD(self.parameters(), lr=0.01, momentum=0.9)
        self.train()
        # enumerate epochs
        for epoch in range(epochCnt):
            # enumerate mini batches
            for i, (inputs, targets) in enumerate(train_dl):
                # clear the gradients
                optimizer.zero_grad()
                # compute the model output
                yhat = self.forward(inputs)
                # calculate loss
                loss = criterion(yhat, targets)
                # credit assignment
                loss.backward()
                # update model weights
                optimizer.step()

    def export_final_layer(self, test_dl):
        final_layer, predictions, actuals = list(), list(), list()
        sm = Softmax(dim=1)
        self.eval()
        for i, (inputs, targets) in enumerate(test_dl):
            # evaluate the model on the test set
            yhat = self.forward(inputs)

            fin_lay = yhat.clone()
            fin_lay = fin_lay.detach().numpy()
            final_layer.append(fin_lay)
            #if i<10:
            #   print(fin_lay.shape)

            yhat = sm(yhat)
            # retrieve numpy array
            yhat = yhat.detach().numpy()
            actual = targets.numpy()
            # convert to class labels
            yhat = np.argmax(yhat, axis=1)
            # reshape for stacking
            actual = actual.reshape((len(actual), 1))
            yhat = yhat.reshape((len(yhat), 1))
            # store
            predictions.append(yhat)
            actuals.append(actual)
        predictions, actuals = np.vstack(predictions), np.vstack(actuals)
        final_layer = np.vstack(final_layer)
        # calculate accuracy
        acc = accuracy_score(actuals, predictions)
        return acc, predictions, actuals, final_layer

    def evaluate_model(self, test_dl):
        predictions, actuals = list(), list()
        sm = Softmax(dim=1)
        self.eval()
        for i, (inputs, targets) in enumerate(test_dl):
            # evaluate the model on the test set
            yhat = self.forward(inputs)
            yhat = sm(yhat)
            # retrieve numpy array
            yhat = yhat.detach().numpy()
            actual = targets.numpy()
            # convert to class labels
            yhat = np.argmax(yhat, axis=1)
            # reshape for stacking
            actual = actual.reshape((len(actual), 1))
            yhat = yhat.reshape((len(yhat), 1))
            # store
            predictions.append(yhat)
            actuals.append(actual)
        predictions, actuals = np.vstack(predictions), np.vstack(actuals)
        # calculate accuracy
        acc = accuracy_score(actuals, predictions)
        return acc, predictions, actuals

    def train_and_evaluate(self, train_dl, valid_dl, test_dl, epochCnt=500, saveBestModelName=None):
        # define the optimization
        criterion = CrossEntropyLoss()
        optimizer = SGD(self.parameters(), lr=0.01, momentum=0.9)
        # enumerate epochs
        accvectr = np.zeros(epochCnt)
        accvecva = np.zeros(epochCnt)
        accvecte = np.zeros(epochCnt)
        acc_va_max = 0

        for epoch in range(epochCnt):
            # enumerate mini batches
            self.train()
            for i, (inputs, targets) in enumerate(train_dl):
                # clear the gradients
                optimizer.zero_grad()
                # compute the model output
                yhat = self.forward(inputs)
                # calculate loss
                loss = criterion(yhat, targets.squeeze_())
                # credit assignment
                loss.backward()
                # update model weights
                optimizer.step()
            acc_tr, _, _ = self.evaluate_model(train_dl)
            acc_va, _, _ = self.evaluate_model(valid_dl)
            acc_te, preds_te, labels_te = self.evaluate_model(test_dl)

            if acc_va_max < acc_va:
                preds_best, labels_best = preds_te, labels_te
                print("best validation epoch so far - epoch ", epoch, "va: %.3f" % acc_va, "te: %.3f" % acc_te)
                acc_va_max = acc_va
                if saveBestModelName is not None:
                    print("Saving model at : ", saveBestModelName)
                    torch_save(self.state_dict(), saveBestModelName)
                    print("Model saved..")
            else:
                print("epoch ", epoch, "tr: %.3f" % acc_tr, "va: %.3f" % acc_va, "te: %.3f" % acc_te)

            accvectr[epoch] = acc_tr
            accvecva[epoch] = acc_va
            accvecte[epoch] = acc_te
        return accvectr, accvecva, accvecte, preds_best, labels_best