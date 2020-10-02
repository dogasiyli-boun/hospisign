from sklearn.preprocessing import LabelEncoder
import pandas as pd
from torch.utils.data import Dataset, random_split

#SUBMIT_CHECK
def getFileName(dataToUse, normMode, pcaCount, numOfSigns, expectedFileType):
    pcaCountStr = str(pcaCount) if pcaCount > 0 else ""
    normModeStr = normMode if normMode == "" else "_" + normMode + "_"
    if expectedFileType == 'PCA':
        fileName_PCA            = dataToUse + normModeStr + 'PCA' + '_' + str(numOfSigns) + '.npy'  # 'hogPCA_41.npy' or 'hogPCA_11.npy' or 'skeletonFeats_11.npy' or 'snFeats_11.npy'
        fileName = fileName_PCA
    if expectedFileType == 'Data':
        fileName_Data           = dataToUse + normModeStr + pcaCountStr + 'Feats' + '_' + str(numOfSigns) + '.npy'  # 'hogFeats_41.npy' or 'hogFeats_11.npy' or 'skeletonFeats_11.npy' or 'snFeats_11.npy'
        fileName = fileName_Data
    if expectedFileType == 'Labels':
        fileName_Labels         = 'labels' + '_' + str(numOfSigns) + '.npy'  # 'labels_41.npy' or 'labels_11.npy'
        fileName = fileName_Labels
    if expectedFileType == 'DetailedLabels':
        fileName_DetailedLabels = 'detailedLabels' + '_' + str(numOfSigns) + '.npy'  # 'detailedLabels_41.npy' or 'detailedLabels_11.npy'
        fileName = fileName_DetailedLabels
    if expectedFileType == 'CorrespendenceVec':
        fileName_Corr = dataToUse + normModeStr + pcaCountStr + '_corrFrames' + '_' + str(numOfSigns) + '.npy'  # 'hog_corrFrames_41.npy' or 'hog_corrFrames_11.npy'
        fileName = fileName_Corr
    if expectedFileType == 'BaseResultName':
        fileName_BaseRes = dataToUse + normModeStr + pcaCountStr + '_' + str(numOfSigns) + '_baseResults' + '.npy'  # 'hog_baseResults_41.npy' or 'hog_baseResults_11.npy'
        fileName = fileName_BaseRes
    return fileName

# dataset definition
class HandCraftedDataset(Dataset):
    # load the dataset
    def __init__(self, path, X=None, y=None):
        # load the csv file as a dataframe
        if X is None and y is None:
            df = pd.read_csv(path, header=None)
            # store the inputs and outputs
            self.X = df.values[:, :-1]
            self.y = df.values[:, -1]
            # ensure input data is floats
            self.X = self.X.astype('float32')
            # label encode target and ensure the values are floats
            self.y = LabelEncoder().fit_transform(self.y)
            #print(self.X)
            #print(self.y)
        else:
            self.X = X.astype('float32')
            self.y = y
        print(self.X.shape)
        print(self.y.shape)

    # number of rows in the dataset
    def __len__(self):
        return len(self.X)

    # get a row at an index
    def __getitem__(self, idx):
        return [self.X[idx], self.y[idx]]

    # get indexes for train and test rows
    def get_splits(self, n_test=0.33):
        # determine sizes
        test_size = round(n_test * len(self.X))
        train_size = len(self.X) - test_size
        # calculate the split
        return random_split(self, [train_size, test_size])