import socket
import os
import sys
import numpy as np
from math import isnan as isNaN
from sklearn import metrics
from sklearn.metrics import confusion_matrix, normalized_mutual_info_score as nmi
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA

from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
from pandas import DataFrame as pd_df
import pandas as pd
import scipy.io
import time
import datetime
import yaml
from types import SimpleNamespace

class NestedNamespace(SimpleNamespace):
    def __init__(self, **kwargs):
        super(NestedNamespace, self).__init__(**kwargs)
        for k, v in kwargs.items():
            if type(v) == dict:
                setattr(self, k, NestedNamespace(**v))
            elif type(v) == list:
                setattr(self, k, list(map(self.map_entry, v)))

    @staticmethod
    def map_entry(e):
        if isinstance(e, dict):
            return NestedNamespace(**e)
        return e

class CustomConfigParser():
    def __init__(self, config_file):
        self.config_path = os.path.normpath(os.path.join(os.path.dirname(__file__), 'configs', config_file))
        _, config_ext = os.path.splitext(config_file)
        if config_ext != '.yaml':
            os.error("couldnt retrieve info from yaml file")
        config_dict = self._load_config_from_yaml()
        self.parameters = NestedNamespace(**config_dict)

    def _load_config_from_yaml(self):
        config_dict = {}
        with open(self.config_path, 'r') as f:
            try:
                config_dict = yaml.safe_load(f)
            except yaml.YAMLError as exc:
                os.error("couldnt retrieve info from yaml file")
        return config_dict

    def __getattr__(self, item):
        return self.parameters.__getattribute__(item)

_CONF_PARAMS_ = CustomConfigParser(config_file="conf.yaml")

def install_package_str(package_name):
    return "!{sys.executable} - m pip install " + package_name

#SUBMIT_CHECK
def getElapsedTimeFormatted(elapsed_miliseconds):
    hours, rem = divmod(elapsed_miliseconds, 3600)
    minutes, seconds = divmod(rem, 60)
    if hours > 0:
        retStr = "{:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds)
    elif minutes > 0:
        retStr = "{:0>2}:{:05.2f}".format(int(minutes), seconds)
    else:
        retStr = "{:05.2f}".format(seconds)
    return retStr

def getFileList(dir2Search, startString="", endString="", sortList=False):
    fileList = [f for f in os.listdir(dir2Search) if f.startswith(startString) and
                                                     f.endswith(endString) and
                                                    os.path.isfile(os.path.join(dir2Search, f))]
    if sortList:
        fileList = np.sort(fileList)
    return fileList

#SUBMIT_CHECKED
def append_to_vstack(stack, new_arr, dtype=int):
    if (stack is None or len(stack)==0):
        stack = new_arr
    else:
        stack = np.asarray(np.vstack((stack, new_arr)), dtype=dtype)
    return stack

#SUBMIT_CHECKED
def setPandasDisplayOpts():
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 1000)
    pd.set_option("display.precision", 3)
    pd.set_option('display.max_rows', None)

#SUBMIT_CHECK
#SUBMIT_CHANGE - put everything under config file
def getVariableByComputerName(variableName):
    curCompName = socket.gethostname()
    if variableName == 'base_dir':
        if curCompName == 'doga-MSISSD':
            base_dir = '/media/doga/SSD258/DataPath'  # for bogazici kasa
        elif curCompName == 'WsUbuntu05':
            base_dir = '/media/dg/SSD_Data/DataPath'  # for WS Doga DHO
        elif curCompName == 'doga-msi-ubu':
            base_dir = '/home/doga/DataFolder'  # for laptop
        else:
            base_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
        retVal = base_dir
    if variableName == 'desktop_dir':
        if curCompName == 'doga-MSISSD':
            desktop_dir = '/media/doga/Desktop'  # for bogazici kasa
        elif curCompName == 'WsUbuntu05':
            desktop_dir = '/media/dg/Desktop'  # for WS Doga DHO
        elif curCompName == 'doga-msi-ubu':
            desktop_dir = '/home/doga/Desktop'  # for laptop
        else:
            desktop_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
        retVal = desktop_dir
    if variableName == 'data_dir':
        if curCompName == 'doga-MSISSD':
            data_dir = '/media/doga/SSD258/DataPath/bdData'  # for bogazici kasa
        elif curCompName == 'WsUbuntu05':
            data_dir = '/media/dg/SSD_Data/DataPath/bdData'  # for WS Doga DHO
        elif curCompName == 'doga-msi-ubu':
            data_dir = '/home/doga/DataFolder/bdData'  # for laptop
        else:
            data_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
        retVal = data_dir
    if variableName == 'results_dir':
        if curCompName == 'doga-MSISSD':
            results_dir = '/media/doga/SSD258/DataPath/bdResults'  # for bogazici kasa
        elif curCompName == 'WsUbuntu05':
            results_dir = '/media/dg/SSD_Data/DataPath/bdResults'  # for WS Doga DHO
        elif curCompName == 'doga-msi-ubu':
            results_dir = '/home/doga/DataFolder/bdResults'  # for laptop
        else:
            results_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
        retVal = results_dir
    return retVal

#SUBMIT_CHECK
def createDirIfNotExist(dir2create):
    if not os.path.isdir(dir2create):
        os.makedirs(dir2create)

#SUBMIT_CHECKED
def normalize(x, axis=None):
    x = x / np.linalg.norm(x, axis=axis)
    return x

#SUBMIT_CHECKED
def normalize2(feat_vec, norm_mode, axis=0):
    if norm_mode == 'nm':
        if axis == 0:
            feat_vec = feat_vec - np.min(feat_vec, axis=0)
            feat_vec = feat_vec / np.max(feat_vec, axis=0)  # divide per column max
        else:
            feat_vec = feat_vec.T
            feat_vec = feat_vec - np.min(feat_vec, axis=0)
            feat_vec = feat_vec / np.max(feat_vec, axis=0)  # divide per column max
            feat_vec = feat_vec.T
    elif norm_mode == 'ns':
        if axis == 0:
            feat_vec = feat_vec / np.sum(feat_vec, axis=0)  # divide per column sum
        else:
            feat_vec = (feat_vec.T / np.sum(feat_vec.T, axis=0)).T  # divide per column sum
    elif norm_mode == 'softmax':
        if axis == 0:
            feat_vec = softmax(feat_vec)
        else:
            feat_vec = softmax(feat_vec.T).T
    return feat_vec

#SUBMIT_CHECKED
def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)  # only difference

#SUBMIT_CHECKED
def get_nmi_only(l, p, average_method='geometric'):
    nmi_res = nmi(l, p, average_method=average_method)
    return nmi_res

#SUBMIT_CHECKED
def applyMatTransform(featVec, applyPca=True, normMode='', verbose=0):
    """
        :param featVec: NxD feature vector
        :param applyPca: True or False
        :param normMode: ['','nm','nl']
        :param verbose: 0 or 1
        :return: feat_vec, exp_var_rat

        Example :
        X = np.array([[3, 5, 7, 9, 11], [4, 6, 15, 228, 245], [28, 19, 225, 149, 81], [18, 9, 125, 49, 2181], [8, 9, 25, 149, 81], [8, 9, 25, 49, 81], [8, 19, 25, 49, 81]])
        print('input array : \n', X)
        print('samples-rows : ', X.shape[0], ' / feats-cols : ', X.shape[1])

        X_nT_pF = applyMatTransform(X, applyNormalization=True, applyPca=False)
        printMatRes(X_nT_pF,'X_nT_pF')

        X_nF_pT = applyMatTransform(X, applyNormalization=False, applyPca=True)
        printMatRes(X_nF_pT,'X_nF_pT')

        X_nT_pT = applyMatTransform(X, applyNormalization=True, applyPca=True)
        printMatRes(X_nT_pT,'X_nT_pT')
    """

    exp_var_rat = []
    if applyPca:
        pca = PCA(whiten=False, svd_solver='auto')
        featVec = pca.fit_transform(featVec)
        exp_var_rat = np.cumsum(pca.explained_variance_ratio_)
        if verbose > 0:
            print('Max of featsPCA = ', np.amax(featVec), ', Min of featsPCA = ', np.amin(featVec))

    if normMode == '':
        pass # do nothing
    elif normMode == 'nm':
        featVec = normalize(featVec, axis=0)  # divide per column max
    elif normMode == 'nl':
        featVec = normalize(featVec, axis=None)  # divide to a scalar = max of matrix
    else:
        os.error("norm_mode must be defined as one of the following = ['','nm','nl']. norm_mode(" + normMode + ")")

    if verbose > 0 and normMode != '':
        print('Max of normedFeats = ', np.amax(featVec), ', Min of normedFeats = ', np.amin(featVec))

    return featVec, exp_var_rat

#SUBMIT_CHECK
def get_cluster_centroids(ft, predClusters, kluster_centers, verbose=0):
    print(predClusters.shape)
    print(kluster_centers.shape)
    centroid_info_mat = np.asarray(np.squeeze(np.zeros((kluster_centers.shape[0], 3)))) - 1

    uniq_preds = np.unique(predClusters)
    cntUniqPred = uniq_preds.size
    for i in range(0, cntUniqPred):  #
        klust_cur = uniq_preds[i]
        inds = getInds(predClusters, klust_cur)
        ft_cur = ft[inds, :]
        center_cur = kluster_centers[i, :]
        center_mat = center_cur[None, :]
        dist_m = np.array(cdist(ft_cur, center_mat))

        predicted_inds_rel = np.argmin(dist_m)
        predicted_inds_abs = inds[predicted_inds_rel]
        centroid_info_mat[i, :] = [klust_cur, predicted_inds_abs, np.min(dist_m)]
        if verbose > 0:
            print("k({:d}), pred_inds_rel({:d}), pred_inds_abs({:d}), min({:5.3f}), max({:5.3f})".format(i,
                                                                                                         predicted_inds_rel,
                                                                                                         predicted_inds_abs,
                                                                                                         np.min(dist_m),
                                                                                                         np.max(
                                                                                                             dist_m)))

    centroid_df = pd.DataFrame(centroid_info_mat, columns=['klusterID', 'sampleID', 'distanceEuc'])
    centroid_df = centroid_df.astype({'klusterID': int, 'sampleID': int, 'distanceEuc': float})
    return centroid_df

#SUBMIT_CHECK
def clusterData(featVec, n_clusters, normMode='', applyPca=True, clusterModel='KMeans'):
    featVec, exp_var_rat = applyMatTransform(np.array(featVec), applyPca=applyPca, normMode=normMode)
    df = pd_df(featVec)

    curTol = 0.0001 if clusterModel == 'KMeans' else 0.01
    max_iter = 300 if clusterModel == 'KMeans' else 200

    numOf_1_sample_bins = 1
    expCnt = 0
    while numOf_1_sample_bins-expCnt > 0 and expCnt < 5:
        t = time.time()
        if expCnt > 0:
            print("running ", clusterModel, " for the ", str(expCnt), " time due to numOf_1_sample_bins(",
                  str(numOf_1_sample_bins), ")")
        print('Clustering the feat_vec(', featVec.shape, ') with n_clusters(', str(n_clusters), ') and model = ',
              clusterModel, ", curTol(", str(curTol), "), max_iter(", str(max_iter), "), at ",
              datetime.datetime.now().strftime("%H:%M:%S"))
        kluster_centers = None
        if clusterModel == 'KMeans':
                #default vals for kmeans --> max_iter=300, 1e-4
                kmeans_result = KMeans(n_clusters=n_clusters, n_init=5, tol=curTol, max_iter=max_iter).fit(df)
                predictedKlusters = kmeans_result.labels_.astype(float)
                kluster_centers = kmeans_result.cluster_centers_.astype(float)
        elif clusterModel == 'GMM_full':
            # default vals for gmm --> max_iter=100, 1e-3
            predictedKlusters = GaussianMixture(n_components=n_clusters, covariance_type='full', tol=curTol, max_iter=max_iter).fit_predict(df)
        elif clusterModel == 'GMM_diag':
            predictedKlusters = GaussianMixture(n_components=n_clusters, covariance_type='diag', tol=curTol, max_iter=max_iter).fit_predict(df)
        elif clusterModel == 'Spectral':
            sc = SpectralClustering(n_clusters=n_clusters, affinity='rbf', random_state=0)
            sc_clustering = sc.fit(featVec)
            predictedKlusters = sc_clustering.labels_
        numOf_1_sample_bins, histSortedInv = analyzeClusterDistribution(predictedKlusters, n_clusters, verbose=0)
        curTol = curTol * 10
        max_iter = max_iter + 50
        expCnt = expCnt + 1
        elapsed = time.time() - t
        print('Clustering done in (', getElapsedTimeFormatted(elapsed), '), ended at ', datetime.datetime.now().strftime("%H:%M:%S"))
    print('Clustering completed with (', np.unique(predictedKlusters).shape, ') clusters,  expCnt(', str(expCnt), ')')

    return np.asarray(predictedKlusters, dtype=int), kluster_centers

#SUBMIT_CHECKED
def del_rows_cols(x, row_ids, col_ids=np.array([])):
    if row_ids.size > 0:
        x = np.delete(x, row_ids, axis=0)
    if col_ids.size > 0:
        x = np.delete(x, col_ids, axis=1)
    x = np.squeeze(x)
    return x

#SUBMIT_CHECK
def calcConfusionStatistics(confMat, categoryNames=None, selectedCategories=None, verbose=0, data_frame_keys=None):
    # https://en.wikipedia.org/wiki/Confusion_matrix
    # confMat-rows are actual(true) classes
    # confMat-cols are predicted classes

    # for all categories a confusion matrix can be re-arranged per category,
    # for example for category "i";
    # ----------  -----------------------------
    # TP | FN |  |      TP     | Type2 Error |
    # FP | TN |  | Type1 Error |      TN     |
    # ----------  -----------------------------
    # --------------------------------------------------------------------------
    #       c_i is classified as ci         |  c_i are classified as non-ci   |
    # other classes falsly predicted as c_i | non-c_i is classified as non-ci |
    # --------------------------------------------------------------------------
    # TP : true positive - c_i is classified as ci
    # FN : false negative - c_i are classified as non-ci
    # FP : false positive - other classes falsly predicted as c_i
    # TN : true negative - non-c_i is classified as non-ci
    categoryCount = confMat.shape[1]
    if categoryCount != confMat.shape[1]:
        print('problem with confusion matrix')
        return

    if selectedCategories is not None and len(selectedCategories) != 0:
        selectedCategories = selectedCategories[np.argwhere(selectedCategories <= categoryCount)]
    else:
        selectedCategories = np.arange(0, categoryCount)

    categoryCount = len(selectedCategories);

    if verbose > 2:
        print('Columns of confusion mat is predictions, rows are ground truth.')

    confMatStats = {}
    sampleCounts_All = np.sum(confMat, axis=1)
    totalCountOfAll = np.sum(sampleCounts_All)
    if verbose > 0:
        print("sampleCounts_All : \n", sampleCounts_All)
        print("selectedCategories : \n", selectedCategories)

    for i in range(categoryCount):
        c = selectedCategories[i]

        totalPredictionOfCategory = np.sum(confMat[:, c], axis=0)
        totalCountOfCategory = sampleCounts_All[c]

        TP = confMat[c, c]
        FN = totalPredictionOfCategory - TP
        FP = totalCountOfCategory - TP
        TN = totalCountOfAll - (TP + FN + FP)

        ACC = (TP + TN) / totalCountOfAll  # accuracy
        TPR = TP / (TP + FN)  # true positive rate, sensitivity
        TNR = TN / (FP + TN)  # true negative rate, specificity
        PPV = TP / (TP + FP)  # positive predictive value, precision
        NPV = TN / (TN + FN)  # negative predictive value
        FPR = FP / (FP + TN)  # false positive rate, fall out
        FDR = FP / (FP + TP)  # false discovery rate
        FNR = FN / (FN + TP)  # false negative rate, miss rate

        F1 = (2 * TP) / (2 * TP + FP + FN)  # harmonic mean of precision and sensitivity
        MCC = (TP * TN - FP * FN) / np.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))  # matthews correlation coefficient
        INFORMEDNESS = TPR + TNR - 1
        MARKEDNESS = PPV + NPV - 1

        #
        c_stats = {
            "totalCountOfAll": totalCountOfAll,
            "totalCountOfCategory": totalCountOfCategory,
            "totalPredictionOfCategory": totalPredictionOfCategory,

            "TruePositive": TP,
            "FalseNegative": FN,
            "FalsePositive": FP,
            "TrueNegative": TN,

            "Accuracy": ACC,
            "Sensitivity": TPR,
            "Specificity": TNR,
            "Precision": PPV,
            "Negative_Predictive_Value": NPV,
            "False_Positive_Rate": FPR,
            "False_Discovery_Rate": FDR,
            "False_Negative_Rate": FNR,

            "F1_Score": F1,
            "Matthews_Correlation_Coefficient": MCC,
            "Informedness": INFORMEDNESS,
            "Markedness": MARKEDNESS,
        }

        if categoryNames is None:
            categoryName = str(i).zfill(2)
        else:
            categoryName = categoryNames[c]
        confMatStats[categoryName] = c_stats


    if data_frame_keys is None:
        data_frame_keys = ["F1_Score", "totalCountOfCategory"]

    df_slctd_table = pd.DataFrame({"khsName": [k for k in confMatStats.keys()]})
    for dfk in data_frame_keys:
        df_add = pd.DataFrame({dfk: [confMatStats[k][dfk] for k in confMatStats.keys()]})
        df_slctd_table = pd.concat([df_slctd_table, df_add], axis=1)

    if verbose > 0:
        print("\n**\nfinal df_slctd_table-\n", df_slctd_table)

    if categoryCount == 1:
        confMatStats = confMatStats[selectedCategories, 1]

    return confMatStats, df_slctd_table

#SUBMIT_CHECK
def plot_confusion_matrix(conf_mat,
                          hide_spines=False, hide_ticks=False,
                          figsize=None, add_true_cnt=True, add_pred_cnt=True, add2XLabel="", add2YLabel="",
                          cmap=None, colorbar=False, show_absolute=True, show_normed=False,
                          class_names=None, iterID=-1, saveConfFigFileName='', figMulCnt=None,
                          confusionTreshold=0.3, show_only_confused=False, rotVal=30):
    """Plot a confusion matrix via matplotlib.
    Parameters
    -----------
    conf_mat : array-like, shape = [n_classes, n_classes]
        Confusion matrix from evaluate.confusion matrix.
    hide_spines : bool (default: False)
        Hides axis spines if True.
    hide_ticks : bool (default: False)
        Hides axis ticks if True
    figsize : tuple (default: (2.5, 2.5))
        Height and width of the figure
    cmap : matplotlib colormap (default: `None`)
        Uses matplotlib.pyplot.cm.Blues if `None`
    colorbar : bool (default: False)
        Shows a colorbar if True
    show_absolute : bool (default: True)
        Shows absolute confusion matrix coefficients if True.
        At least one of  `show_absolute` or `show_normed`
        must be True.
    show_normed : bool (default: False)
        Shows normed confusion matrix coefficients if True.
        The normed confusion matrix coefficients give the
        proportion of training examples per class that are
        assigned the correct label.
        At least one of  `show_absolute` or `show_normed`
        must be True.
    class_names : array-like, shape = [n_classes] (default: None)
        List of class names.
        If not `None`, ticks will be set to these values.
    Returns
    -----------
    fig, ax : matplotlib.pyplot subplot objects
        Figure and axis elements of the subplot.
    Examples
    -----------
    For usage examples, please see
    http://rasbt.github.io/mlxtend/user_guide/plotting/plot_confusion_matrix/
    """

    if not (show_absolute or show_normed):
        raise AssertionError('Both show_absolute and show_normed are False')
    if class_names is not None and len(class_names) != len(conf_mat):
        raise AssertionError('len(class_names) should be equal to number of'
                             'classes in the dataset')

    if figMulCnt is None:
        figMulCnt = 0.5 + 0.25 * int(show_absolute) + 0.25 * int(show_normed)
        print("figMulCnt = ", figMulCnt)
    if figsize is None:
        figsize = ((len(conf_mat)) * figMulCnt, (len(conf_mat)) * figMulCnt)

    acc = np.sum(np.diag(conf_mat)) / np.sum(np.sum(conf_mat))

    x_preds_ids = np.arange(conf_mat.shape[0])
    y_true_ids = np.arange(conf_mat.shape[1])

    total_samples = conf_mat.sum(axis=1)[:, np.newaxis]
    total_preds = conf_mat.sum(axis=0)[:, np.newaxis]
    normed_conf_mat = conf_mat.astype('float') / total_samples
    total_samples = np.squeeze(total_samples)
    total_preds = np.squeeze(total_preds)

    if class_names is not None:
        class_names_x_preds = class_names.copy()
        class_names_y_true = class_names.copy()

    if show_only_confused:
        ncm = normed_conf_mat.copy()
        np.fill_diagonal(ncm, 0)
        confused_cols = np.squeeze(np.where(np.any(ncm > confusionTreshold, axis=0)))
        confused_rows = np.squeeze(np.where(np.any(ncm > confusionTreshold, axis=1)))
        all_rows = np.arange(normed_conf_mat.shape[0])
        all_cols = np.arange(normed_conf_mat.shape[1])
        ok_rows = del_rows_cols(all_rows, confused_rows)
        ok_cols = del_rows_cols(all_cols, confused_cols)

        conf_mat = del_rows_cols(conf_mat, ok_rows, ok_cols)
        normed_conf_mat = del_rows_cols(normed_conf_mat, ok_rows, ok_cols)

        x_preds_ids = x_preds_ids[confused_cols]
        y_true_ids = y_true_ids[confused_rows]

        total_samples = del_rows_cols(total_samples, ok_rows)
        total_preds = del_rows_cols(total_preds, ok_cols)
        figsize = (conf_mat.shape[0] * figMulCnt, conf_mat.shape[1] * figMulCnt * 2)
        print("figsize=", figsize)
        if class_names is not None:
            class_names_x_preds = class_names_x_preds[confused_cols]
            class_names_y_true = class_names_y_true[confused_rows]

    fig, ax = plt.subplots(figsize=figsize)
    ax.grid(False)
    if cmap is None:
        cmap = plt.cm.Blues

    if show_normed:
        matshow = ax.matshow(normed_conf_mat, cmap=cmap)
    else:
        matshow = ax.matshow(conf_mat, cmap=cmap)

    if colorbar:
        fig.colorbar(matshow)

    for i in range(conf_mat.shape[0]):
        for j in range(conf_mat.shape[1]):
            cell_text = ""
            class_name_i = class_names_y_true[i]
            class_name_j = class_names_x_preds[j]

            if show_absolute and conf_mat[i, j] > 0:
                cell_text += format(conf_mat[i, j], 'd')
                if show_normed and normed_conf_mat[i, j] > 0.005:
                    cell_text += "\n" + '('
                    cell_text += format(normed_conf_mat[i, j], '.2f') + ')'
            elif conf_mat[i, j] > 0 and normed_conf_mat[i, j] > 0.005:
                cell_text += format(normed_conf_mat[i, j], '.2f')

            if (class_name_i != class_name_j and normed_conf_mat[i, j] > confusionTreshold):
                text_color = "red"
            elif show_normed:
                text_color = "white" if normed_conf_mat[i, j] > 0.5 else "black"
            else:
                text_color = "white" if conf_mat[i, j] > np.max(conf_mat) / 2 else "black"
            ax.text(x=j, y=i,
                    s=cell_text,
                    va='center', ha='center',
                    color=text_color, fontsize='x-large')
    if class_names is not None:
        tick_marks_x = np.arange(len(class_names_x_preds))
        tick_marks_y = np.arange(len(class_names_y_true))
        if add_true_cnt:
            for i in range(len(class_names_x_preds)):
                class_names_x_preds[i] = str(x_preds_ids[i]) + '.' + class_names_x_preds[i] + '\n({:.0f})'.format(
                    total_preds[i])
        if add_pred_cnt:
            for i in range(len(class_names_y_true)):
                class_names_y_true[i] = str(y_true_ids[i]) + '.' + class_names_y_true[i] + '\n({:.0f})'.format(
                    total_samples[i])
        plt.xticks(tick_marks_x, class_names_x_preds, rotation=rotVal)
        plt.yticks(tick_marks_y, class_names_y_true)

    if hide_spines:
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    if hide_ticks:
        ax.axes.get_yaxis().set_ticks([])
        ax.axes.get_xaxis().set_ticks([])

    if iterID != -1:
        plt.xlabel('Predicted - iter {:03d}'.format(iterID + 1) + ' ' + add2XLabel)
    else:
        plt.xlabel('Predicted - ' + ' ' + add2XLabel)
    plt.xlabel('True - ' + ' ' + add2YLabel)

    plot_title_str = saveConfFigFileName.split(os.path.sep)[-1]
    plot_title_str = plot_title_str.split('.')[0]
    plot_title_str += '_accuracy<{:4.2f}>_'.format(100*acc)
    plt.title(plot_title_str[0:-1])

    if saveConfFigFileName == '':
        plt.show()
    else:
        plt.tight_layout()
        saveConfFigFileName = saveConfFigFileName.replace(".", "_acc(" + '{:.0f}'.format(acc * 100) + ").")
        saveConfFigFileName = saveConfFigFileName.replace(".", "_rot(" + str(rotVal) + ").")
        saveConfFigFileName = saveConfFigFileName.replace(".",
                                                          "_ctv(" + '{:.0f}'.format(confusionTreshold * 100) + ").")
        saveConfFigFileName = saveConfFigFileName.replace(".", "_fmc(" + '{:.2f}'.format(figMulCnt) + ").")
        plt.savefig(saveConfFigFileName)

    return fig, ax

#SUBMIT_CHECKED
def sortVec(v):
    idx = np.unravel_index(np.argsort(-v, axis=None), v.shape)
    v = v[idx]
    return v, idx

#SUBMIT_CHECKED
#mat = loadMatFile('/mnt/USB_HDD_1TB/neuralNetHandVideos_11/surfImArr_all_pca_256.mat')
def loadMatFile(matFileNameFull, verbose=1):
    mat = scipy.io.loadmat(matFileNameFull)
    if verbose > 0:
        print(sorted(mat.keys()))
    return mat

#SUBMIT_CHECKED
def analyzeClusterDistribution(predictedKlusters, n_clusters, verbose=0, printHistCnt=10):
    histOfClust, binIDs = np.histogram(predictedKlusters, np.unique(predictedKlusters))
    numOfBins = len(binIDs)
    numOf_1_sample_bins = np.sum(histOfClust==1)
    if verbose > 0:
        print(n_clusters, " expected - ", numOfBins, " bins extracted. ", numOf_1_sample_bins, " of them have 1 sample")
    histSortedInv = np.sort(histOfClust)[::-1]
    if verbose > 1:
        print("hist counts ascending = ", histSortedInv[0:printHistCnt])
    return numOf_1_sample_bins, histSortedInv

#SUBMIT_CHECKED
def getDict(retInds, retVals):
    ret_data = []
    for i in range(0, len(retInds)):
        ret_data.append([retInds[i], retVals[i]])
    return ret_data

#SUBMIT_CHECKED
def getPandasFromDict(retInds, retVals, columns):
    ret_data = getDict(retInds, retVals)
    if "pandas" in sys.modules:
        ret_data = pd_df(ret_data, columns=columns)
        print(ret_data)
    return ret_data

#SUBMIT_CHECKED
def calcClusterMetrics(labels_true, labels_pred, removeZeroLabels=False, labelNames=None):
    # print("This function gets predictions and real labels")
    # print("also a parameter that says either remove 0 labels or not")
    # print("an optional parameter as list of label names")
    np.unique(labels_true)
    # 1 remove zero labels if necessary

    # 2 calculate
    ars = metrics.adjusted_rand_score(labels_true, labels_pred)
    mis = metrics.mutual_info_score(labels_true, labels_pred)
    amis = metrics.adjusted_mutual_info_score(labels_true, labels_pred)
    nmi = metrics.normalized_mutual_info_score(labels_true, labels_pred)
    hs = metrics.homogeneity_score(labels_true, labels_pred)
    cs = metrics.completeness_score(labels_true, labels_pred)
    vms = metrics.v_measure_score(labels_true, labels_pred)
    fms = metrics.fowlkes_mallows_score(labels_true, labels_pred)

    # 3 print
    retVals = [ars, mis, amis, cs, vms, fms, nmi, hs]
    retInds = ['adjusted_rand_score', 'mutual_info_score', 'adjusted_mutual_info_score', 'completeness_score',
               'v_measure_score', 'fowlkes_mallows_score', 'normalized_mutual_info_score', 'homogeneity_score']
    ret_data = getPandasFromDict(retInds, retVals, columns=['metric', 'value'])

    return ret_data

#SUBMIT_CHECKED
def getInds(vec, val):
    return np.array([i for i, e in enumerate(vec) if e == val])

#SUBMIT_CHECKED
def getIndicedList(baseList, indsList):
    return [baseList[i] for i in indsList]

#SUBMIT_CHECKED
def get_most_frequent(List):
    return max(set(List), key=List.count)

#SUBMIT_CHECKED
def calcPurity(labels_k, mappedClass=None, verbose=0):
    cnmxh_perc_add = 0
    most_frequent_class = get_most_frequent(labels_k)
    if mappedClass is None:
        mappedClass = most_frequent_class
    else:
        cnmxh_perc_add = int(mappedClass == most_frequent_class)

    if verbose > 0:
        print("mappedClass({:d}), most_frequent_class({:d})".format(mappedClass, most_frequent_class))
    try:
        correctLabelInds = getInds(labels_k, mappedClass)
    except:
        print("labels_k = ", labels_k)
        print("mappedClass = ", mappedClass)
        sys.exit("Error message")

    purity_k = 0
    if len(labels_k) > 0:
        purity_k = 100 * (len(correctLabelInds) / len(labels_k))

    return purity_k, correctLabelInds, mappedClass, cnmxh_perc_add

#SUBMIT_CHECK
def countPredictionsForConfusionMat(labels_true, labels_pred, labelNames=None, centroid_info_pdf=None, verbose=0):
    sampleCount = labels_pred.size
    labels_pred2class = labels_pred.copy()
    kluster2Classes = []
    uniq_preds = np.unique(labels_pred)
    kr_data = []
    weightedPurity = 0
    cntUniqPred = uniq_preds.size
    cnmxh_perc = 0
    for i in range(0, cntUniqPred):
        klust_cur = uniq_preds[i]
        inds = getInds(labels_pred, klust_cur)
        labels_k = getIndicedList(labels_true, inds)

        if centroid_info_pdf is not None:
            klusterID, sampleID = centroid_info_pdf['klusterID'][i], centroid_info_pdf['sampleID'][i]
            if klusterID == klust_cur:
                mappedClass = labels_true[sampleID]
            else:
                mappedClass = None
        else:
            mappedClass = None
        purity_k, correctLabelInds, mappedClass, cnmxh_perc_add = calcPurity(labels_k, mappedClass=mappedClass, verbose=verbose)
        cnmxh_perc = cnmxh_perc + cnmxh_perc_add

        kluster2Classes.append([klust_cur, mappedClass, len(labels_k), correctLabelInds.size])
        labels_pred2class[inds] = mappedClass

        weightedPurity += purity_k * (len(inds) / sampleCount)

        try:
            cStr = "c(" + str(klust_cur) + ")" if labelNames is None else labelNames[mappedClass]
            #print('mappedClass {:d} : {}'.format(mappedClass, labelNames[mappedClass]))
        except:
            print("klust_cur = ", klust_cur)
            print("mappedClass = ", mappedClass)
            print("labelNames2.size = ", len(labelNames))
            sys.exit("Some indice error maybe")

        kr_data.append(["k" + str(uniq_preds[i]), cStr, len(correctLabelInds), len(inds), purity_k])

    kr_pdf = pd_df(kr_data, columns=['kID', 'mappedClass', '#of', 'N', '%purity'])
    kr_pdf.sort_values(by=['%purity', 'N'], inplace=True, ascending=[False, False])

    _confMat = confusion_matrix(labels_true, labels_pred2class)
    cnmxh_perc = 100*cnmxh_perc/cntUniqPred

    return _confMat, kluster2Classes, kr_pdf, weightedPurity, cnmxh_perc

#SUBMIT_CHECKED
def calcCluster2ClassMetrics(labels_true, labels_pred, labelNames=None, predictDefStr="", verbose=0):
    # print("This function gets predictions and real labels")
    # print("an optional parameter as list of label names")
    # print("calc purity and assignment per cluster")

    _confMat, kluster2Classes, kr_pdf, weightedPurity, cnmxh_perc = countPredictionsForConfusionMat(labels_true, labels_pred)

    sampleCount = np.sum(np.sum(_confMat))
    acc = 100 * np.sum(np.diag(_confMat)) / sampleCount

    if verbose > 0:
        print(predictDefStr, "-k2c", kluster2Classes)
        print("\r\n\r\n")

    kr_data = []
    uniq_preds = np.unique(labels_pred)
    weightedPurity = 0
    for i in range(0, uniq_preds.size):
        klust_cur = uniq_preds[i]
        mappedClass = kluster2Classes[i][1]

        inds = getInds(labels_pred, klust_cur)
        labels_k = getIndicedList(labels_true, inds)
        correctLabelInds = getInds(labels_k, mappedClass)
        purity_k = 0
        if len(labels_k) > 0:
            purity_k = 100 * (len(correctLabelInds) / len(labels_k))

        weightedPurity += purity_k * (len(inds) / sampleCount)

        cStr = "c(" + str(klust_cur) + ")" if labelNames is None else labelNames[mappedClass]
        kr_data.append(["k" + str(uniq_preds[i]), cStr, len(correctLabelInds), len(inds), purity_k])
    kr_pdf = pd_df(kr_data, columns=['kID', 'mappedClass', '#of', 'N', '%purity'])
    kr_pdf.sort_values(by=['%purity', 'N'], inplace=True, ascending=[False, False])

    analyzeClusterDistribution(labels_pred, max(uniq_preds), verbose=2, printHistCnt=len(uniq_preds))

    if verbose > 0:
        print(predictDefStr, "-kr_pdf:\r\n", kr_pdf, "\r\n\r\n")
    c_data = []
    uniq_labels = np.unique(labels_true)
    trueCnt = np.sum(_confMat, axis=1)
    predCnt = np.sum(_confMat, axis=0)
    weightedPrecision = 0
    weightedRecall = 0
    weightedF1Score = 0
    for i in range(0, uniq_labels.size):
        class_cur = uniq_labels[i]

        correctCnt = _confMat[class_cur, class_cur]
        if correctCnt==0:
            recallCur = 0
            precisionCur = 0
            f1Cur = 0
        else:
            recallCur = 100 * (correctCnt / trueCnt[class_cur])
            precisionCur = 100 * (correctCnt / predCnt[class_cur])
            f1Cur = 2 * ((precisionCur * recallCur) / (precisionCur + recallCur))
        if verbose > 0 and isNaN(f1Cur):
            print("****************************None")

        wp = precisionCur * (trueCnt[class_cur] / sampleCount)
        wr = recallCur * (trueCnt[class_cur] / sampleCount)
        wf = f1Cur * ((trueCnt[class_cur]) / (sampleCount))

        weightedPrecision += wp
        weightedRecall += wr
        weightedF1Score += wf

        cStr = ["c(" + class_cur + ")" if labelNames is None else labelNames[class_cur]]
        c_data.append([cStr, correctCnt, precisionCur, recallCur, f1Cur, wp, wr, wf])
    c_pdf = pd_df(c_data, columns=['class', '#', '%prec', '%recall', '%f1', '%wp', '%wr', '%wf'])
    c_pdf.sort_values(by=['%f1', '#'], inplace=True, ascending=[False, False])

    retVals = [acc, weightedPurity, weightedPrecision, weightedRecall, weightedF1Score]
    retInds = ['accuracy', 'weightedPurity', 'weightedPrecision', 'weightedRecall', 'weightedF1Score']
    classRet = getPandasFromDict(retInds, retVals, columns=['metric', 'value'])

    if verbose > 0:
        print(predictDefStr, "-c_pdf:\r\n", c_pdf, "\r\n\r\n")

    plot_confusion_matrix(_confMat, add_true_cnt=True, add_pred_cnt=True, show_absolute=True, show_normed=True,
                          class_names=labelNames)
    if verbose > 0:
        print(predictDefStr, "-_confMat:\r\n", pd_df(_confMat.T, columns=labelNames, index=labelNames), "\r\n\r\n")

    return classRet, _confMat, c_pdf, kr_pdf

#SUBMIT_CHECKED
def reset_labels(allLabels, labelIDs, labelStrings, sortBy=None, verbose=0):
    labelIDs = np.asarray(labelIDs, dtype=int)
    if verbose > 1:
        print("1(reset_labels)-len(allLabels) and type=", allLabels.shape, allLabels.dtype)
        print("2(reset_labels)-unique(allLabels)=", np.unique(allLabels))
        print("3(reset_labels)-labelIDs.shape=", labelIDs.shape)
        print("4(reset_labels)-labelStrings.shape=", labelStrings.shape)
        print("5(reset_labels)-vstack((labelIDs,labelStrings))\n", np.vstack((labelIDs, labelStrings)).T)

    sortedLabelsMap = pd.DataFrame({'labelIDs': labelIDs, 'labelStrings': labelStrings})

    if sortBy == "name":
        sort_names_khs = np.argsort(np.argsort(labelStrings))
        sort_vals = np.argsort(labelStrings)  # sort_names_khs.values
        di = {labelIDs[i]: int(sort_names_khs[i]) for i in range(len(sort_names_khs))}
        if verbose > 1:
            print("5.1(reset_labels)-sort_b_name - sort_names_khs:", sort_names_khs)
            print("6(reset_labels)-sort_b_name - sort_vals:", sort_vals)
            print("7(reset_labels)-sort_b_name id map : \n", di)
    else:
        sort_to_zero_n = np.argsort(labelIDs)
        sort_vals = sort_to_zero_n
        di = {int(labelIDs[i]): int(sort_to_zero_n[i]) for i in range(len(labelIDs))}
        if verbose > 1:
            print("6(reset_labels)-sort_b_name - sort_vals:", sort_vals)
            print("7(reset_labels)-sort_to_zero_n id map : \n", di)

    sortedLabelsAll = pd.DataFrame({"labelIDs":allLabels})
    sortedLabelsAll["labelIDs"].replace(di, inplace=True)
    if verbose > 1:
        print("8.1(reset_labels)-sortedLabelsAll.shape=", sortedLabelsAll.shape)
        print("8.2(reset_labels)-sortedLabelsMap\n", sortedLabelsMap)
    sortedLabelsMap["labelIDs"].replace(di, inplace=True)
    if verbose > 1:
        print("9(reset_labels)-sortedLabelsMap\n", sortedLabelsMap)
    sortedLabelsMap["labelIDs"] = [sortedLabelsMap["labelIDs"][sort_vals[i]] for i in range(len(sort_vals))]
    if verbose > 1:
        print("10(reset_labels)-sortedLabelsMap\n", sortedLabelsMap)
    sortedLabelsMap["labelStrings"] = [labelStrings[sort_vals[i]] for i in range(len(sort_vals))]
    if verbose > 1:
        print("11(reset_labels)-sortedLabelsMap\n", sortedLabelsMap)
        print("12(reset_labels)***************************\n")

    return sortedLabelsAll, sortedLabelsMap
