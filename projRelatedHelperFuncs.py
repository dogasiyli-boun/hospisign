import helperFuncs as funcH
import dataLoaderFuncs as funcD
import ensembleFuncs as funcEns
import numpy as np
import os
from numpy.random import seed
import pandas as pd
import time
import datetime
from sklearn.metrics import confusion_matrix
from collections import Counter

from torch.utils.data import DataLoader

#SUBMIT_CHECKED
def get_nos(dataset_ident_str):
    if dataset_ident_str == 'Dev':
        return 11
    if dataset_ident_str == 'Exp':
        return 41
    os.error("Input can be either Dev or Exp")

def get_hospisign_file_name(data_ident_vec, dataset_ident_str):
    """
    :param data_ident_vec: "hog", "snv", "skeleton", "list_dict"
    :param dataset_ident_str: 'Dev' or 'Exp'
    :return: filename

    downloads the extracted features from web
    """
    file_type = ".txt" if data_ident_vec == "list_dict" else ".mat"
    # possible file names :
    # <hog, snv, skeleton>_<Dev, Exp>Set.mat
    # list_dict_<Dev, Exp>Set.txt
    filename = data_ident_vec + "_" + dataset_ident_str + "Set" + file_type
    return filename

#features, labels = getFeatsFromMat(mat,'dataCurdim', 'labelVecs_all')
#SUBMIT_CHECK
def getFeatsFromMat(mat, featureStr, labelStr):
    features = np.asarray(mat[featureStr], dtype=float)
    labels = np.asarray(mat[labelStr], dtype=int)
    return features, labels

def getMatFile(data_dir, signCnt, possible_fname_init):
    # possible_fname_init = ['surfImArr', 'snFeats']
    # possible_fname_init = ['skeleton', 'skelFeats']
    matFileName_1 = os.path.join(data_dir, possible_fname_init[0] + '_' + str(signCnt) + '.mat')
    matFileName_2 = os.path.join(data_dir, possible_fname_init[1] + '_' + str(signCnt) + '.mat')
    if os.path.isfile(matFileName_1):
        matFileName = matFileName_1
    elif os.path.isfile(matFileName_2):
        matFileName = matFileName_2
    else:
        print('neither ', matFileName_1, ' nor ', matFileName_2, ' is a file')
        return []
    print(matFileName, ' is loaded :) ')
    mat = funcH.loadMatFile(matFileName)
    return mat

#SUBMIT_CHECK
def run_clustering_hospisign(ft, labels_all, lb_map, dataToUse, numOfSigns, pcaCount, clustCntVec = None, clusterModels = ['KMeans'], randomSeed=5, enforce_rerun=False):
    seed(randomSeed)
    prevPrintOpts = np.get_printoptions()
    np.set_printoptions(precision=4, suppress=True)
    labels_all = np.squeeze(np.array(labels_all))
    class_names = np.asarray(lb_map["khsName"])

    baseResultFileName = funcD.getFileName(dataToUse=dataToUse, normMode="", pcaCount=pcaCount, numOfSigns=numOfSigns, expectedFileType='BaseResultName')
    results_dir = funcH._CONF_PARAMS_.DIR.RESULTS  # '/media/dg/SSD_Data/DataPath/bdResults'
    funcH.createDirIfNotExist(os.path.join(results_dir, 'baseResults'))
    baseResultFileNameFull = os.path.join(results_dir, 'baseResults', baseResultFileName)

    featsName = baseResultFileName.replace(".npy", "")  # '<dataToUse><pcaCount>_baseResults_<nos>.npy'

    print('*-*-*-*-*-*-*running for : ', featsName, '*-*-*-*-*-*-*')
    print('featSet(', ft.shape, '), labels_All(', labels_all.shape, ')')

    if clustCntVec is None:
        if ft.shape[1] > 128:
            clustCntVec = [128, 256, 512]
        else:
            clustCntVec = [32, 64, 96]
    # clustCntVec = [64, 128, 256] #[32, 64, 128, 256, 512]
    if os.path.isfile(baseResultFileNameFull):
        print('resultDict will be loaded from(', baseResultFileNameFull, ')')
        resultDict = list(np.load(baseResultFileNameFull, allow_pickle=True))
    else:
        resultDict = []

    headerStrFormat = "+++frmfile(%15s) clusterModel(%8s), clusCnt(%4s)"
    valuesStrFormat = "nmiAll(%.2f) * acc_cent(%.2f) * meanPurity_cent(%.3f) * weightedPurity_cent(%.3f) * acc_mxhs(%.2f) * meanPurity_mxhs(%.3f) * weightedPurity_mxhs(%.3f) * cnmxh_perc(%.3f) * emptyClusters(%d)"

    for clusterModel in clusterModels:
        for curClustCnt in clustCntVec:
            foundResult = False
            for resultList in resultDict:
                if (resultList[1] == clusterModel and resultList[2] == curClustCnt):
                    str2disp = headerStrFormat + "=" + valuesStrFormat
                    data2disp = (baseResultFileName, resultList[1], resultList[2],
                                 resultList[3][0],
                                 resultList[3][1], resultList[3][2], resultList[3][3],
                                 resultList[3][4], resultList[3][5], resultList[3][6], resultList[3][7],
                                 resultList[3][8])
                    #histCnt=', resultList[3][9][0:10])
                    print(str2disp % data2disp)
                    foundResult = True
                if foundResult:
                    if not enforce_rerun:
                        break
            predictionFileName = baseResultFileName.replace("_baseResults.npy", "") + "_" + clusterModel + "_" + str(curClustCnt) + ".npz"
            predictionFileNameFull = os.path.join(results_dir, 'baseResults', predictionFileName)
            predictionFileExist = os.path.isfile(predictionFileNameFull)
            if not foundResult or not predictionFileExist or enforce_rerun:
                if foundResult and not predictionFileExist:
                    print('running again for saving predictions')
                if enforce_rerun:
                    print('rerunning the experiment is enforced')
                t = time.time()
                print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
                print(featsName, 'clusterModel(', clusterModel, '), clusterCount(', curClustCnt, ') running.')
                predClusters, kluster_centers = funcH.clusterData(featVec=ft, n_clusters=curClustCnt, normMode='', applyPca=False, clusterModel=clusterModel)
                print('elapsedTime(', funcH.getElapsedTimeFormatted(time.time() - t), ')')

                centroid_info_pdf = funcH.get_cluster_centroids(ft, predClusters, kluster_centers, verbose=0)

                nmi_cur = 100*funcH.get_nmi_only(labels_all, predClusters)
                _confMat_mapped_preds_center, _, kr_pdf_center, weightedPurity_center, cnmxh_perc = funcH.countPredictionsForConfusionMat(labels_all, predClusters, labelNames=class_names, centroid_info_pdf=centroid_info_pdf, verbose=0)
                _confMat_mapped_preds_mxhist, _, kr_pdf_mxhist, weightedPurity_mxhist, _ = funcH.countPredictionsForConfusionMat(labels_all, predClusters, labelNames=class_names, centroid_info_pdf=None, verbose=0)
                acc_center = 100*np.sum(np.diag(_confMat_mapped_preds_center)) / np.sum(np.sum(_confMat_mapped_preds_center))
                acc_mxhist = 100*np.sum(np.diag(_confMat_mapped_preds_mxhist)) / np.sum(np.sum(_confMat_mapped_preds_mxhist))
                numOf_1_sample_bins, histSortedInv = funcH.analyzeClusterDistribution(predClusters, curClustCnt, verbose=2)
                meanPurity_center = np.mean(np.asarray(kr_pdf_center["%purity"]))
                meanPurity_mxhist = np.mean(np.asarray(kr_pdf_mxhist["%purity"]))

                resultList = [featsName, clusterModel, curClustCnt, [nmi_cur, acc_center, meanPurity_center, weightedPurity_center, acc_mxhist, meanPurity_mxhist, weightedPurity_mxhist, cnmxh_perc, numOf_1_sample_bins, histSortedInv]]
                resultDict.append(resultList)

                print(valuesStrFormat % (nmi_cur,  acc_center, meanPurity_center, weightedPurity_center, acc_mxhist, meanPurity_mxhist, weightedPurity_mxhist, cnmxh_perc, numOf_1_sample_bins))
                print("histogram of clusters max first 10", resultList[3][9][0:10])
                if foundResult and predictionFileExist and enforce_rerun:
                    print("the results wont be saved due to the rerunning situation")
                    break
                np.save(baseResultFileNameFull, resultDict, allow_pickle=True)
                np.savez(predictionFileNameFull, labels_all, predClusters)

    np.set_printoptions(prevPrintOpts)
    return resultDict, centroid_info_pdf

def loadBaseResult(fileName):
    results_dir = funcH._CONF_PARAMS_.DIR.RESULTS  # '/media/dg/SSD_Data/DataPath/bdResults'
    baseLineResultFolder = os.path.join(results_dir, 'baseResults')  # '/media/dg/SSD_Data/DataPath/bdResults/baseResults'
    if not str(fileName).endswith(".npz"):
        fileName = fileName + ".npz"
    fileName = os.path.join(baseLineResultFolder, fileName)
    #print("fileName=", fileName)
    preds = np.load(fileName)
    labels_true = np.asarray(preds['arr_0'], dtype=int)
    labels_pred = np.asarray(preds['arr_1'], dtype=int)
    return labels_true, labels_pred

def getBaseResults_Aug2020(dataToUse, pcaCount, numOfSigns, displayResults=True, baseResultFileName=''):
    if baseResultFileName == '':
        baseResultFileName = funcD.getFileName(dataToUse=dataToUse, normMode="", pcaCount=pcaCount,
                                               numOfSigns=numOfSigns, expectedFileType='BaseResultName')
    results_dir = funcH._CONF_PARAMS_.DIR.RESULTS  # '/media/dg/SSD_Data/DataPath/bdResults'
    baseLineResultFolder = os.path.join(results_dir, 'baseResults')  # '/media/dg/SSD_Data/DataPath/bdResults/baseResults'
    baseResultFileNameFull = os.path.join(baseLineResultFolder, baseResultFileName)
    resultDict = np.load(baseResultFileNameFull, allow_pickle=True)
    returnDict = []
    for resultList in resultDict:
        clusterModel = resultList[1]
        clusterCount = resultList[2]
        nmiAll = resultList[3][0]
        acc_cent = resultList[3][1]
        meanPurity_cent = resultList[3][2]
        weightedPurity_cent = resultList[3][3]
        acc_mxhs = resultList[3][4]
        meanPurity_mxhs = resultList[3][5]
        weightedPurity_mxhs = resultList[3][6]
        cnmxh_perc = resultList[3][7]
        emptyK = resultList[3][8]
        dataUsed = baseResultFileName.replace('.npy', '').replace('_baseResults', '')
        returnDict.append([dataUsed, clusterModel, clusterCount, nmiAll, acc_cent, meanPurity_cent, weightedPurity_cent, acc_mxhs, meanPurity_mxhs, weightedPurity_mxhs, cnmxh_perc, emptyK])

    df = pd.DataFrame(returnDict, columns=['npyFileName', 'clusModel', 'clusCnt', 'nmiAll', 'acc_cent', 'meanPurity_cent', 'weightedPurity_cent', 'acc_mxhs', 'meanPurity_mxhs', 'weightedPurity_mxhs', 'cnmxh_perc', 'emptyK'])
    funcH.setPandasDisplayOpts()
    if displayResults:
        print(df)

    return returnDict

#SUBMIT_CHECKED
def traverse_base_results_folder_hospising():
    """
    Looks into the baseResults folder and displays all results as a Pandas Dataframe
    Saving the result table as "baseResults/baseLineResults.csv"
    """
    results_dir = funcH._CONF_PARAMS_.DIR.RESULTS  # '/media/dg/SSD_Data/DataPath/bdResults'
    baseLineResultFolder = os.path.join(results_dir, 'baseResults')  # '/media/dg/SSD_Data/DataPath/bdResults/baseResults'
    fileList = funcH.getFileList(dir2Search=baseLineResultFolder, startString='', endString='.npy')
    brAll = []
    for f in fileList:
        br = getBaseResults_Aug2020(dataToUse='', pcaCount=-1, numOfSigns=-1, displayResults=False, baseResultFileName=f)
        brAll = brAll + br
    returnDictAll = pd.DataFrame(brAll, columns=['npyFileName', 'clusModel', 'clusCnt', 'nmiAll', 'acc_cent', 'meanPurity_cent', 'weightedPurity_cent', 'acc_mxhs', 'meanPurity_mxhs', 'weightedPurity_mxhs', 'cnmxh_perc', 'emptyK']).sort_values(by='cnmxh_perc', ascending=False)
    print(returnDictAll)

    baseResults_csv = os.path.join(baseLineResultFolder, 'baseLineResults.csv')
    returnDictAll.to_csv(path_or_buf=baseResults_csv, sep=',', na_rep='NaN', float_format='%8.4f')

#SUBMIT_CHECK
def runForPred(labels_true, labels_pred, labelNames, predictDefStr):
    print("\r\n*-*-*start-", predictDefStr, "-end*-*-*\r\n")
    print("\r\n\r\n*-*-", predictDefStr, "calcClusterMetrics-*-*\r\n\r\n")
    klusRet = funcH.calcClusterMetrics(labels_true, labels_pred, removeZeroLabels=False, labelNames=labelNames)

    print("\r\n\r\n*-*-", predictDefStr, "calcCluster2ClassMetrics-*-*\r\n\r\n")
    classRet, _confMat, c_pdf, kr_pdf = funcH.calcCluster2ClassMetrics(labels_true, labels_pred, removeZeroLabels=False, labelNames=labelNames, predictDefStr=predictDefStr)

    results_dir = funcH._CONF_PARAMS_.DIR.RESULTS
    predictResultFold = os.path.join(results_dir, "predictionResults")
    funcH.createDirIfNotExist(predictResultFold)

    confMatFileName = predictDefStr + ".csv"
    confMatFileName = os.path.join(predictResultFold, confMatFileName)
    _confMat_df = pd.DataFrame(data=_confMat, index=labelNames, columns=labelNames)
    # _confMat_df = _confMat_df[(_confMat_df.T != 0).any()]
    pd.DataFrame.to_csv(_confMat_df, path_or_buf=confMatFileName)

    kr_pdf_FileName = "kluster_evaluations_" + predictDefStr + ".csv"
    kr_pdf_FileName = os.path.join(predictResultFold, kr_pdf_FileName)
    pd.DataFrame.to_csv(kr_pdf, path_or_buf=kr_pdf_FileName)

    c_pdf_FileName = "class_evaluations_" + predictDefStr + ".csv"
    c_pdf_FileName = os.path.join(predictResultFold, c_pdf_FileName)
    pd.DataFrame.to_csv(c_pdf, path_or_buf=c_pdf_FileName)

    print("*-*-*end-", predictDefStr, "-end*-*-*\r\n")
    return klusRet, classRet, _confMat, c_pdf, kr_pdf

def load_labels_pred_for_ensemble_hospising(class_names, dataToUseVec=["hog", "sn", "sk"], dataset_ident_str="Dev", pcaCountVec=[96, 256],
                                          clusterModelsVec=["KMeans"], clustCntVec=[256]):
    results_dir = funcH._CONF_PARAMS_.DIR.RESULTS
    nos = get_nos(dataset_ident_str)
    predictionsDict = []
    N = 0
    for dataToUse in dataToUseVec:
        for pcaCount in pcaCountVec:
            baseResultFileName = funcD.getFileName(dataToUse=dataToUse, normMode="", pcaCount=pcaCount, numOfSigns=nos,
                                                   expectedFileType='BaseResultName')
            for clusterModel in clusterModelsVec:
                for curClustCnt in clustCntVec:
                    predictionFileName = baseResultFileName.replace("_baseResults.npy",
                                                                    "") + "_" + clusterModel + "_" + str(
                        curClustCnt) + ".npz"
                    predictionFileNameFull = os.path.join(results_dir, 'baseResults', predictionFileName)
                    if os.path.isfile(predictionFileNameFull):
                        print("EXIST - ", predictionFileNameFull)

                        predStr = predictionFileName.replace(".npy", "").replace(".npz", "")

                        a = np.load(predictionFileNameFull)
                        labels_all = a["arr_0"]
                        predClusters = a["arr_1"]
                        _confMat_mapped_preds_mxhist, _, kr_pdf_mxhist, weightedPurity_mxhist, _ = funcH.countPredictionsForConfusionMat(
                            labels_all, predClusters, labelNames=class_names, centroid_info_pdf=None, verbose=0)
                        print("predStr", predStr)
                        print("preds(", predClusters, ") loaded of size", predClusters.shape)
                        predictionsDict.append({"str": predStr, "prd": predClusters})
                        N = N + 1

    cluster_runs = None
    for i in range(0, N):
        cluster_runs = funcH.append_to_vstack(cluster_runs, predictionsDict[i]["prd"], dtype=int)

    return class_names, labels_all, predictionsDict, cluster_runs, N

#SUBMIT_CHECK
def ensemble_cluster_analysis(cluster_runs, predictionsDict, labels,
                     consensus_clustering_max_k=256, useNZ=True, dataset_ident_str="Dev",
                     resultsToCombineDescriptorStr="",
                     labelNames = None, verbose=False):
    nos = get_nos(dataset_ident_str)
    N = cluster_runs.shape[0]
    results_dir = funcH._CONF_PARAMS_.DIR.RESULTS
    predictResultFold = os.path.join(results_dir, "predictionResults")

    # 1.run cluster_ensembles
    t = time.time()
    consensus_clustering_labels = funcEns.get_consensus_labels(cluster_runs,
                                                       consensus_clustering_max_k=consensus_clustering_max_k,
                                                       verbose=verbose)
    elapsed_cluster_ensembles = time.time() - t
    print('cluster_ensembles - elapsedTime({:4.2f})'.format(elapsed_cluster_ensembles), ', ended at ',
          datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    # 2.append ensembled consensus_clustering_labels into cluster_runs dictionary
    if resultsToCombineDescriptorStr == "":
        resultsToCombineDescriptorStr = "klusterResults_" + str(nos) + ("_nz_" if useNZ else "_") + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    resultsToCombineDescriptorStr = "klusterResults_" + str(nos) + ("_nz_" if useNZ else "_") + resultsToCombineDescriptorStr

    predictionsDict.append({"str": resultsToCombineDescriptorStr, "prd": consensus_clustering_labels})
    cluster_runs = funcH.append_to_vstack(cluster_runs, consensus_clustering_labels, dtype=int)

    # 3.for all clusterings run analysis of clusters and classes
    resultsDict = []
    for i in range(0, N + 1):
        t = time.time()
        klusRet, classRet, _confMat, c_pdf, kr_pdf = runForPred(labels, predictionsDict[i]["prd"], labelNames, predictionsDict[i]["str"])
        elapsed_runForPred = time.time() - t
        print('runForPred(', predictionsDict[i]["str"], ') - elapsedTime({:4.2f})'.format(elapsed_runForPred),
              ' ended at ', datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

        resultsDict.append({"klusRet": klusRet, "classRet": classRet,
                            "_confMat": _confMat, "c_pdf": c_pdf, "kr_pdf": kr_pdf})

    klusRet = resultsDict[0]["klusRet"].copy().rename(columns={"value": predictionsDict[0]["str"]})
    for i in range(1, N + 1):
        klusRet.insert(i + 1, predictionsDict[i]["str"], resultsDict[i]["klusRet"]['value'], True)
    print("\r\ncluster metrics comparison\r\n")
    print(klusRet)
    print("\r\n")

    classRet = resultsDict[0]["classRet"].copy().rename(columns={"value": predictionsDict[0]["str"]})
    for i in range(1, N + 1):
        classRet.insert(i + 1, predictionsDict[i]["str"], resultsDict[i]["classRet"]['value'], True)
    if verbose:
        print("\r\nclassification metrics comparison\r\n")
        print(classRet)
        print("\r\n")
    class_metrics_FileName = os.path.join(predictResultFold, resultsToCombineDescriptorStr.replace("klusterResults", "class_metrics") + ".csv")
    pd.DataFrame.to_csv(classRet, path_or_buf=class_metrics_FileName)

    c_pdf = resultsDict[0]["c_pdf"][['class', '%f1']].sort_index().rename(
        columns={"class": "f1Score", "%f1": predictionsDict[0]["str"]})
    for i in range(1, N + 1):
        c_pdf.insert(i + 1, predictionsDict[i]["str"], resultsDict[i]["c_pdf"][['%f1']].sort_index(), True)
    if verbose:
        print("\r\nf1 score comparisons for classes\r\n")
        print(c_pdf)
        print("\r\n")
    f1_comparison_FileName = os.path.join(predictResultFold, resultsToCombineDescriptorStr.replace("klusterResults", "f1_comparison") + ".csv")
    pd.DataFrame.to_csv(c_pdf, path_or_buf=f1_comparison_FileName)

    print('calc_ensemble_driven_cluster_index - started at ', datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    t = time.time()
    eci_vec, clusterCounts = funcEns.calc_ensemble_driven_cluster_index(cluster_runs=cluster_runs)
    elapsed = time.time() - t
    print('calc_ensemble_driven_cluster_index - elapsedTime({:4.2f})'.format(elapsed), ' ended at ',
          datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    print('create_LWCA_matrix - started at ', datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    t = time.time()
    lwca_mat = funcEns.create_LWCA_matrix(cluster_runs, eci_vec=eci_vec, verbose=0)
    elapsed = time.time() - t
    print('create_LWCA_matrix - elapsedTime({:4.2f})'.format(elapsed), ' ended at ',
          datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    print('create_quality_vec - started at ', datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    t = time.time()
    quality_vec = funcEns.calc_quality_weight_basic_clustering(cluster_runs, logType=0, verbose=0)
    elapsed = time.time() - t
    print('create_quality_vec - elapsedTime({:4.2f})'.format(elapsed), ' ended at ',
          datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    sampleCntToPick = np.array([1, 3, 5, 10], dtype=int)
    columns = ['1', '3', '5', '10']
    columns_2 = ['c1', 'c3', 'c5', 'c10']
    colCnt = len(columns)

    resultTable_mat = np.zeros((colCnt+2,N+1))
    resultTable_pd_columns = ['c1', 'c3', 'c5', 'c10', 'All', 'sum']
    resultTable_pd_index = []
    resultTable_FileName = os.path.join(predictResultFold, resultsToCombineDescriptorStr.replace("klusterResults", "resultTable") + ".csv")

    #cluster_runs_cmbn = []
    for i in range(0, N+1):
        kr_pdf_cur = resultsDict[i]["kr_pdf"]
        eci_vec_cur = eci_vec[i].copy()
        predictDefStr = predictionsDict[i]["str"]
        resultTable_pd_index.append(predictDefStr)
        #cluster_runs_cmbn = funcH.append_to_vstack(cluster_runs_cmbn, predictionsDict[i]["prd"], dtype=int)
        print(predictDefStr, "Quality of cluster = {:6.4f}".format(quality_vec[i]), "number of clusters : ", kr_pdf_cur.shape)
        predictions_cur = predictionsDict[i]["prd"]
        unique_preds = np.unique(predictions_cur)

        kr_pdf_cur.sort_index(inplace=True)
        eci_N = np.array(eci_vec_cur * kr_pdf_cur['N'], dtype=float)
        eci_pd = pd.DataFrame(eci_vec_cur, columns=['ECi'])
        eci_N_pd = pd.DataFrame(eci_N, columns=['ECi_n'])
        pd_comb = pd.concat([kr_pdf_cur, eci_pd, eci_N_pd], axis=1)
        pd_comb.sort_values(by=['ECi_n', 'N'], inplace=True, ascending=[False, False])

        kr_pdf_FileName = "kluster_evaluations_" + predictDefStr + ".csv"
        kr_pdf_FileName = os.path.join(predictResultFold, kr_pdf_FileName)

        cols2add = np.zeros((clusterCounts[i], colCnt), dtype=float)
        cols2add_pd = pd.DataFrame(cols2add, columns=columns)
        cols2add_2 = np.zeros((clusterCounts[i], colCnt), dtype=float)
        cols2add_2_pd = pd.DataFrame(cols2add_2, columns=columns_2)
        pd_comb = pd.concat([kr_pdf_cur, eci_pd, eci_N_pd, cols2add_pd, cols2add_2_pd], axis=1)

        pd_comb.sort_index(inplace=True)
        pd.DataFrame.to_csv(pd_comb, path_or_buf=kr_pdf_FileName)

        # pick first 10 15 20 25 samples according to lwca_mat
        for pi in range(0, clusterCounts[i]):
            cur_pred = unique_preds[pi]
            predictedSamples = funcH.getInds(predictions_cur, cur_pred)
            sampleLabels = labels[predictedSamples]
            lwca_cur = lwca_mat[predictedSamples, :]
            lwca_cur = lwca_cur[:, predictedSamples]
            simSum = np.sum(lwca_cur, axis=0) + np.sum(lwca_cur, axis=1).T
            v, idx = funcH.sortVec(simSum)
            sortedPredictionsIdx = predictedSamples[idx]
            sortedLabelIdx = labels[sortedPredictionsIdx]
            curSampleCntInCluster = len(sampleLabels)
            mappedClassOfKluster = funcH.get_most_frequent(list(sortedLabelIdx))

            for sj in range(0, colCnt):
                sCnt = sampleCntToPick[sj] if curSampleCntInCluster>sampleCntToPick[sj] else curSampleCntInCluster
                sampleLabelsPicked = sortedLabelIdx[:sCnt]
                purity_k, correctLabelInds, mappedClass, _ = funcH.calcPurity(list(sampleLabelsPicked))
                if mappedClass == mappedClassOfKluster:
                    cols2add[pi, sj] = purity_k
                else:
                    cols2add[pi, sj] = -mappedClass+(mappedClassOfKluster/100)
                cols2add_2[pi, sj] = np.sum(sortedLabelIdx == mappedClass)

        cols2add_pd = pd.DataFrame(cols2add, columns=columns)
        cols2add_2_pd = pd.DataFrame(cols2add_2, columns=columns_2)
        pd_comb = pd.concat([kr_pdf_cur, eci_pd, eci_N_pd, cols2add_pd, cols2add_2_pd], axis=1)
        pd_comb.sort_index(inplace=True)
        pd.DataFrame.to_csv(pd_comb, path_or_buf=kr_pdf_FileName)

        allPredCorrectSum = np.asarray(np.sum(kr_pdf_cur.iloc[:,2:3]))
        numOfSamples = np.asarray(np.sum(kr_pdf_cur.iloc[:, 3:4]))
        precidtCols = np.sum(cols2add_2, axis=0, keepdims=True)
        resultTable_mat[0:colCnt, i] = precidtCols.T.squeeze()
        resultTable_mat[colCnt, i] = allPredCorrectSum
        resultTable_mat[colCnt+1, i] = numOfSamples

    resultTable_mat[:-1, :] = resultTable_mat[:-1, :] / resultTable_mat[-1, :]
    resultTable_pd = pd.DataFrame(resultTable_mat, index=resultTable_pd_columns, columns=resultTable_pd_index)
    pd.DataFrame.to_csv(resultTable_pd, path_or_buf=resultTable_FileName)

    _confMat_consensus, _, kr_pdf_consensus, weightedPurity_consensus, _ = funcH.countPredictionsForConfusionMat(labels, consensus_clustering_labels, labelNames=labelNames, centroid_info_pdf=None, verbose=0)
    meanPurity_consensus = np.mean(np.asarray(kr_pdf_consensus["%purity"]))
    nmi_consensus = funcH.get_nmi_only(labels, consensus_clustering_labels)
    acc_consensus = np.sum(np.diag(_confMat_consensus)) / np.sum(np.sum(_confMat_consensus))
    print(resultsToCombineDescriptorStr, "\nnmi_consensus", nmi_consensus, "acc_consensus", acc_consensus, "meanPurity_consensus", meanPurity_consensus)
    del(_confMat_consensus, kr_pdf_consensus, weightedPurity_consensus)

#SUBMIT_CHECKED
def prepare_hospisign_data(X, y, detailedLabels, validUser, testUser):
    print("y.dtype=", type(y))
    y = np.asarray(y, dtype=int)
    uniqLabels = np.unique(y)
    #print("uniqLabels=", uniqLabels)
    #print("uniqLabels2=", np.unique(detailedLabels[:, 2]))

    tr_inds = np.array([], dtype=int)
    va_inds = np.array([], dtype=int)
    te_inds = np.array([], dtype=int)
    empty_test_classes = {}
    empty_validation_classes = {}
    for label in uniqLabels:
        inds = funcH.getInds(y, label)
        cnt = len(inds)

        if cnt > 0:
            # userid = testUser to test
            # userid = validUser to valid
            # others to train
            userIDs = detailedLabels[inds, 1]
            teIDs = funcH.getInds(userIDs, testUser)
            if teIDs.shape[0] > 0:
                teIndsCur = inds[teIDs]
                te_inds = np.concatenate((te_inds, teIndsCur))
            else:
                teIndsCur = []
                empty_test_classes[label] = 1

            vaIDs = funcH.getInds(userIDs, validUser)
            if vaIDs.shape[0] > 0:
                vaIndsCur = inds[vaIDs]
                va_inds = np.concatenate((va_inds, vaIndsCur))
            else:
                vaIndsCur = []
                empty_validation_classes[label] = 1

            usedInds = np.squeeze(np.concatenate((teIDs, vaIDs)))

            allInds = np.arange(0, len(inds))
            trIDs = np.delete(allInds, usedInds)
            trIndsCur = inds[trIDs]
            tr_inds = np.concatenate((tr_inds, trIndsCur))

            trCnt = len(trIndsCur)
            vaCnt = len(vaIndsCur)
            teCnt = len(teIndsCur)

            if cnt != trCnt + vaCnt + teCnt:
                print("xxxxxx cnt all = ", cnt, "!=", trCnt + vaCnt + teCnt)
            print("label(", label, "),cnt(", cnt, "),trCnt(", trCnt, "),vaCnt(", vaCnt, "),teCnt(", teCnt, ")")
    print("len train = ", len(tr_inds))
    print("len valid = ", len(va_inds))
    print("len test = ", len(te_inds))
    print("len all = ", len(y), "=", len(tr_inds) + len(va_inds) + len(te_inds))

    print("empty_test_classes", empty_test_classes)
    print("empty_validation_classes", empty_validation_classes)

    dataset_tr = funcD.HandCraftedDataset("", X=X[tr_inds, :], y=y[tr_inds])
    train_dl = DataLoader(dataset_tr, batch_size=32, shuffle=True)
    dataset_va = funcD.HandCraftedDataset("", X=X[va_inds, :], y=y[va_inds])
    valid_dl = DataLoader(dataset_va, batch_size=1024, shuffle=False)
    dataset_te = funcD.HandCraftedDataset("", X=X[te_inds, :], y=y[te_inds])
    test_dl = DataLoader(dataset_te, batch_size=1024, shuffle=False)

    return train_dl, valid_dl, test_dl

#SUBMIT_CHECKED
def get_hospisign_labels(dataset_ident_str="Dev", sort_by=None, verbose=0):
    list_dict_file = os.path.join(funcH._CONF_PARAMS_.DIR.DATA, get_hospisign_file_name(data_ident_vec="list_dict", dataset_ident_str=dataset_ident_str))
    a = pd.read_csv(list_dict_file, delimiter="*", header=None,
                    names=["sign", "user", "rep", "frameID", "khsID", "khsName", "hand"])
    b, uniqKHSinds = np.unique(np.asarray(a["khsID"]), return_index=True)
    labelsAll = np.asarray(a["khsID"], dtype=int)
    namesAll = np.asarray(a["khsName"])

    if verbose > 0:
        print(len(np.unique(namesAll)))
        print(np.unique(namesAll))

    labels_sui = np.squeeze(np.asarray(a[["sign", "user", "khsID"]]))
    # get sort index first
    assignedKHSinds = labelsAll[uniqKHSinds]
    selectedKHSnames = []
    for cur_ind in uniqKHSinds:
        cur_label = labelsAll[cur_ind]
        label_samples = funcH.getInds(labelsAll, cur_label)
        label_names = namesAll[label_samples]
        cur_names = np.asarray([np.char.strip(nm) for nm in np.unique(label_names)])
        cur_name = '|'.join(cur_names)
        selectedKHSnames.append(cur_name)

    selectedKHSnames = np.asarray(selectedKHSnames)
    if verbose > 1:
        print(labelsAll.shape, labelsAll.dtype)
    sortedLabelsAll, sortedLabelsMap = funcH.reset_labels(labelsAll, assignedKHSinds, selectedKHSnames, sortBy=sort_by,
                                                          verbose=verbose)
    if verbose > 1:
        print("sortedLabelsAll:\n", sortedLabelsAll.head())
        print("sortedLabelsMap:\n", sortedLabelsMap)
        print(labels_sui.shape, labels_sui.dtype)
    labels_sui[:, 2] = np.squeeze(np.array(sortedLabelsAll))

    lb_map = np.vstack((sortedLabelsMap["labelIDs"], sortedLabelsMap["labelStrings"])).T

    x = Counter(np.squeeze(labelsAll).astype(int))

    khsCntVec = [v for k, v in x.most_common()]
    khsIndex = [k for k, v in x.most_common()]
    if verbose > 2:
        print("x:\n", x)
        khsNameCol = [str(np.squeeze(lb_map[np.where(lb_map[:, 0] == k), 1])) for k, v in x.most_common()]
        print("khsNameCol:\n", khsNameCol)
        print("khsCntVec:\n", khsCntVec)
        print("khsIndex:\n", khsIndex)
    khsCntCol = np.asarray(khsCntVec)[np.argsort(khsIndex)]
    if verbose > 2:
        print("khsCntVec(sorted accordingly):\n", khsCntCol)

    lb_map_new = pd.DataFrame({"khsID": lb_map[:, 0], "khsName": lb_map[:, 1], "khsCnt": khsCntCol})
    lb_map_cnt = lb_map_new.sort_values(by='khsCnt', ascending=False)
    lb_map_id = lb_map_new.sort_values(by='khsID', ascending=True)
    lb_map_name = lb_map_new.sort_values(by='khsName', ascending=True)
    if verbose > 1:
        print("lb_map_cnt=\n", lb_map_cnt)
        print("lb_map_id=\n", lb_map_id)
        print("lb_map_name=\n", lb_map_name)

    hospisign_labels = {
        "labels": sortedLabelsAll,
        "labels_sui": labels_sui,
        "khsInds": sortedLabelsMap["labelIDs"],
        "khsNames": sortedLabelsMap["labelStrings"],
        "label_map": lb_map_id,
        "label_map_cnt": lb_map_cnt,
        "label_map_name": lb_map_name,
    }
    return hospisign_labels

#SUBMIT_CHECKED
def get_hospisign_feats(dataset_ident_str = "Dev", labelsSortBy=None, verbose=0):
    hogMat = os.path.join(funcH._CONF_PARAMS_.DIR.DATA, get_hospisign_file_name(data_ident_vec="hog", dataset_ident_str=dataset_ident_str))
    skelMat = os.path.join(funcH._CONF_PARAMS_.DIR.DATA, get_hospisign_file_name(data_ident_vec="skeleton", dataset_ident_str=dataset_ident_str))
    snMat = os.path.join(funcH._CONF_PARAMS_.DIR.DATA, get_hospisign_file_name(data_ident_vec="snv", dataset_ident_str=dataset_ident_str))

    hg_ft = funcH.loadMatFile(hogMat, verbose=verbose)
    sn_ft = funcH.loadMatFile(snMat, verbose=verbose)
    sk_ft = funcH.loadMatFile(skelMat, verbose=verbose)

    hg_ft = hg_ft['hogImArr']
    sn_ft = sn_ft['mat2sa']
    sk_ft = sk_ft['mat2sa']

    if verbose > 0:
        print("hog = ", hg_ft.shape, hg_ft.dtype)
        print("surfNorm = ", sn_ft.shape, sn_ft.dtype)
        print("skeleton = ", sk_ft.shape, sk_ft.dtype)

    hospisign_labels = get_hospisign_labels(dataset_ident_str=dataset_ident_str, sort_by=labelsSortBy)
    labels = hospisign_labels["labels"]
    labels_sui = hospisign_labels["labels_sui"]
    label_map = hospisign_labels["label_map"]

    if verbose > 0:
        print("labels_sui = ", labels_sui.shape, labels_sui.dtype)
        print("labels = ", labels.shape, type(labels))
    ft = {
        "hg": hg_ft,
        "sn": sn_ft,
        "sk": sk_ft,
    }
    lab = {
        "labels_sui": labels_sui,
        "labels": labels,
        "label_map": label_map,
    }
    return ft, lab

def f_apply_pca(ft, data_ident, pca_dim):
    #  ftpca, exp_vec, exp_val = f_apply_pca(ft, "hg", pca_dim)
    ftpca, exp_vec = funcH.applyMatTransform(ft[data_ident], applyPca=True, normMode="", verbose=1)
    data_dim = ftpca.shape[1]
    print("data_dim({}),pca_dim({}),exp_vec({})".format(data_dim,pca_dim,len(exp_vec)))
    slc_dim = min(data_dim,pca_dim)
    ftpca = ftpca[:, 0:slc_dim]
    return ftpca, exp_vec, exp_vec[slc_dim-1]

#SUBMIT_CHECKED
def combine_pca_hospisign_data(data_ident, pca_dim, dataset_ident_str="Dev", verbose=1, concat_method=1):
    ft, lab = get_hospisign_feats(dataset_ident_str=dataset_ident_str, verbose=verbose - 1)
    if verbose > 0:
        print('hg_ft.shape = ', ft["hg"].shape, ', sn_ft.shape = ', ft["sn"].shape, ', sk_ft.shape = ', ft["sk"].shape)
        print("hg - min(", ft["hg"].min(), "), max(", ft["hg"].max(), ")")
        print("sn - min(", ft["sn"].min(), "), max(", ft["sn"].max(), ")")
        print("sk - min(", ft["sk"].min(), "), max(", ft["sk"].max(), ")")

    nan_sk, nan_sn, nan_hg = list(), list(), list()
    for i in range(ft["sk"].shape[0]):
        if (np.isnan(ft["sk"][i]).any()):
            nan_sk.append(i)
            ft["sk"][i] = np.zeros(ft["sk"][i].shape)  # print(i,":",ft["sk"][i]) #print(ft["sk"][i].shape)
        if (np.isnan(ft["sn"][i]).any()):
            nan_sn.append(i)
            ft["sn"][i] = np.zeros(ft["sn"][i].shape)  # print(i,":",ft["sn"][i]) #print(ft["sn"][i].shape)
        if (np.isnan(ft["hg"][i]).any()):
            nan_hg.append(i)
            ft["hg"][i] = np.zeros(ft["hg"][i].shape)  # print(i,":",ft["hg"][i]) #print(ft["hg"][i].shape)

    if nan_hg:
        nan_hg = np.squeeze(np.vstack(nan_hg))
        if verbose > 0:
            print("nan_hg: ", nan_hg)
    if nan_sn:
        nan_sn = np.squeeze(np.vstack(nan_sn))
        if verbose > 0:
            print("nan_sn: ", nan_sn)
    if nan_sk:
        nan_sk = np.squeeze(np.vstack(nan_sk))
        if verbose > 0:
            print("nan_sk: ", nan_sk)

    explainibility_vec_dict = {"hg": np.nan, "sn": np.nan, "sk": np.nan, data_ident: np.nan}
    explainibility_val_dict = {"hg": np.nan, "sn": np.nan, "sk": np.nan, data_ident: np.nan}

    if (data_ident == "hog" or data_ident == "hg"):
        feats = ft["hg"].copy()
    elif (data_ident == "skeleton" or data_ident == "sk"):
        feats = ft["sk"].copy()
    elif (data_ident == "snv" or data_ident == "sn"):
        feats = ft["sn"].copy()
    elif concat_method == 1:
        if (data_ident == "hgsk"):
            feats = np.concatenate([ft["hg"].T, ft["sk"].T]).T
        elif (data_ident == "hgsn"):
            feats = np.concatenate([ft["hg"].T, ft["sn"].T]).T
        elif (data_ident == "snsk"):
            feats = np.concatenate([ft["sn"].T, ft["sk"].T]).T
        elif (data_ident == "hgsnsk"):
            feats = np.concatenate([ft["hg"].T, ft["sn"].T, ft["sk"].T]).T
    else:  # first get pca of the original data then
        ft_hg_pca, explainibility_vec_dict["hg"], explainibility_val_dict["hg"] = f_apply_pca(ft, "hg", pca_dim)
        ft_sk_pca, explainibility_vec_dict["sk"], explainibility_val_dict["sk"] = f_apply_pca(ft, "sk", pca_dim)
        ft_sn_pca, explainibility_vec_dict["sn"], explainibility_val_dict["sn"] = f_apply_pca(ft, "sn", pca_dim)
        if (data_ident == "hgsk"):
            feats = np.concatenate([ft_hg_pca.T, ft_sk_pca.T]).T
        elif (data_ident == "hgsn"):
            feats = np.concatenate([ft_hg_pca.T, ft_sn_pca.T]).T
        elif (data_ident == "snsk"):
            feats = np.concatenate([ft_sn_pca.T, ft_sk_pca.T]).T
        elif (data_ident == "hgsnsk"):
            feats = np.concatenate([ft_hg_pca.T, ft_sn_pca.T, ft_sk_pca.T]).T

    feats_pca, exp_var_rat = funcH.applyMatTransform(feats, applyPca=True, normMode="", verbose=verbose)
    feats = feats_pca[:, 0:pca_dim]
    explainibility_vec_dict[data_ident] = exp_var_rat
    explainibility_val_dict[data_ident] = exp_var_rat[pca_dim]
    print(data_ident, '.shape = ', feats.shape, ' loaded.', explainibility_val_dict[data_ident], '<--explainibility.')

    return feats, lab["labels"], lab["labels_sui"], lab["label_map"], explainibility_vec_dict, explainibility_val_dict

#SUBMIT_CHECK
def get_result_table_out(result_file_name_full, class_names):
    a = np.load(result_file_name_full)
    print(a.files)

    dataIdent_ = a["dataIdent_"]
    testUser_ = a["testUser_"]
    validUser_ = a["validUser_"]
    hid_state_cnt_vec_ = a["hid_state_cnt_vec_"]

    accvectr = a["accvectr_"]
    accvecva = a["accvecva_"]
    accvecte = a["accvecte_"]
    bestVaID = np.argmax(accvecva)
    bestTeID = np.argmax(accvecte)
    formatStr = "5.3f"
    print(("bestTeID({:" + formatStr + "}),vaAcc({:" + formatStr + "}),teAcc({:" + formatStr + "})").format(bestTeID,accvecva[bestTeID],accvecte[bestTeID]))
    print(("last, vaAcc({:" + formatStr + "}),teAcc({:" + formatStr + "})").format(accvecva[-1], accvecte[-1]))
    print("accvectr_=", accvectr.shape, ", accvecva_=", accvecva.shape, ", accvecte_=", accvecte.shape)

    preds_best_ = np.squeeze(a["preds_best_"])
    labels_best_ = np.squeeze(a["labels_best_"])
    print("preds_best_=", preds_best_.shape, ", labels_best_=", labels_best_.shape)

    # check if exist. if yes go on
    # uniqLabs = np.unique(labels_best_)
    # classCount = len(uniqLabs)
    # print("uniqLabs=", uniqLabs, ", classCount_=", classCount)

    conf_mat_ = confusion_matrix(labels_best_, preds_best_)
    print(conf_mat_.shape, class_names.shape)
    saveConfFileName_full = result_file_name_full.replace('.npz', '.png')
    if not os.path.exists(saveConfFileName_full):
        print("saving--", saveConfFileName_full)
        try:
            funcH.plot_confusion_matrix(conf_mat_, class_names=class_names, saveConfFigFileName=saveConfFileName_full,
                                        confusionTreshold=0.2, show_only_confused=True)
        except:
            pass

    confMatStats, df_slctd_table = funcH.calcConfusionStatistics(conf_mat_, categoryNames=class_names,
                                                                 selectedCategories=None, verbose=0)
    df_slctd_table = df_slctd_table.sort_values(["F1_Score"], ascending=False)
    print(df_slctd_table)
    saveF1TableFileName_full = result_file_name_full.replace('.npz', '_F1.csv')
    if not os.path.exists(saveF1TableFileName_full):
        print("saving--", saveF1TableFileName_full)
        df_slctd_table.to_csv(saveF1TableFileName_full)
    else:
        print("already saved--", saveF1TableFileName_full)

    print("dataIdent_=", dataIdent_, "\ntestUser_=", testUser_, "\nvalidUser_=", validUser_, "\nhid_state_cnt_vec_=", hid_state_cnt_vec_)
    print(("bestVaID({:" + formatStr + "}),trAcc({:" + formatStr + "}),vaAcc({:" + formatStr + "}),teAcc({:" + formatStr + "})").format(bestVaID,accvectr[bestVaID],accvecva[bestVaID],accvecte[bestVaID]))
    return df_slctd_table