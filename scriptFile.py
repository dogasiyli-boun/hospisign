import helperFuncs as funcH
import projRelatedHelperFuncs as prHF
import modelFuncs as moF
import numpy as np
import os
import pandas as pd
import wget
from torch import manual_seed
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

def download_data(data_ident_vec, dataset_ident_str):
    """
    :param data_ident_vec: "hog", "snv", "skeleton", "list_dict"
    :param dataset_ident_str: 'Dev' or 'Exp'
    :param nested_conf: nested configuration object from conf.yaml
    :return: none

    downloads the extracted features from web
    """
    filename = prHF.get_hospisign_file_name(data_ident_vec, dataset_ident_str)

    target_file_name = os.path.join(funcH._CONF_PARAMS_.DIR.DATA, filename)
    url = "ftp://dogasiyli:Doga.Siyli@dogasiyli.com/hospisign.dogasiyli.com/extractedData/" + filename
    if not os.path.isfile(target_file_name):
        print(target_file_name, " will be downloaded from url = ", url)
        filename = wget.download(url, out=target_file_name)
        print("Download completed  : ", filename)
    else:
        print(target_file_name, " is already downloaded ")

def download_data_script():
    for data_ident_vec in ["hog", "snv", "skeleton", "list_dict"]:
        for dataset_ident_str in ["Dev", "Exp"]:
            download_data(data_ident_vec, dataset_ident_str)

#SUBMIT_CHECKED
def get_hid_state_vec(hidStateID):
    hid_state_cnt_vec = [2048, 1024, 1024, 512, 512, 256, 256]
    if hidStateID == 1:
        hid_state_cnt_vec = [256, 256]
    elif hidStateID == 2:
        hid_state_cnt_vec = [512, 512]
    elif hidStateID == 3:
        hid_state_cnt_vec = [512, 512, 256, 256]
    elif hidStateID == 4:
        hid_state_cnt_vec = [1024, 512, 512, 256]
    elif hidStateID == 5:
        hid_state_cnt_vec = [64, 64, 64, 64]
    elif hidStateID == 6:
        hid_state_cnt_vec = [128, 128, 128, 128]
    elif hidStateID == 7:
        hid_state_cnt_vec = [256, 256, 256, 256]
    elif hidStateID == 8:
        hid_state_cnt_vec = [512, 512, 512, 512]
    elif hidStateID == 9:
        hid_state_cnt_vec = [128, 128]

    return hid_state_cnt_vec

def mlp_hospisign(dropout_value, hidStateID, data_ident_vec, dataset_ident_str ="Dev", pca_dim = 256, verbose = 0, validationUserVec = [2, 3, 4, 5, 6, 7], testUserVec = [2, 3, 4, 5, 6, 7], resultFolder=funcH._CONF_PARAMS_.DIR.CLUSTER, rs_range=1):
    nos = prHF.get_nos(dataset_ident_str)
    for dataIdent in data_ident_vec:
        # prepare and get the data along with labels and necessary variables
        ft, lb, lb_sui, lb_map = prHF.combine_pca_hospisign_data(data_ident=dataIdent, pca_dim=pca_dim, dataset_ident_str=dataset_ident_str, verbose=verbose)

        doStr = ""
        if dropout_value is not None:
            doStr = "_do{:4.2f}".format(dropout_value)

        hid_state_cnt_vec = get_hid_state_vec(hidStateID)
        hidStatesDict = moF.create_hidstate_dict(hid_state_cnt_vec, init_mode_vec=None, act_vec=None)

        for userVa in validationUserVec:
            for userTe in testUserVec:
                if userVa == userTe:
                    continue
                print("*****\nprepare_hospisign_data : \nteUser=", userTe, ", vaUser=", userVa)
                dl_tr, dl_va, dl_te = prHF.prepare_hospisign_data(ft, lb, lb_sui, validUser=userVa, testUser=userTe)
                for rs in range(rs_range):
                    result_file_name = 'di' + dataIdent + '_nos' + str(nos) + '_te' + str(userTe) + '_va' + str(userVa) + '_rs' + str(rs) + '_hs' + str(hidStateID) + doStr + '.npz'
                    result_file_name_full = os.path.join(resultFolder, result_file_name)
                    result_file_exist = os.path.exists(result_file_name_full)
                    best_model_file_name = result_file_name.replace('.npz', '.model')
                    best_model_file_name_full = result_file_name_full.replace('.npz', '.model')
                    model_file_exist = os.path.exists(best_model_file_name_full)
                    if result_file_exist:
                        print("RESULT FILE EXIST +++ teUser=", userTe, ", vaUser=", userVa, ": ", result_file_name)
                    else:
                        print("RESULT FILE DOESNT EXIST +++")
                    if model_file_exist:
                        print("MODEL FILE EXIST +++ teUser=", userTe, ", vaUser=", userVa, ": ", best_model_file_name)
                    else:
                        print("MODEL FILE DOESNT EXIST +++")

                    try:
                        if result_file_exist and model_file_exist:
                            print("**********teUser=", userTe, ", vaUser=", userVa, " -- skipping : ", result_file_name, "-", best_model_file_name)
                            class_names = np.asarray(lb_map["khsName"])
                            df_slctd_table = prHF.get_result_table_out(result_file_name_full, class_names)
                            continue
                    except:
                        pass

                    print("***************\nteUser=", userTe, ", vaUser=", userVa, ", rs=", rs)

                    # check if exist. if yes go on

                    print("teUser=", userTe, ", vaUser=", userVa, ", rs=", rs)
                    np.random.seed(rs)
                    manual_seed(rs)
                    uniqLabs = np.unique(lb)
                    classCount = len(uniqLabs)
                    print("uniqLabs=", uniqLabs, ", classCount_=" + dataIdent, classCount)

                    model_ = moF.MLP_Dict(ft.shape[1], hidStatesDict, classCount, dropout_value=dropout_value)

                    accvectr, accvecva, accvecte, preds_best, labels_best = model_.train_and_evaluate(dl_tr, dl_va, dl_te, epochCnt=30, saveBestModelName=best_model_file_name_full)

                    print("data_ident=", dataIdent)
                    print("nos=", nos)
                    print("userVa=", userVa)
                    print("rs=", rs)
                    print("hid_state_cnt_vec=", hid_state_cnt_vec)

                    bestVaID = np.argmax(accvecva)
                    bestTeID = np.argmax(accvecte)
                    formatStr = "5.3f"
                    print(("bestVaID({:" + formatStr + "}),vaAcc({:" + formatStr + "}),teAcc({:" + formatStr + "})").format(bestVaID, accvecva[bestVaID], accvecte[bestVaID]))
                    print(("bestTeID({:" + formatStr + "}),vaAcc({:" + formatStr + "}),teAcc({:" + formatStr + "})").format(bestTeID, accvecva[bestTeID], accvecte[bestTeID]))
                    print(("last, vaAcc({:" + formatStr + "}),teAcc({:" + formatStr + "})").format(accvecva[-1],accvecte[-1]))

                    np.savez(result_file_name_full, dataIdent_=dataIdent, testUser_=userTe, validUser_=userVa,
                             hid_state_cnt_vec_=hid_state_cnt_vec, accvectr_=accvectr, accvecva_=accvecva,
                             accvecte_=accvecte, preds_best_=preds_best, labels_best_=labels_best, allow_pickle=True)
                    result_files = np.load(result_file_name_full, allow_pickle=True)
                    print(result_files.files, "\n***************")
                print("*****\n")

#SUBMIT_CHECK
def mlp_analyze_result_hgsnsk(userTe, userVa, dropout_value, data_ident, dataset_ident_str, rs=0, hidStateID = 0, pca_dim = 256, verbose = 0, resultFolder = funcH._CONF_PARAMS_.DIR.CLUSTER):
    nos = prHF.get_nos(dataset_ident_str)
    ft, lb, lb_sui, lb_map = prHF.combine_pca_hospisign_data(data_ident=data_ident, pca_dim=pca_dim, nos=nos,
                                                             verbose=verbose)
    class_names = np.asarray(lb_map["khsName"])
    classCount = len(class_names)

    doStr = ""
    if dropout_value is not None:
        doStr = "_do{:4.2f}".format(dropout_value)

    result_file_name = 'di' + data_ident + '_nos' + str(nos) + '_te' + str(userTe) + '_va' + str(userVa) + '_rs' + str(rs) + '_hs' + str(hidStateID) + doStr + '.npz'
    result_file_name_full = os.path.join(resultFolder, result_file_name)
    result_file_exist = os.path.exists(result_file_name_full)
    best_model_file_name_full = result_file_name_full.replace('.npz', '.model')
    model_file_exist = os.path.exists(best_model_file_name_full)
    if result_file_exist:
        print("RESULT FILE EXIST +++ teUser=", userTe, ", vaUser=", userVa, ": ", result_file_name)
    else:
        print("RESULT FILE DOESNT EXIST +++", result_file_name)
    if model_file_exist:
        print("MODEL FILE EXIST +++ teUser=", userTe, ", vaUser=", userVa, ": ", best_model_file_name_full)
    else:
        print("MODEL FILE DOESNT EXIST +++", best_model_file_name_full)

    df_slctd_table = []
    if result_file_exist:
        df_slctd_table = prHF.get_result_table_out(result_file_name_full, class_names)

    model_exports = []
    if model_file_exist:
        hid_state_cnt_vec = get_hid_state_vec(hidStateID)
        hidStatesDict = moF.create_hidstate_dict(hid_state_cnt_vec, init_mode_vec=None, act_vec=None)
        model_ = moF.MLP_Dict(ft.shape[1], hidStatesDict, classCount)
        print("loading best model--", best_model_file_name_full)
        model_.load_model(best_model_file_name_full)

        dl_tr, dl_va, dl_te = prHF.prepare_hospisign_data(ft, lb, lb_sui, validUser=userVa, testUser=userTe)
        print("exporting train features...")
        acc_tr, preds_tr, labs_tr, final_layer_tr = model_.export_final_layer(dl_tr)
        print("exporting validation features...")
        acc_va, preds_va, labs_va, final_layer_va = model_.export_final_layer(dl_va)
        print("exporting test features...")
        acc_te, preds_te, labs_te, final_layer_te = model_.export_final_layer(dl_te)
        model_exports = {
            "tr": {"acc": acc_tr, "preds": preds_tr, "labs": labs_tr, "final_layer": final_layer_tr},
            "va": {"acc": acc_va, "preds": preds_va, "labs": labs_va, "final_layer": final_layer_va},
            "te": {"acc": acc_te, "preds": preds_te, "labs": labs_te, "final_layer": final_layer_te},
        }
    return df_slctd_table, model_exports

def get_from_model(model_exports_x, model_str, normalizationMode, data_va_te_str, verbose=0):
    ft = model_exports_x[data_va_te_str]["final_layer"]
    pr = model_exports_x[data_va_te_str]["preds"]
    la = model_exports_x[data_va_te_str]["labs"]
    if verbose > 0:
        print(ft.shape)
        print(pr.shape)
        print(la.shape)
        print(model_str, ", ", data_va_te_str, " acc = ", "{:5.3f}".format(accuracy_score(la, pr)), ", norm_mode=", normalizationMode)
    if normalizationMode == 'max':
        ft_n = funcH.normalize2(ft, norm_mode='nm', axis=1)
    elif normalizationMode == 'sum':
        ft_n = funcH.normalize2(ft, norm_mode='ns', axis=1)
    elif normalizationMode == 'softmax':
        ft_n = funcH.softmax(ft.T).T
    else:
        ft_n = ft
    return ft_n, pr, la

#SUBMIT_CHECK
def check_model_exports(model_exports_x, model_export_string):
    print("model_exports_", model_export_string)
    print(model_exports_x["tr"]["acc"], model_exports_x["va"]["acc"], model_exports_x["te"]["acc"])
    print("tr-", model_exports_x["tr"]["final_layer"].shape)
    print("va-", model_exports_x["va"]["final_layer"].shape)
    print("te-", model_exports_x["te"]["final_layer"].shape)

def mlp_study_score_fuse_apply(model_export_dict, defStr, data_va_te_str, data_ident_vec=["hog", "sn", "sk", "hgsk", "hgsn", "snsk", "hgsnsk"]):
    ft_comb_ave = None
    ft_comb_max = None
    ft_comb_sof = None

    df_final = pd.DataFrame({"khsName": model_export_dict["hog"]["df_slctd_table"]["khsName"].sort_index()})
    acc_vec_all = {}
    for data_ident in data_ident_vec:
        #print("****\n", defStr, "\n", data_ident)
        ft_max, preds_te, labels_xx = get_from_model(model_export_dict[data_ident]["model_export"], model_str=data_ident, normalizationMode="max", data_va_te_str=data_va_te_str)
        ft_sof, _, _ = get_from_model(model_export_dict[data_ident]["model_export"], model_str=data_ident, normalizationMode="softmax", data_va_te_str=data_va_te_str)
        ft_none, _, _ = get_from_model(model_export_dict[data_ident]["model_export"], model_str=data_ident, normalizationMode=None, data_va_te_str=data_va_te_str)

        print(data_ident+data_va_te_str+"_acc = ", "{:5.3f}".format(accuracy_score(labels_xx, preds_te)))

        df_final = pd.concat([df_final, model_export_dict[data_ident]["df_slctd_table"]["F1_Score"].sort_index()], axis = 1).rename(columns={"F1_Score": data_ident})
        ft_comb_ave = ft_max if ft_comb_ave is None else ft_max+ft_comb_ave
        ft_comb_max = ft_none if ft_comb_max is None else np.maximum(ft_none, ft_comb_max)
        ft_comb_sof = ft_sof if ft_comb_sof is None else ft_sof+ft_comb_sof

        preds_max = np.argmax(ft_max, axis=1)
        preds_sof = np.argmax(ft_sof, axis=1)

        acc_max = accuracy_score(labels_xx, preds_max)
        acc_sof = accuracy_score(labels_xx, preds_sof)
        if acc_max-acc_sof != 0.0:
            print(defStr, data_ident, ", max norm acc = ", "{:5.3f}".format(acc_max))
            print(defStr, data_ident, ", softmax norm acc = ", "{:5.3f}".format(acc_sof))
        #print("****")
        acc_vec_all[data_ident] = acc_max

    classNames = np.asarray(model_export_dict["hog"]["df_slctd_table"]["khsName"].sort_index())

    pr_comb_ave = np.argmax(ft_comb_ave, axis=1)
    pr_comb_max = np.argmax(ft_comb_max, axis=1)
    pr_comb_sof = np.argmax(ft_comb_sof, axis=1)
    acc_ave = accuracy_score(labels_xx, pr_comb_ave)
    acc_max = accuracy_score(labels_xx, pr_comb_max)
    acc_sof = accuracy_score(labels_xx, pr_comb_sof)
    print(defStr, data_va_te_str, "comb-(AVE) acc = {:6.4f}".format(acc_ave))
    print(defStr, data_va_te_str, "comb-(MAX) acc = {:6.4f}".format(acc_max))
    print(defStr, data_va_te_str, "comb-(SOFTMAX) acc = {:6.4f}".format(acc_sof))

    str_j = '_'.join(data_ident_vec)
    acc_vec_all[str_j+'_ave'] = acc_ave
    acc_vec_all[str_j+'_max'] = acc_max
    acc_vec_all[str_j+'_sof'] = acc_sof

    conf_mat_ave = confusion_matrix(labels_xx, pr_comb_ave)
    conf_mat_max = confusion_matrix(labels_xx, pr_comb_max)
    conf_mat_sof = confusion_matrix(labels_xx, pr_comb_sof)

    while len(classNames) < conf_mat_max.shape[0] or len(classNames) < conf_mat_sof.shape[0]:
        classNames = np.hstack((classNames, "wtf"))

    cmStats_ave, df_ave = funcH.calcConfusionStatistics(conf_mat_ave, categoryNames=classNames, selectedCategories=None, verbose=0)
    cmStats_max, df_max = funcH.calcConfusionStatistics(conf_mat_max, categoryNames=classNames, selectedCategories=None, verbose=0)
    cmStats_sofx, df_sof = funcH.calcConfusionStatistics(conf_mat_sof, categoryNames=classNames, selectedCategories=None, verbose=0)
    df_final = pd.concat([df_final, df_ave["F1_Score"].sort_index()], axis=1).rename(columns={"F1_Score": "df_ave"})
    df_final = pd.concat([df_final, df_max["F1_Score"].sort_index()], axis=1).rename(columns={"F1_Score": "df_max"})
    df_final = pd.concat([df_final, df_sof["F1_Score"].sort_index()], axis=1).rename(columns={"F1_Score": "df_sof"})

    df_final.to_csv(os.path.join(funcH.getVariableByComputerName('desktop_dir'), "comb", "comb_" + defStr + data_va_te_str + "_all.csv"))

    print(df_final)
    print(acc_vec_all)

    results_dict = {
        "conf_mat_max": conf_mat_max,
        "conf_mat_sof": conf_mat_sof,
        "df_final": df_final,
        "df_ave": df_ave,
        "df_max": df_max,
        "df_sof": df_sof,
        "acc_vec_all": acc_vec_all,
    }
    return results_dict

#SUBMIT_CHECK
def append_to_all_results(results_dict, index_name, dropout_value, rs, hidStateID, dataset_ident_str, data_va_te_str,
                          model_export_dict_folder=funcH._CONF_PARAMS_.DIR.MODEL_EXPORTS):
    nos = prHF.get_nos(dataset_ident_str)
    columns = ["hog", "sk", "sn", "hgsk", "hgsn", "snsk", "hgsnsk", "hogsk", "hogskhgsk", "hogsn", "hogsnsk", "ALLax", "ALLmx", "ALLsm"]
    doStr = ""
    if dropout_value is not None:
        doStr = "_do{:4.2f}".format(dropout_value)
    all_results_filename = 'nos' + str(nos) + '_rs' + str(rs) + '_hs' + str(hidStateID) + doStr + data_va_te_str + '.npy'
    all_results_filename = os.path.join(model_export_dict_folder, all_results_filename)

    if os.path.exists(all_results_filename):
        print("loading...", all_results_filename)
        all_results = pd.read_pickle(all_results_filename)
        print(all_results_filename, " loaded : \n", all_results)
    else:
        all_results = pd.DataFrame(index=None, columns=columns)
        all_results.to_pickle(all_results_filename)
        print(all_results_filename, " saved as empty list")

    if index_name not in all_results.index:
        a = pd.DataFrame(np.nan, index=[index_name], columns=columns)
        all_results = all_results.append(a)
        all_results.to_pickle(all_results_filename)
        print(all_results_filename, " saved adding index_name = ", index_name)

    keys_added = list()
    for key, value in results_dict["acc_vec_all"].items():
        # keyN = keyN.replace('hog','hg')
        keyN = key.replace('_', '')
        keyN = keyN.replace('hogsnskhgskhgsnsnskhgsnsk', 'ALL')
        keyN = keyN.replace('ALLave', 'ALLax')
        keyN = keyN.replace('ALLmax', 'ALLmx')
        keyN = keyN.replace('ALLsof', 'ALLsm')
        keyN = keyN.replace('max', '')
        keyN = keyN.replace('ave', '')
        try:
            if all_results[keyN][index_name] != value:
                all_results[keyN][index_name] = value
                print("added", keyN, value)
                all_results.loc[index_name, keyN] = value
                print(all_results[keyN][index_name])
                keys_added.append(keyN)
            else:
                print("(same)skipped", key, keyN, value)
        except:
            print("(error)skipped", key, keyN, value)
    if len(keys_added) > 0:
        print(all_results_filename, " updated by index_name = ", index_name, ", keys_added=", keys_added)
        all_results.to_pickle(all_results_filename)
        print("updated : \n", all_results)
    return all_results

#SUBMIT_CHECK
def mlp_study_score_fuse(userTe, userVa, dropout_value, rs, hidStateID, dataset_ident_str, data_ident_vec=["hog", "sn", "sk", "hgsk", "hgsn", "snsk", "hgsnsk"]):
    nos = prHF.get_nos(dataset_ident_str)
    model_export_dict = {}
    for data_ident in data_ident_vec:
        df_slctd, model_exports = mlp_analyze_result_hgsnsk(userTe=userTe, userVa=userVa, dropout_value=dropout_value,
                                                            data_ident=data_ident, dataset_ident_str=dataset_ident_str, rs=rs, hidStateID=hidStateID)
        dict_cur = {"df_slctd_table": df_slctd, "model_export": model_exports}
        model_export_dict[data_ident] = dict_cur
    for data_ident in data_ident_vec:
        check_model_exports(model_export_dict[data_ident]["model_export"], data_ident)
    return model_export_dict

# SUBMIT_CHECK
def save_df_tables_for_te_va(userTe, userVa, dropout_value, hidStateID, dataset_ident_str,
                             model_export_dict_folder=funcH._CONF_PARAMS_.DIR.MODEL_EXPORTS):
    nos = prHF.get_nos(dataset_ident_str)
    defStr = "te" + str(userTe) + "_va" + str(userVa) + "_nos" + str(nos)
    model_export_dict_fname = os.path.join(model_export_dict_folder, defStr + "_mex.npy")
    if os.path.exists(model_export_dict_fname):
        print("loading model_export_dict from :", model_export_dict_fname)
        model_export_dict = np.load(model_export_dict_fname, allow_pickle=True).item()
        print("loaded model_export_dict from :", model_export_dict_fname)
    else:
        print("creating model_export_dict for :", defStr)
        model_export_dict = mlp_study_score_fuse(userTe=userTe, userVa=userVa, dropout_value=dropout_value, rs=0,
                                                 hidStateID=hidStateID, dataset_ident_str=dataset_ident_str,
                                                 data_ident_vec=["hog", "sn", "sk", "hgsk", "hgsn", "snsk", "hgsnsk"])
        np.save(model_export_dict_fname, model_export_dict, allow_pickle=True)
        print("saving model_export_dict at :", model_export_dict_fname)
    return model_export_dict, defStr

#SUBMIT_CHECK
def append_to_all_results_dv_loop(userTe, userVa, dataset_ident_str, dv, data_va_te_str, dropout_value, rs, hidStateID):
    nos = prHF.get_nos(dataset_ident_str)
    model_export_dict, defStr = save_df_tables_for_te_va(userTe=userTe, userVa=userVa, dropout_value=dropout_value, hidStateID=hidStateID, dataset_ident_str=dataset_ident_str)
    index_name = "te" + str(userTe) + "_va" + str(userVa) + "_nos" + str(nos)
    for data_ident_vec in dv:
        str_j = '_'.join(data_ident_vec)
        defStr2 = defStr + "_" + str_j
        print("defStr=", defStr)
        results_dict = mlp_study_score_fuse_apply(model_export_dict, defStr2, data_va_te_str=data_va_te_str, data_ident_vec=data_ident_vec)
        all_results = append_to_all_results(results_dict, index_name, dropout_value=dropout_value, rs=rs,
                                            hidStateID=hidStateID, dataset_ident_str=dataset_ident_str, data_va_te_str=data_va_te_str)
    return all_results

#SUBMIT_CHECKED
def print_cluster_results_hospisign(dataset_ident_str='Dev'): #  'Dev' or 'Exp'
    """
    Prints out the base clustering results
    According to the saved 'npz' files with cluster results
    :param dataset_ident_str:
    :return: none
    """
    nos = prHF.get_nos(dataset_ident_str)
    hospisign_labels = prHF.get_hospisign_labels(dataset_ident_str=dataset_ident_str, sort_by=None)
    labelNames = list(np.squeeze(hospisign_labels["khsNames"]))
    results_dir = funcH._CONF_PARAMS_.DIR.RESULTS  # '/media/dg/SSD_Data/DataPath/bdResults'
    baseLineResultFolder = os.path.join(results_dir, 'baseResults')  # '/media/dg/SSD_Data/DataPath/bdResults/baseResults'
    baseResFiles = funcH.getFileList(baseLineResultFolder, startString="", endString=".npz", sortList=False)
    for f in baseResFiles:
        if not str(f).__contains__("_" + str(nos)):
            continue
        if not str(f).__contains__("KMeans"):
            continue
        labels_true, labels_pred = prHF.loadBaseResult(f)

        _confMat_preds, kluster2Classes, kr_pdf, weightedPurity, cnmxh_perc = funcH.countPredictionsForConfusionMat(labels_true, labels_pred, labelNames=labelNames)
        acc = 100 * np.sum(np.diag(_confMat_preds)) / np.sum(np.sum(_confMat_preds))
        meanPurity = np.mean(np.asarray(kr_pdf["%purity"]))

        f = f.replace("_" + str(nos), "")
        f = f.replace("_KMeans", "")
        print("f({:s}),n({:d},{:d}), acc({:5.3f}), meanPurity({:5.3f}), weightedPurity({:5.3f})".format(f, labels_true.shape[0], labels_pred.shape[0], acc, meanPurity, weightedPurity))
        print("****")

#SUBMIT_CHECKED
def cluster_hospisign(randomSeed,  #  some integer for reproducibility
                      dataset_ident_str='Dev',  #  'Dev' or 'Exp'
                      data_ident_vec=["hog", "sn", "sk", "hgsk", "hgsn", "snsk", "hgsnsk"],
                      clustCntVec=[256],  #  96 for skeleton(sk), 256 for others
                      verbose=0,
                      pca_dim=256):
    nos = prHF.get_nos(dataset_ident_str)
    for data_ident in data_ident_vec:
        pca_dim_2_use = pca_dim
        if (data_ident == 'skeleton' or data_ident == 'sk') and pca_dim > 112:
            pca_dim_2_use = 96
        ft, lb, lb_sui, lb_map = prHF.combine_pca_hospisign_data(data_ident=data_ident, dataset_ident_str=dataset_ident_str, pca_dim=pca_dim_2_use, verbose=verbose)
        prHF.run_clustering_hospisign(ft, lb, lb_map, dataToUse=data_ident, numOfSigns=nos,
                                      pcaCount=pca_dim_2_use, clustCntVec=clustCntVec, randomSeed=randomSeed)
    prHF.traverse_base_results_folder_hospising()
    return

#SUBMIT_CHECKED
def combine_clusters_hospisign(dataset_ident_str,
                               dataToUseVec = ["hog", "sn", "sk"],
                               clustCntVec=[256],
                               consensus_clustering_max_k=256):
    nos = prHF.get_nos(dataset_ident_str)
    #bad coding here :)
    # SUBMIT_CHANGE - load khs names differently
    ft, lb, lb_sui, lb_map = prHF.combine_pca_hospisign_data(data_ident="sk", nos=nos, pca_dim=None, verbose=0)
    del(ft, lb, lb_sui)
    class_names = np.asarray(lb_map["khsName"])
    del (lb_map)

    resultsToCombineDescriptorStr = '|'.join(dataToUseVec)
    labelNames, labels, predictionsDict, cluster_runs, N = prHF.load_labels_pred_for_ensemble_hospising(
                    class_names, nos=nos, clustCntVec=clustCntVec, dataToUseVec=dataToUseVec)
    prHF.ensemble_cluster_analysis(cluster_runs, predictionsDict, labels,
                                   consensus_clustering_max_k=consensus_clustering_max_k, useNZ=False, nos=nos,
                                   resultsToCombineDescriptorStr=resultsToCombineDescriptorStr,
                                   labelNames=labelNames, verbose=True)