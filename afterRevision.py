from matplotlib import pyplot as plt
import numpy as np
from pandas import DataFrame as df
import pandas as pd

import os
import sys

from scipy.spatial import distance

from sklearn.metrics.cluster import normalized_mutual_info_score as nmi
from sklearn.metrics.cluster import mutual_info_score as mi
from sklearn.preprocessing import normalize

import projRelatedHelperFuncs as prHF
import scriptFile as sf
import helperFuncs as funcH


def get_from_params(param_name, default="exit", **params):
    if type(default) == str and default == "exit":
        return params[param_name] if param_name in params else sys.exit("{} must be in params".format(param_name))
    else:
        return params[param_name] if param_name in params else default
    return None


def row_normalize(X):
    row_sums = X.sum(axis=1)
    return X / row_sums[:, np.newaxis]


def test_row_normalize():
    zz = np.arange(0, 27, 3).reshape(3, 3).astype(float)
    zz[1, 1] = np.nan
    zz[np.isnan(zz)] = 0.0
    # display(df(zz))
    # display(df(normalize(zz, axis=1, norm='l1')))
    # display(df(row_normalize(zz)))
    # display(df(normalize(zz, axis=1, norm='l2')))


def dist_2_centroid(a, feats, centroidIDs, i, j, verbose=0):
    x = feats[centroidIDs[i - 1], :]
    y = feats[centroidIDs[j - 1], :]
    d = np.linalg.norm(x - y)
    kx = a.iloc[centroidIDs[i - 1], 4]
    ky = a.iloc[centroidIDs[j - 1], 4]
    if verbose >= 1:
        print("calculating distance of ", kx, ky)
    xl = [k for k in range(feats.shape[0]) if a.iloc[k, 4] == kx]
    yl = [k for k in range(feats.shape[0]) if a.iloc[k, 4] == ky]
    if verbose > 1:
        print("----xl({}),yl({})".format(np.shape(xl), np.shape(yl)))
    xM = feats[xl, :]
    yM = feats[yl, :]
    if verbose > 1:
        print("----xM({}),yM({})".format(np.shape(xM), np.shape(yM)))
    dst = distance.cdist(xM, yM, 'euclidean')
    if verbose > 1:
        print("----dst.shape({})".format(dst.shape))
    dMax = np.amax(dst)
    dMin = np.amin(dst)
    dMea = np.mean(dst)
    if verbose > 1:
        print("----dMin{},d{},dMea{},dMax{}".format(dMin, d, dMea, dMax))
    return dMin, d, dMea, dMax


def fill_with_dists(a, feats, centroidIDs, X = None, fill_mode="right_up_diag", verbose=1):
    if X is None:
        X = np.zeros((len(centroidIDs), len(centroidIDs), 3), dtype=float)
        X[:, :, :] = np.NAN
    if fill_mode == "right_up_diag":
        for ci in range(0, len(centroidIDs)):
            for cj in range(ci + 1, len(centroidIDs)):
                dMin, d, dMea, dMax = dist_2_centroid(a, feats, centroidIDs, ci + 1, cj + 1, verbose=verbose - 1)
                X[ci, cj, :] = (dMin, dMea, d)
                if verbose > 0:
                    print(ci, cj, len(centroidIDs), " done..")
    elif fill_mode == "left_down_diag_nan":
        for ci in range(0, len(centroidIDs)):
            for cj in range(ci, len(centroidIDs)):
                if X.ndim == 3:
                    X[cj, ci, :] = np.nan
                else:
                    X[cj, ci] = np.nan
    elif fill_mode == "left_down_diag_zero":
        for ci in range(0, len(centroidIDs)):
            for cj in range(ci + 1, len(centroidIDs)):
                if X.ndim == 3:
                    X[cj, ci, :] = (0.0, 0.0, 0.0)
                else:
                    X[cj, ci] = 0.0
    elif fill_mode == "full_except_diag":
        for ci in range(0, len(centroidIDs)):
            for cj in range(ci + 1, len(centroidIDs)):
                dMin, d, dMea, dMax = dist_2_centroid(a, feats, centroidIDs, ci + 1, cj + 1, verbose=verbose - 1)
                X[ci, cj, :] = (dMin, dMea, d)
                X[cj, ci, :] = (dMin, dMea, d)
                if verbose > 0:
                    print(ci, cj, len(centroidIDs), " done..")
    return X


def initialize_procedure(**params):
    use_dist_as = get_from_params("use_dist_as", default=1, **params)  # 0-min, 1-mean, 2-d
    cluster_count_use = get_from_params("cluster_count_use", default=256, **params)  # 256
    data_ident = get_from_params("data_ident", default="hgsk", **params)  # "hgsk"
    dataset_ident_str = get_from_params("dataset_ident_str", default="Dev", **params)  # "Exp"
    pca_dim = get_from_params("pca_dim", default=256, **params)  # 256

    featuse = "{}{}".format(data_ident, pca_dim)
    numOfSigns = 11 if dataset_ident_str == "Dev" else 41

    if use_dist_as == 0:
        diststr = "min"
    elif use_dist_as == 1:
        diststr = "mean"
    else:
        diststr = "dist"

    clustID = 1 if cluster_count_use == 32 else 2

    dist_folder = "/home/doga/GithUBuntU/hospisign/dists_for_web"

    print("featuse = ", featuse)
    print("dataset_ident_str = ", dataset_ident_str)
    print("cluster_count_use({}) --> clustID({})".format(cluster_count_use, clustID))
    print("use_dist_as({}) --> diststr({})".format(use_dist_as, diststr))
    print("dist_folder = ", dist_folder)

    use_params = {
        "use_dist_as": use_dist_as,
        "cluster_count_use": cluster_count_use,
        "data_ident": data_ident,
        "dataset_ident_str": dataset_ident_str,
        "numOfSigns": numOfSigns,
        "pca_dim": pca_dim,
        "featuse": featuse,
        "diststr": diststr,
        "dist_folder": dist_folder,
        "clustID": clustID,
    }

    return use_params


def concat_and_check_explainibility(concat_method):
    index_vec = ["hg", "sk", "sn", "hgsk", "hgsn", "snsk", "hgsnsk"]
    cols_vec = ["dim", "pcaDim96", "pcaDim256", "pcaDim512"]
    row_cnt = len(index_vec)
    col_cnt = len(cols_vec)

    info_loss = np.empty((row_cnt, col_cnt), dtype=object)
    info_loss[:] = np.nan

    info_table = df(info_loss, columns=cols_vec, index=index_vec)
    print(info_table)

    acc_table = df(info_loss, columns=cols_vec, index=index_vec)
    print(acc_table)

    dataset_ident_str = "Dev"
    numOfSigns = 11 if dataset_ident_str == "Dev" else 41
    clustCntVec = [256]
    for (c, dim) in enumerate([96, 256, 512]):
        for (i, data_ident) in enumerate(index_vec):
            col_name = 'pcaDim{}'.format(dim)
            if dim == 96:
                print(i, data_ident, c, col_name)
            if len(data_ident) < 3 and not data_ident == "sk" and dim == 96:
                print("skip ", data_ident, dim)
                continue
            if data_ident == "sk" and dim > 96:
                print("skip ", data_ident, dim)
                continue
            feats, labels, labels_sui, label_map, exp_vec_dict, exp_val_dict = prHF.combine_pca_hospisign_data(
                data_ident,
                pca_dim=dim,
                dataset_ident_str=dataset_ident_str,
                verbose=1,
                concat_method=concat_method)
            print("exp_val_dict:\n", exp_val_dict)
            info_table.iloc[i, c + 1] = exp_val_dict[data_ident]
            resultDict = prHF.run_clustering_hospisign(ft=feats, labels_all=labels, lb_map=label_map,
                                                       dataToUse=data_ident, numOfSigns=numOfSigns, pcaCount=dim,
                                                       clustCntVec=clustCntVec, clusterModels=['KMeans'], randomSeed=5,
                                                       enforce_rerun=True)
            print(np.shape(resultDict))
            acc_table.iloc[i, c + 1] = np.squeeze(resultDict[-1:])[3][1]
    print(info_table)
    print(acc_table)
    return acc_table, info_table


# this can be deleted
def run_cluster_test(**params):
    data_ident = get_from_params("data_ident", default="hgsk", **params)  # "hgsk, hgsksn"
    pca_dim = get_from_params("pca_dim", default=256, **params)  # 96, 256, 512
    concat_method = get_from_params("concat_method", default=2, **params)  # 1 or 2
    clusterCount = get_from_params("clusterCount", default=256, **params)  # 32, 64, 128, 256, 512
    enforce_rerun = get_from_params("enforce_rerun", default=True, **params)  # 32, 64, 128, 256, 512

    ft, labels, labels_sui, label_map, exp_vec_dict, exp_val_dict = prHF.combine_pca_hospisign_data(data_ident,
                                                                                                    pca_dim=pca_dim,
                                                                                                    dataset_ident_str="Dev",
                                                                                                    verbose=1,
                                                                                                    concat_method=concat_method)
    resultDict = prHF.run_clustering_hospisign(ft=ft, labels_all=labels, lb_map=label_map, dataToUse=data_ident,
                                               numOfSigns=11, pcaCount=pca_dim, clustCntVec=[clusterCount],
                                               clusterModels=['KMeans'], randomSeed=5, enforce_rerun=enforce_rerun)

    # ft_pca_hgsk, exp_var_rat_hgsk = funcH.applyMatTransform(np.concatenate([ft["hg"].T, ft["sk"].T]).T, applyPca=True,
    #                                                         normMode="", verbose=1)
    # plt.clf()
    # plt.plot(exp_var_rat_hgsk)
    # plt.show()
    # print(exp_var_rat_hgsk[255])
    #
    # ft_pca_hgsk, exp_var_rat_hgsk = funcH.applyMatTransform(np.concatenate([ft["hg"].T, ft["sk"].T]).T, applyPca=True,
    #                                                         normMode="", verbose=1)
    # plt.clf()
    # plt.plot(exp_var_rat_hgsk)
    # plt.show()
    # print(exp_var_rat_hgsk[255])

    return resultDict


def extract_centroid_info(centroid_info_pdf, labels, label_map, verbose=1):
    centroidIDs = centroid_info_pdf[["sampleID"]].values.squeeze()
    centroidLabels = np.asarray(labels.squeeze().copy(), dtype=int)[centroidIDs]
    centroidKHSNames = (np.asarray(label_map))[centroidLabels, 1]
    if verbose > 0:
        print(centroidIDs)
        print(centroidLabels)
        print(centroidKHSNames)
        print(centroid_info_pdf)

    # resultName = resultDict[2][0]
    # resultClusterMethod = resultDict[2][1]
    # resultClusterCount = resultDict[2][2]
    # resultClusterCentroids = resultDict[2][3][0]
    # resultDict[2][3]
    # labels_sui[centroidIDs, :]

    return centroidIDs, centroidLabels, centroidKHSNames


def get_pandas_labels(centroidIDs, verbose=0, **params):
    pandas_dict_labels = pd.read_csv(
        "/home/doga/DataFolder/hs_data/list_dict_{}Set.txt".format(params["dataset_ident_str"]), delimiter="*",
        header=None,
        names=["sign", "user", "rep", "frameID", "khsID", "khsName", "hand"])
    if verbose > 0:
        print(pandas_dict_labels)
    if verbose > 1:
        print(pandas_dict_labels.iloc[centroidIDs, [0, 1, 2, 3, 6]])

    fname = "cluster_centroids_{}_c{}".format(params["featuse"], params["cluster_count_use"])
    fnamefull = os.path.join(params["dist_folder"], fname)
    pandas_dict_labels.iloc[centroidIDs, [0, 1, 2, 3, 6]].to_csv(fnamefull, index=False)
    print("pandas_dict saved as {}.csv".format(fnamefull))

    if verbose > 0:
        for index, row in pandas_dict_labels.iloc[centroidIDs, :].iterrows():
            handID = 1 if row['hand'] == ' L' else (2 if row['hand'] == ' R' else 3)
            print(row['sign'], row['user'], row['rep'], row['frameID'], handID)

    if verbose > 0:
        print(pandas_dict_labels.iloc[centroidIDs, :])

    return pandas_dict_labels


def get_dist_matrix(pandas_dict_labels, feats, centroidIDs, **use_params):
    ### clustID = 1 if cluster_count_use==32 else 2
    X_all_filename = "dist_clust{:02d}.npy".format(use_params["clustID"])
    X_all_save_file_name = os.path.join(use_params["dist_folder"], X_all_filename)
    create_X_all = not os.path.exists(X_all_save_file_name)

    if create_X_all:
        print("{} is being created".format(X_all_save_file_name))
        X_all = fill_with_dists(pandas_dict_labels, feats, centroidIDs, X=None, fill_mode="full_except_diag", verbose=1)
        print("{} is being saved".format(X_all_save_file_name))
        np.save(X_all_save_file_name, X_all)
        print("{} is saved".format(X_all_save_file_name))
    else:
        print("{} is load".format(X_all_save_file_name), end='')
        X_all = np.load(X_all_save_file_name)
        print("ed")

    return X_all


def change_dist_to_readable_csv(X_all, pandas_dict_labels, feats, centroidIDs, **params):
    Xd = np.squeeze(X_all[:, :, params["use_dist_as"]])
    print("<{}> Xd({}) extracted from X_all({})".format(params["diststr"], np.shape(Xd), np.shape(X_all)))

    Xd_nan_map = np.isnan(Xd)
    Xd[Xd_nan_map] = 0.0
    Xd = normalize(Xd, axis=1, norm='l2')
    Xd = fill_with_dists(pandas_dict_labels, feats, centroidIDs, X=Xd, fill_mode="left_down_diag_zero")
    Xd = Xd / np.nanmax((np.nanmax(Xd)))  # max normalize the values so that max dist is 1
    print("mx=", np.nanmax((np.nanmax(Xd))))
    Xd = fill_with_dists(pandas_dict_labels, feats, centroidIDs, X=Xd, fill_mode="left_down_diag_nan")

    csv_file_name = "readable_distance_{}_{}.csv".format(params["cluster_count_use"], params["diststr"])
    fnameall = os.path.join(params["dist_folder"], csv_file_name)
    print(fnameall, " is being save", end='')
    df(Xd).to_csv(fnameall)
    print('d')

    return Xd


def change_dist_to_similarity(Xd, centroidIDs):
    Xs1 = 1 - Xd  # so that max dist becomes 0 as in similarity - 0 means no similarity at all
    Xs2 = Xs1 / np.nanmax((np.nanmax(Xs1)))  # max normalize the values so that max similarity is 1
    Xs3 = np.nan_to_num(Xs2)
    Xs4 = np.asarray(99 * Xs3, dtype=int)  # so that max similarity is 99
    Xs = Xs4
    str_for_db = ""
    for ci in range(0, len(centroidIDs)):
        for cj in range(ci + 1, len(centroidIDs)):
            str_for_db += "{};".format(Xs[ci, cj])

    return Xs, str_for_db


def write_string_similarity_for_web(str_for_db, **params):
    distance_text_filename = "dist_clust{:02d}_{}.txt".format(params["clustID"], params["diststr"])
    fnameall = os.path.join(params["dist_folder"], distance_text_filename)
    print("writing distances of {} chars to {} as integers ';' seperated".format(len(str_for_db), fnameall))
    with open(fnameall, "w") as text_file:
        text_file.write("%s" % str_for_db)


def write_readable_distance_csv(Xs, **params):
    csv_file_name_normalized = "readable_distance_{}_{}.csv".format(params["cluster_count_use"], params["diststr"])
    fnameall = os.path.join(params["dist_folder"], csv_file_name_normalized)
    print("writing normalized distances to {} as csv".format(fnameall))
    df(Xs).to_csv(fnameall)


def get_annotation_labels_from_str(web_str, n, verbose=0):
    ann = np.zeros(n, dtype=int)
    web_cells = web_str.split(";")
    for i in web_cells:
        grp, ids = i.split('-')
        ids = np.asarray(ids.split(','),dtype=int)
        ann[ids-1] = grp
        if verbose:
            print("group({}) : inds({})".format(grp, ids-1))
    if verbose:
        print("annotation list as str:", end='')
        print(",".join([str(i) for i in (ann - 1).tolist()]))

    return ann


def get_annotations_vs_khsNames(annotation_vec, centroidLabels, label_map):
    centroidKHSNames = (np.asarray(label_map))[centroidLabels, 1]
    return df([annotation_vec, centroidKHSNames]).transpose(), centroidKHSNames


def convert_string_annotations_to_int(X):
    # create a test array
    uniq_X, uniq_inds = np.unique(X, return_index=True)
    int_vec = np.zeros(len(X), dtype=int)
    i = 0
    for x in uniq_X:
        inds_list = (X==x)
        res = list(filter(lambda i: inds_list[i], range(len(inds_list))))
        int_vec[res] = i
        i = i + 1
    return int_vec, uniq_X


def test_nmi_func(ann_vec, gt_vec, verbose=0):
    cl_l = [0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 0, 0, 1, 2, 2, 2, 2, 2, 2, 2]
    kl_a = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    if verbose > 3:
        print("nmi(ann_test, gt_test):", nmi(cl_l, kl_a))
        print("nmi(gt_test, ann_test):", nmi(kl_a, cl_l))
    if verbose > 0:
        print("nmi(ann_vec, gt_vec):", nmi(ann_vec, gt_vec))
        print("nmi(gt_vec, ann_vec):", nmi(gt_vec, ann_vec))
    str_ann_vec = ",".join([str(i) for i in (ann_vec - 1).tolist()])
    str_gt_vec = ",".join([str(i) for i in (gt_vec - 1).tolist()])
    if verbose > 1:
        print("str_ann_vec=", str_ann_vec)
        print("str_gt_vec=", str_gt_vec)


def calc_entropy_of_label(all_labels, the_label, verbose=2, ck='class', cond=''):
    n = len(all_labels)
    j = 0
    for i in range(n):
        if all_labels[i] == the_label:
            j += 1
    frac = j / n
    # print(the_label,j,n,frac)
    ent = -frac * np.log2(frac)
    if verbose > 1:
        print("{}entropy of {} {} = {}".format(cond, ck, the_label, ent, verbose=1))
    return ent


def calc_entropy_of_dist(class_labs, verbose=2, ck='class', cond=''):
    cllb_uniq = np.unique(class_labs)
    cllb_cnt = len(cllb_uniq)
    cllb_ent = np.zeros(cllb_cnt, dtype=float)
    clent = 0
    for ci in range(cllb_cnt):
        the_label = cllb_uniq[ci]
        cllb_ent[ci] = calc_entropy_of_label(class_labs, the_label, verbose=verbose, ck=ck, cond=cond)
        clent += cllb_ent[ci]
    return clent, cllb_uniq, cllb_ent, cllb_cnt


def grab_dist_from_dist(dist_source, pick_dist_elm, other_dist, verbose=0):
    other_dist_picks = []
    for i in range(len(dist_source)):
        if dist_source[i] == pick_dist_elm:
            if (verbose > 2):
                print("append {}th element of vec with len ({})".format(i, len(other_dist)))
            other_dist_picks.append(other_dist[i])
    return np.array(other_dist_picks)


def count_in_vec(v, x):
    i = 0
    for j in v:
        i += j == x
    # print("there are {} of {} in {}".format(i,x,v))
    return i


def calc_count_of_labels(V):
    vu = np.unique(V)
    n = len(vu)
    V_cnt = np.zeros(n, dtype=float)
    for i in range(n):
        x = vu[i]
        V_cnt[i] = count_in_vec(V, x)
    return V_cnt


def calc_nmi(class_labs, kluster_annot, verbose=2, ck=['class', 'kluster']):
    n = len(class_labs)
    if verbose > 0:
        print("**********START************")
    clent, cllb_uniq, cllb_ent, cllb_cnt = calc_entropy_of_dist(class_labs, verbose=verbose, ck=ck[0], cond='')
    klent, klan_uniq, klan_ent, klan_cnt = calc_entropy_of_dist(kluster_annot, verbose=verbose, ck=ck[1], cond='')
    if verbose > 0:
        print("clent=", clent)
        print("klent=", klent)
    mui = 0
    for kj in range(klan_cnt):  # 1,1,2,2,1,1,1,1,2
        cur_k = klan_uniq[kj]
        cl_k_j = grab_dist_from_dist(kluster_annot, cur_k, class_labs, verbose=verbose)
        for ci in range(cllb_cnt):  # a,a,a,a,b,b,b,b,b
            cur_c = cllb_uniq[ci]
            nk = count_in_vec(kluster_annot, cur_k)
            nc = count_in_vec(class_labs, cur_c)
            nkc = count_in_vec(cl_k_j, cur_c)
            if (nkc == 0 or nk == 0 or nc == 0):
                # print("k({}), c({}),nkc({}),nk({}),nc({}),n({})".format(cur_k,cur_c,nkc,nk,nc,n))
                continue

            P_kc = nkc / n
            P_k = nk / n
            P_c = nc / n
            mui_add = P_kc * np.log2((n * nkc) / (nk * nc))
            mui = mui + mui_add
            if verbose > 1:
                print("k({}), c({}),nkc({}),nk({}),nc({}),n({})".format(cur_k, cur_c, nkc, nk, nc, n))
                print("kj({}), ci({}), P_kc({:4.2f}),P_k({:4.2f}), P_c({:4.2f}), mui_add({:4.2f})".format(cur_k, cur_c, P_kc,
                                                                                                    P_k, P_c, mui_add))

    if verbose > 0:
        print("mutualinf = {}".format(mui))

    val_nmi = (2 * mui) / (clent + klent)
    if verbose > 0:
        print("normalized mutual information = {}".format(val_nmi))
        print("***********END*************")
    return val_nmi

def run_calc_mi_nmi(class_labs, kluster_annot):
    calc_nmi(class_labs=class_labs, kluster_annot=kluster_annot, verbose=1, ck=['class', 'kluster'])
    calc_nmi(class_labs=kluster_annot, kluster_annot=class_labs, verbose=1, ck=['kluster', 'class'])
    print("mi(c,k)=", mi(class_labs, kluster_annot))
    print("mi(k,c)=", mi(kluster_annot, class_labs))
    print("nmi(c,k)=", nmi(class_labs, kluster_annot))
    print("nmi(k,c)=", nmi(kluster_annot, class_labs))

# delete this
def calc_nmi_tests():
    ex3_c = [0, 0, 0, 0, 1, 1, 1, 1, 1]
    ex3_k = [1, 1, 2, 2, 1, 1, 1, 1, 2]
    calc_nmi(class_labs=ex3_k, kluster_annot=ex3_c, verbose=2)
    calc_nmi(class_labs=ex3_c, kluster_annot=ex3_k, verbose=2)

    ex1_c = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]
    ex1_k = [0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1]
    calc_nmi(class_labs=ex1_c, kluster_annot=ex1_k, verbose=2)
    calc_nmi(class_labs=ex1_k, kluster_annot=ex1_c, verbose=2)

    ex2_c = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]
    ex2_k = [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1]
    calc_nmi(class_labs=ex2_c, kluster_annot=ex2_k, verbose=3)
    calc_nmi(class_labs=ex2_k, kluster_annot=ex2_c, verbose=3)

def retrieve_web_str_dict():
    web_str_dict = {
        "ogulcan_32": "1-1;2-2;3-3;4-4,11,24;5-5,12,13,14,18,25,29,30;6-6;7-7;8-8,9,21,23;9-10,15,22,28,32;10-16;11-17;12-19,26;13-20,27;14-31",
        "simal_32": "1-1,6,9,11,21,23;2-2,5,7,12,13,14,18,25,29,30;3-3,16;4-4,8,19,24,26;5-17;6-20,27;7-10,15,22,28,32;8-31",
        "alp_32": "1-1;2-2;3-3;4-4,11,24;5-5,12,14,25,29,30;6-6;7-7;8-8;9-9,21,23;10-10,15,28,32;11-13,18;12-16;13-17;14-19,26;15-20,27;16-22;17-31",
        "doga_256": "1-1,29,69,111,177,204,227;2-2,41,64,73,109,122,129,149,179,183;3-3,13,20,32,38,39,40,43,59,62,66,67,74,77,78,88,90,92,100,104,112,118,119,120,127,141,143,146,147,148,158,159,162,167,171,184,188,192,195,199,200,202,214,233,234,235,236,237,241,244,246,247,248,250,251;4-4,44,63,87,139,187,239;5-5,21,49,60,72,106,128,130,136,142,178,193,206,230,249;6-6,22,46,50,80,89,93,98,144,212,218,242;7-7,8,15,17,25,28,31,33,47,54,57,58,65,70,79,83,91,94,96,103,105,108,110,114,117,123,131,134,137,138,140,145,151,156,170,172,185,196,201,205,225,228,229,238,240,256;8-9,30,243;9-10,16,23,164,173,190,209;10-11,24,36,42,53,56,61,82,124,154,155,157,166,175,181,207,211,221,252;11-12,51,71,81,125,165,194,213,223;12-14,45,52,95,126,153,189,191,216,220,253,255;13-26,27,97,113,176,180,182,254;14-35,99,121,210,217,224,226,245;0-84,116,135,160,161,169,174,186,203,208,232;15-19,68,101,115;16-86,102,132,152,163,222;17-107,133,215;18-55,75,150,198,219,231;19-85,168;20-76,197;21-18,34,37,48",
        "doga_256_8": "1-1,29,69,111,177,204,227;2-2,41,64,73,109,122,129,149,179,183;3-3,13,32,38,40,43,59,62,66,67,77,78,120,127,141,143,147,158,159,162,171,188,195,200,202,241,246,247,248;4-4,44,63,87,139,187,239;5-5,21,49,60,72,106,128,130,136,142,178,230,249;6-6,22,46,50,80,89,93,98,144,212,218,242;7-7,8,15,17,25,28,31,33,47,54,57,58,65,70,79,83,91,94,96,103,105,108,110,114,117,123,131,134,137,138,140,145,172,185,196,201,205,225,228,240,256;8-9,30,169,174,243;9-10,16,23,86,102,132,152,163,164,173,190,209,222;10-11,24,36,42,53,56,61,82,124,154,155,157,166,175,181,207,211,221,252;11-12,51,71,81,125,165,194,213,223;12-14,45,52,95,126,153,160,189,191,216,220,253,255;13-18,20,34,37,39,48,74,88,90,92,100,104,112,118,119,146,148,167,184,192,199,214,233,234,235,236,237,244,250,251;14-19,68,101,115;15-26,27,97,113,176,180,182,254;16-35,99,121,210,217,224,226,245;0-55,76,161,186,203;17-75,150,193,198,206,219,231;18-107,133,215;19-116,197;20-84,151,156,170,229,232,238;21-85,135,168,208",
        "ufuk_256": "1-2,41,64,73,109,122,129,149,179,183;2-1,29,44,57,58,63,65,69,79,85,87,96,111,123,135,139,156,168,177,185,186,187,204,232,245;3-3,10,16,18,20,22,23,34,38,39,40,43,46,50,51,71,80,86,89,90,93,95,98,100,102,118,119,120,125,126,132,148,152,163,164,167,173,184,189,190,209,212,213,218,220,222,242,253;4-4,5,9,12,14,19,21,25,30,35,45,49,52,55,60,68,72,75,76,81,99,101,106,107,115,116,121,128,130,133,136,142,144,150,151,153,165,169,174,178,180,182,191,193,194,197,198,203,206,210,215,216,217,219,223,224,226,230,231,239,243,249,254,255;5-6,7,31,47,70,83,84,91,94,110,117,140,160,170,201,238;6-8,15,17,26,28,33,36,54,61,103,105,108,114,131,134,137,138,145,172,196,205,225,227,228,229,240,256;7-11,13,24,27,32,37,42,48,53,56,59,62,66,67,74,77,78,82,88,92,97,104,112,113,124,127,141,143,146,147,154,155,157,158,159,161,162,166,171,175,176,181,188,192,195,199,200,202,207,208,211,214,221,233,234,235,236,237,241,244,246,247,248,250,251,252",
        "alp_256": "1-1,69,111,177,204;2-2,64,109,122,129,149,183;3-3,20,34,37,38,39,43,59,62,66,88,119,120,146,147,167,195,199,200,233,236,244,246,247;4-5;5-4,44,63,87,139,187,239;6-6,22,46,50,80,89,93,98,144,212,218,242;7-7,47,70,91,110,117,140,201;8-8,15,17,28,33,54,57,58,65,79,96,103,105,108,114,123,131,134,137,138,145,172,185,196,205,225,228,240,256;9-9,30,151,156,169,174,229,232,243;10-10,16,23,86,102,132,152,163,164,173,190,209,222;11-11,24,36,42,53,56,61,82,124,135,154,155,157,166,175,181,207,211,221;12-12,81,194;13-13,32,40,67,77,78,127,141,143,158,159,162,171,188,202,241,248;14-14,45,52,95,126,153,160,189,191,220,253,255;15-18,48,74,90,92,100,104,112,118,148,184,192,214,234,235,237,250,251;16-19,68,101,115;17-21,49,60,106,128,130,136,142,178,230,249;18-25,31,83,84,94,170,238;19-26,113,176,180,254;20-27,97,182;0-29,186,227;21-35,99,121,210,217,224,226,245;22-41,73,179;23-51;24-55,72,75,150,193,198,206,219,231;25-71,125,165,213,223;26-76,116,197;27-85,168,208;28-107,133,215;29-161,252;30-203;31-216",
        "ogulcan_256": "1-2,64,73,109,122,149,183;2-3,13,20,32,37,38,40,43,46,48,59,62,66,67,77,78,120,127,141,143,147,158,159,162,171,184,188,195,200,202,241,246,247,248,251;3-4,44,63,87,139,187,239;4-6,22,50,80,89,93,98,144,212,218,242;5-9,174,232,243;6-10,16,23,86,102,132,152,163,164,173,190,209,222;7-11,24,36,42,53,56,61,76,82,85,124,135,154,155,157,166,168,175,181,207,208,211,221;8-12,71,81,125,165,194,213,223;9-14,45,52,95,126,153,160,189,191,216,220,253,255;10-18,34,39,74,88,90,92,100,104,112,118,119,146,148,167,192,199,214,233,234,235,236,237,244,250;11-19,68,101,115;12-5,21,49,60,72,106,128,130,136,142,230,249;13-8,15,17,25,30,31,33,47,54,57,58,65,79,83,84,96,103,108,110,123,137,138,145,151,156,169,170,172,185,205,225,229,238,256;14-26,27,97,113,176,180,182,254;15-28,114,131,134,196,228,240;16-1,29,69,111,177,204,227;17-35,99,121,210,217,226;18-41,129,179;19-51;20-55,75,150,193,198,206,219,231;21-7,70,91,94,117,140,201;22-105;23-107,133,178,215;24-116,197;25-161,252;26-186;27-203;28-224,245",
        "lale_256": "1-1,29,69,111,177,204,227;2-2,41,64,73,109,122,129,149,179,183;3-3,13,32,38,40,43,59,62,66,67,77,78,120,127,141,143,146,147,148,158,159,162,171,188,192,195,202,235,237,241,246,247,248,250;4-4,44,63,87,139,187,239;5-5,72,75,150,198,206,219,231;6-6,22,50,89,144,212,242;0-7,25,55,105,186,193;7-8,15,17,28,33,54,57,58,65,79,96,103,108,114,123,131,134,137,138,145,169,172,185,196,205,225,228,229,240,256;8-9,26,30,174,232,243;9-10,16,23,86,102,132,152,163,164,173,190,209,222;10-11,24,36,42,53,56,61,76,82,124,135,154,155,157,166,175,181,207,211,221,252;11-12,51,81,125,194,213;12-14,45,52,80,95,98,126,153,160,189,191,216,220,253,255;13-19,68,101,115;14-21,49,60,106,128,130,136,142,178,230,249;15-27,97,113,176,180,182,254;16-31,47,70,83,84,91,94,110,117,140,170,201,238;17-71,165,223;18-85,168,208;19-35,99,121,210,217,224,226,245;20-107,133,215;21-116,197;22-18,20,34,37,39,48,74,88,90,92,100,104,112,118,119,167,184,199,200,214,233,234,236,244,251;23-161;24-151,156;25-203;26-46,93,218",
    }
    return web_str_dict

def get_annotation_dict_from_web_str_dict(web_str_dict, verbose=0):
    ann_dict = {}
    for k in web_str_dict:
        if "_32" in k:
            print(32, k)
            ann_dict[k] = get_annotation_labels_from_str(web_str_dict[k], 32, verbose=verbose)
        if "_256" in k:
            print(256, k)
            ann_dict[k] = get_annotation_labels_from_str(web_str_dict[k], 256, verbose=verbose)

    return ann_dict

def cal_nmi_by_of_web_str_dict(web_str_dict, ann_dict, int_vec, cluster_count_use, verbose=0):
    for k in web_str_dict:
        if "_32" in k and  cluster_count_use== 32:
            print("{}_nmi_score={}".format(k, calc_nmi(int_vec, ann_dict[k], verbose=verbose, ck=['class', 'kluster'])))
        if "_256" in k and cluster_count_use == 256:
            print("{}_nmi_score={}".format(k, calc_nmi(int_vec, ann_dict[k], verbose=verbose, ck=['class', 'kluster'])))

def analyze_web_str_dict(web_str_dict, ann_dict, int_vec, uniq_X, cluster_count_use, verbose=0):
    cohens_kappa_dict = {}
    nmi_dict = {}
    classRet_dict = {}
    for k in web_str_dict:
        if "_32" in k and cluster_count_use != 32:
            continue
        if "_256" in k and cluster_count_use != 256:
            continue
        nmi_dict[k] = calc_nmi(int_vec, ann_dict[k], verbose=verbose, ck=['class', 'kluster'])
        print("{}_nmi_score={}".format(k, nmi_dict[k]))
        classRet_dict[k], _confMat, c_pdf, kr_pdf = funcH.calcCluster2ClassMetrics(labels_true=int_vec, labels_pred=ann_dict[k],
                                                                           labelNames=uniq_X, predictDefStr=k,
                                                                           verbose=verbose)
        print("Cohens Kappa for {}".format(k))
        cohens_kappa_dict[k] = cohens_kappa_from_confmat(_confMat, verbose=1)

    return cohens_kappa_dict, nmi_dict, classRet_dict

def cohens_kappa_from_confmat(_conf_mat, verbose=0):
    _asum = np.sum(np.sum(_conf_mat))
    _csum = np.sum(_conf_mat, axis=0)
    _rsum = np.sum(_conf_mat, axis=1)
    _dsum = np.sum(np.diag(_conf_mat))
    _msum = _csum * _rsum
    if verbose > 2:
        print("_asum = ", _asum)
        print("_csum = ", _csum)
        print("_rsum = ", _rsum)
        print("_dsum = ", _dsum)
        print("_msum = ", _msum)

    k00 = _dsum
    k01 = np.sum(_msum) / _asum
    k10 = _asum
    K = (k00 - k01) / (k10 - k01)
    if verbose > 0:
        print("K({:4.2f}) = (k00-k01)/(k10-k01)".format(K))
    if verbose > 1:
        print("({:4.2f}-{:4.2f})/({:4.2f}-{:4.2f})".format(k00, k01, k10, k01))
    return K