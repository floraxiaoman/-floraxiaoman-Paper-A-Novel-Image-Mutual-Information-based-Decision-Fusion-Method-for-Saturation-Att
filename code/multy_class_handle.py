# -*- coding: utf-8 -*
# from openrainbow.OVO_Five.Mass_function.Mass_function import mass_function as BPA
#
#
from openrainbow.OVO_Five.DS_formula.other_ds_xm import other_DS as DS
from openrainbow.OVO_Five.comp_test.comp_one import comp_one
from openrainbow.OVO_Five.comp_test.comp_two_pr import comp_two_pr
from openrainbow.OVO_Five.comp_test.comp_three_svmds import comp_three_svmds

from multi_classification import multi_classification
import numpy as np
import cv2.cv2 as cv2
from sklearn.metrics import precision_score,f1_score,recall_score
import time
from sklearn import metrics
import math
import psutil

from scipy import interp

from line_profiler import LineProfiler

from functools import wraps
import scipy.ndimage
from numpy.ma.core import exp
from scipy.constants.constants import pi

from skimage.measure import compare_ssim
import argparse
import imutils
import cv2
import os



class_number=5
multclass_base_number = class_number*(class_number-1)*0.5
evidence_number=3
example_number=1200
class multy_class_handle():
    def __init__(self):
        self.manman_input = []
        self.sample_matrix = []
        self.DS_lable = []
        self.DS_lable_vector = []
        self.y_true = []
        self.DS_score = []
        self.mutil_class = multi_classification()
        pass
    # =====================================以下 计算图片相似度的方法===========================================

    def compute_ssim(self,imageA, imageB):
        # # 2. Construct the argument parse and parse the arguments
        # ap = argparse.ArgumentParser()
        # ap.add_argument("-f", "--first", required=True, help="Directory of the image that will be compared")
        # ap.add_argument("-s", "--second", required=True, help="Directory of the image that will be used to compare")
        # args = vars(ap.parse_args())
        #
        # # 3. Load the two input images
        # imageA = cv2.imread(args["first"])
        # imageB = cv2.imread(args["second"])

        # # 4. Convert the images to grayscale
        # grayA = cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY)
        # grayB = cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY)

        # 5. Compute the Structural Similarity Index (SSIM) between the two
        #    images, ensuring that the difference image is returned
        (score, diff) = compare_ssim(imageA, imageB,win_size=3, multichannel=True,full=True)
        diff = (diff * 255).astype("uint8")

        # 6. You can print only the score if you want
        # print("SSIM: {}".format(score))
        #print score

        return score

    # 均值哈希算法
    def aHash(self, img):
        # 缩放为8*8
        img = cv2.resize(img, (8, 8))
        # 转换为灰度图
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # s为像素和初值为0，hash_str为hash值初值为''
        s = 0
        hash_str = ''
        # 遍历累加求像素和
        for i in range(8):
            for j in range(8):
                s = s + gray[i, j]
        # 求平均灰度
        avg = s / 64
        # 灰度大于平均值为1相反为0生成图片的hash值
        for i in range(8):
            for j in range(8):
                if gray[i, j] > avg:
                    hash_str = hash_str + '1'
                else:
                    hash_str = hash_str + '0'
        return hash_str

    # 差值感知算法
    def dHash(self, img):
        # 缩放8*8
        img = cv2.resize(img, (9, 8))
        # 转换灰度图
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        hash_str = ''
        # 每行前一个像素大于后一个像素为1，相反为0，生成哈希
        for i in range(8):
            for j in range(8):
                if gray[i, j] > gray[i, j + 1]:
                    hash_str = hash_str + '1'
                else:
                    hash_str = hash_str + '0'
        return hash_str

    # 感知哈希算法(pHash)
    def pHash(self, img):
        # 缩放32*32
        img = cv2.resize(img, (32, 32))  # , interpolation=cv2.INTER_CUBIC

        # 转换为灰度图
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # 将灰度图转为浮点型，再进行dct变换
        dct = cv2.dct(np.float32(gray))
        # opencv实现的掩码操作
        dct_roi = dct[0:8, 0:8]

        hash = []
        avreage = np.mean(dct_roi)
        for i in range(dct_roi.shape[0]):
            for j in range(dct_roi.shape[1]):
                if dct_roi[i, j] > avreage:
                    hash.append(1)
                else:
                    hash.append(0)
        return hash

    # 通过得到RGB每个通道的直方图来计算相似度
    def classify_hist_with_split(self, image1, image2, size=(256, 256)):
        # 将图像resize后，分离为RGB三个通道，再计算每个通道的相似值
        image1 = cv2.resize(image1, size)
        image2 = cv2.resize(image2, size)
        sub_image1 = cv2.split(image1)
        sub_image2 = cv2.split(image2)
        sub_data = 0
        for im1, im2 in zip(sub_image1, sub_image2):
            sub_data += self.calculate(im1, im2)
        sub_data = sub_data / 3
        return sub_data

    # 计算单通道的直方图的相似值
    def calculate(self, image1, image2):
        hist1 = cv2.calcHist([image1], [0], None, [256], [0.0, 255.0])
        hist2 = cv2.calcHist([image2], [0], None, [256], [0.0, 255.0])
        # 计算直方图的重合度
        degree = 0
        for i in range(len(hist1)):
            if hist1[i] != hist2[i]:
                degree = degree + (1 - abs(hist1[i] - hist2[i]) / max(hist1[i], hist2[i]))
            else:
                degree = degree + 1
        degree = degree / len(hist1)

        return degree

    # Hash值对比
    def cmpHash(self, hash1, hash2):
        n = 0
        # hash长度不同则返回-1代表传参出错
        if len(hash1) != len(hash2):
            return -1
        # 遍历判断
        for i in range(len(hash1)):
            # 不相等则n计数+1，n最终为相似度
            if hash1[i] != hash2[i]:
                n = n + 1
        return n

    # =====================================以上 计算图片相似度的方法===========================================

    def matrix_to_pic(self, matrix, index):
        # print "matrix_to_pic"
        array = np.array(matrix)

        for i in range(len(array)):
            for j in range(len(array[0])):
                if matrix[i][j]>0:
                    f1 = 1.0/(1+math.exp(0-i))
                    f2 = 0.1*i
                    array[i][j] = array[i][j]*255*(f1+f2)*0.5

        img = array
        # img = array.reshape(class_number,-1)*255

        # begin_time = time.time()

        # 下面两行是生成图片的代码
        # cv2.imshow('img'+str(index),img)
        # if index == 0:
        cv2.imwrite('/home/xianyu/software/openrainbow/openrainbow/mm_create_pic/img'+str(index)+'.jpg',img)
            # end_time = time.time()
            # print "输出生成图片的时间"
            # print end_time - begin_time


        img_str = '../../mm_create_pic/img'+str(index)+'.jpg'

        return img_str

    def matrix_to_vector(self, matrix):
        vector = []
        for col in range(len(matrix[0])):
            vector.append(0)
        for row in range(len(matrix)):
            for col in range(len(matrix[0])):
                vector[col]+=matrix[row][col]
        return vector

    def pic_to_weight(self,pic_s,idx):
        # print "pic_to_weight"
        similar_list = []
        weight_list = []
        # print '=============pic_s=========='
        # print len(pic_s)
        for base_index in range(len(pic_s)):
            similar_sum = 0
            for index in range(len(pic_s)):
                if index == base_index:
                    continue
                img1 = cv2.imread(pic_s[base_index])
                img2 = cv2.imread(pic_s[index])
                # =================选择其中一种计算相似度=============================

                # hash1 = self.aHash(img1)
                # hash2 = self.aHash(img2)
                # n = self.cmpHash(hash1, hash2)
                # print '均值哈希算法相似度：' + str(n)
                #
                # hash1 = self.dHash(img1)
                # hash2 = self.dHash(img2)
                # n = self.cmpHash(hash1, hash2)
                # print '差值哈希算法相似度：' + str(n)
                #
                # hash1 = self.pHash(img1)
                # hash2 = self.pHash(img2)
                # n = self.cmpHash(hash1, hash2)
                # print '感知哈希算法相似度：' + str(n)

                # n = self.classify_hist_with_split(img1, img2)
                n = self.compute_ssim(img1,img2)
                # print '三直方图算法' + '图片' + str(base_index) + '与图片' + str(index) + '的相似度：' + str(n)

                if idx==4:
                    print "相似度"
                    print base_index,index,n

                similar_sum += n
            similar_list.append(similar_sum)
        # print "-----------输出相似度sum列表-------------"
        # print similar_list
        weight_list = self.matrix_normalization_by_row([similar_list])
        # print "-----------输出相似度sum归一化后列表-------------"
        # print weight_list[0]
        return weight_list[0]

    def vector_multi_weight(self,vector_matrix,weight_list):
        # print "vector_multi_weight"
        for row in range(len(vector_matrix)):
            for col in range(len(vector_matrix[row])):
                vector_matrix[row][col]*=weight_list[row]
        return vector_matrix

    # 矩阵按行归一化
    def matrix_normalization_by_row(self,matrix):
        for row_index in range(len(matrix)):
            col_sum = 0
            for col_index in range(len(matrix[row_index])):
                col_sum += matrix[row_index][col_index]
            for col_index in range(len(matrix[row_index])):
                matrix[row_index][col_index] = matrix[row_index][col_index] / col_sum
        return matrix

    "Step1 调用mass函数得到BPA和最终Lable"
    def attain_input(self):
        # self.manman_input = BPA().for_D_S_call()

        self.manman_input = self.mutil_class.proba_matrix_comb()


        # print self.manman_input

        print len(self.manman_input[0])
        for sample_index in range(example_number):
            self.sample_matrix.append([])
        for base_index in range(len(self.manman_input)):  # base_index表示选一个多分类器
            # print base_index
            # print "-------multiclass"
            for sample_index in range(len(self.manman_input[base_index])):  # sample_index表示一个多分类器中的一个样本
                # print self.manman_input[base_index][sample_index]
                # print "---------------------sample", sample_index
                self.sample_matrix[sample_index].append(self.manman_input[base_index][sample_index])

        #输出第4个例子的三个概率矩阵
        print self.sample_matrix[4]

        # print '===================输出样本为0 3个矩阵======================'
        # for base_index in range(evidence_number):
        #     print base_index
        #     print self.sample_matrix[0][base_index]
        # print '===================输出样本为0 第一个分类器的向量======================'
        # print self.matrix_to_vector(self.sample_matrix[0][0])
        # print '===================输出样本为0 测试第一个分类器的向量×权重======================'
        # print self.vector_multi_weight(self.sample_matrix[0][0],[1,2,3])

    def _evul_Final_DS(self,idx):
        # y_true_, x_test = BPA()._test_attain_()
        # print y_true_
        # y_true = y_true_[0:example_number]
        # print "DS---------------------Y_TRUE"
        # print y_true
        # print "DS-------------------Y_predict"
        y_true = self.y_true

        print y_true
        print len(y_true)

        y_predict = self.DS_lable
        # print y_predict
        # print "-------DS_formula--------DS_formula------------DS_formula------DS_formula----------DS_formula------DS_formula-----"

        self._evaluat_(y_true, y_predict,idx)


        self.get_chaos_matrix(y_true, y_predict)

    def get_chaos_matrix(self,y_true, y_predict):
        chaos_matrix = np.zeros((class_number,class_number),dtype = int)
        print chaos_matrix
        for idx in range(len(y_true)):
            chaos_matrix[int(y_true[idx])][int(y_predict[idx])] += 1
        print ('---------输出混沌矩阵--------')
        print chaos_matrix
        return chaos_matrix

    def data_x2_handle(self,y_true, y_predict):
        print("data_x2_handle")
        data_true = [0,0,0,0,0]
        data_pre = [0, 0, 0, 0, 0]

        for idx in range(len(y_true)):
            data_true[int(y_true[idx])]+=1
            data_pre[int(y_predict[idx])] += 1

        return data_true,data_pre



    def _evaluat_(self, y_true, y_pred,idx):
       
        print('Accuracy------all‘ +''+: ' + str(metrics.accuracy_score(y_true, y_pred)))
        print('------Weighted------')
        print('Weighted precision', precision_score(y_true, y_pred, average='weighted'))
        print('Weighted recall', recall_score(y_true, y_pred, average='weighted'))
        print('Weighted f1-score', f1_score(y_true, y_pred, average='weighted'))
        print('------Macro------')
        print('Macro precision', precision_score(y_true, y_pred, average='macro'))
        print('Macro recall', recall_score(y_true, y_pred, average='macro'))
        print('Macro f1-score', f1_score(y_true, y_pred, average='macro'))
        print('------Micro------')
        print('Micro precision', precision_score(y_true, y_pred, average='micro'))
        print('Micro recall', recall_score(y_true, y_pred, average='micro'))
        print('Micro f1-score', f1_score(y_true, y_pred, average='micro'))

        # ---------------------------以下 计算平台指标-------------------------------------------

        cnf_matrix = self.get_chaos_matrix(y_true, y_pred)

        data_true_x2,data_pre_x2 = self.data_x2_handle(y_true, y_pred)

        print(data_true_x2, data_pre_x2)

        # self.get_tpr_fpr(cnf_matrix)

        print '---------tpr_fpr的结果-----------'
        print len(self.get_tpr_fpr(cnf_matrix))
        print self.get_tpr_fpr(cnf_matrix)

        # y_true_arr = np.array(y_true)
        # y_pred_score = np.array(self.DS_score)
        # y_pred_arr = np.array(y_pred)

        y_true_arr = []
        for i in range(example_number):
            tmp = []
            for j in range(class_number):
                if j == y_pred[i]:
                    tmp.append(1)
                else:
                    tmp.append(0)
            y_true_arr.append(tmp)

        y_true_arr = np.array(y_true_arr)
        ds_score_arr = np.array(self.DS_score)

        fpr = dict()
        tpr = dict()
        precision = dict()
        recall = dict()
        roc_auc = dict()
        for i in range(class_number):
            fpr[i], tpr[i], _ = metrics.roc_curve(y_true_arr[:, i], ds_score_arr[:, i])
            precision[i], recall[i], thresholds = metrics.precision_recall_curve(y_true_arr[:, i], ds_score_arr[:, i])
            roc_auc[i] = metrics.auc(fpr[i], tpr[i])

        # First aggregate all false positive rates
        all_fpr = np.unique(np.concatenate([fpr[i] for i in range(class_number)]))

        # Then interpolate all ROC curves at this points
        mean_tpr = np.zeros_like(all_fpr)
        for i in range(class_number):
            mean_tpr += interp(all_fpr, fpr[i], tpr[i])

        # Finally average it and compute AUC
        mean_tpr /= class_number
        fpr["macro"] = all_fpr
        tpr["macro"] = mean_tpr



        # print fpr, tpr, thresholds
        np.savetxt("txt/fpr_"+str(idx)+".txt", fpr["macro"], fmt="%.8f")
        np.savetxt("txt/tpr_"+str(idx)+".txt", tpr["macro"], fmt="%.8f")


        all_recall = np.unique(np.concatenate([recall[i] for i in range(class_number)]))  # 变成一纬去掉重复的
        # Then interpolate all ROC curves at this points
        mean_pre = np.zeros_like(all_recall)

        for i in range(class_number):
            mean_pre += interp(all_recall, precision[i], recall[i]) # 返回线性差值 返回所有all_recall在对应右边参数构成的坐标系中的函数值

        # # Finally average it and compute AUC
        # print 'mean_pre'
        # print mean_pre
        mean_pre /= class_number
        recall["macro"] = all_recall
        precision["macro"] = mean_pre
        np.savetxt("txt/recall_" + str(idx) + ".txt", recall["macro"], fmt="%.8f")
        np.savetxt("txt/precision_" + str(idx) + ".txt", precision["macro"], fmt="%.8f")

        # auc = metrics.auc(fpr, tpr)
        # print 'cnf_matrix：', cnf_matrix
        # after = self._zero_to_marix(cnf_matrix)
        # TPR, TNR, FPR, FNR, ACC = self._tpr_fpr(cnf_matrix)
        # # print "before--------------"
        # print "TPR"
        # print TPR
        # print "TNR"
        # print TNR
        # print "FPR"
        # print FPR
        # print "FNR"
        # print FNR
        # print "ACC"
        # print ACC
        # afterTPR, afterTNR, afterFPR, afterFNR, afterACC = self._tpr_fpr(after)
        # print "after--------------"
        # print "afterTPR"
        # print afterTPR
        # print "afterTNR"
        # print afterTNR
        # print "afterFPR"
        # print afterFPR
        # print "afterFNR"
        # print afterFNR
        # print "afterACC"
        # print afterACC
        # # 2、手动计算micro类型的AUCs

    def get_tpr_fpr(self, cnf_matrix):

        FP = cnf_matrix.sum(axis=0) - np.diag(cnf_matrix)
        FN = cnf_matrix.sum(axis=1) - np.diag(cnf_matrix)
        TP = np.diag(cnf_matrix)
        TN = cnf_matrix.sum() - (FP + FN + TP)

        FP = FP.astype(float)
        FN = FN.astype(float)
        TP = TP.astype(float)
        TN = TN.astype(float)
        # Sensitivity, hit rate, recall, or true positive rate
        TPR = TP / (TP + FN)
        # Specificity or true negative rate
        TNR = TN / (TN + FP)
        # Precision or positive predictive value
        PPV = TP / (TP + FP)
        # Negative predictive value
        NPV = TN / (TN + FN)
        # Fall out or false positive rate
        FPR = FP / (FP + TN)
        # False negative rate
        FNR = FN / (TP + FN)
        # False discovery rate
        FDR = FP / (TP + FP)
        # Overall accuracy
        ACC = (TP + TN) / (TP + FP + FN + TN)
        return np.mean(TPR), np.mean(TNR), np.mean(FPR), np.mean(FNR), np.mean(ACC)

    def _zero_to_marix(self, arr):
        arr_after = []
        find = 0
        replacewith = 0.000001
        print "before"
        print arr
        for i in range(0, len(arr)):
            arr1 = []
            for j in range(0, len(arr[i])):
                if arr[i][j] == find:
                    # print 1111
                    arr1.append(replacewith)
                else:
                    arr1.append(arr[i][j])
            arr_after.append(arr1)
        print "after"
        print np.array(arr_after)
        return np.array(arr_after)

    # ---------------------------以上 计算平台指标-------------------------------------------

    def mass_handle(self):



        self.attain_input()

        self.y_true = self.mutil_class.y_test
##
        #
        begin_time = time.time()


        print '===================下面输出1250个样本 ds结果======================'

        for example_index in range(example_number):
            # print "=====================样本" + str(example_index)
            pic_s = []
            weight_list = []
            vector_matrix = []
            matrix_for_ds = []
            for evidence_index in range(evidence_number):
                # if evidence_index == 0:
                #     print self.sample_matrix[example_index][evidence_index]
                pic_s.append(self.matrix_to_pic(self.sample_matrix[example_index][evidence_index],example_index*10+evidence_index))
                # if evidence_index == 0:
                #     print self.sample_matrix[example_index][evidence_index]
                vector_matrix.append(self.matrix_to_vector(self.sample_matrix[example_index][evidence_index]))
                # if evidence_index == 0:
                #     print self.sample_matrix[example_index][evidence_index]
            weight_list = self.pic_to_weight(pic_s, example_index)

            if example_index==4:
                print "weight_list"
                print weight_list

            # print '===================输出权重列表======================'
            # print weight_list
            # print '===================输出权重修正的结果======================'
            matrix_for_ds = self.vector_multi_weight(vector_matrix,weight_list)
            # print matrix_for_ds
            # print '===================输出矩阵归一化======================'
            matrix_for_ds = self.matrix_normalization_by_row(matrix_for_ds)

            if example_index==4:
                print "matrix_for_ds"
                print matrix_for_ds

            # print matrix_for_ds
            # print '===================输出ds计算的结果======================'
            vector_after_ds, K = DS()._DS_envience_calcul(matrix_for_ds)

            if example_index==4:
                print "vector_after_ds"
                print vector_after_ds

            # print vector_after_ds
            idx = np.argmax(vector_after_ds)
            self.DS_lable.append(idx)
            self.DS_score.append(vector_after_ds)
            for i in range(class_number):
                tmp = []
                if idx==i:
                    tmp.append(1)
                else:
                    tmp.append(0)
            self.DS_lable_vector.append(tmp)


        self._evul_Final_DS(0)



        end_time = time.time()

        print "输出实验的时间"
        print end_time - begin_time
      


if __name__ == '__main__':
    pid = os.getpid()
    print('pid : ', pid)
    multy_class_handle().mass_handle()
