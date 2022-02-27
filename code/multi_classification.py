# -*- coding: utf-8 -*
from sklearn import svm, datasets
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.multiclass import OneVsOneClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier
from sklearn import tree
import matplotlib.pyplot as plt
#from sklearn.metrics import roc_curve, roc_auc_score, auc, plot_roc_curve
from sklearn.preprocessing import label_binarize
import numpy as np
from collections import Counter



datatrain = '/home/xianyu/software/openrainbow/xm_data/All_data/train_data/dataSetTrain.txt'
datatest = '/home/xianyu/software/openrainbow/xm_data/All_data/test_data/dataSetTest.txt'

class multi_classification:

    def __init__(self, data_train_path=datatrain, data_test_path=datatest):
        self.data_train_path = data_train_path
        self.data_test_path = data_test_path
        self.sample_split, self.x_test, self.y_test = self.input_dataset()
        self.class_num = len(Counter(self.y_test))

        self.clf = None
        pass

    # 将分类器的ovo概率矩阵组合
    def proba_matrix_comb(self):
        proba_matrix_comb = []
        proba_matrix_comb.append(self.ramdomforst())
        proba_matrix_comb.append(self.adaBoost())
        proba_matrix_comb.append(self.knn())
        #proba_matrix_comb.append(self.gaussiannb())
        #proba_matrix_comb.append(self.c45())
        #proba_matrix_comb.append(self.svm())
        #proba_matrix_comb.append(self.tree())
        #proba_matrix_comb.append(self.logisticRegression())
        return proba_matrix_comb
        pass

    # clf：基分类器
    def ovoclassifier(self, clf):
        """
        数据分为0,1,2,3,4共五类（self.class_num=5）；两两组合01，02，03，...，34共十个训练集，分别用这十个训练集训练分类器，分别对测试集预测得到十组proba_matrix
        comb_label_proba = {'01': proba_matrix of 01, '02': proba_matrix of 02,
        '03':..'04':..'12':..'13':..'14':..'23':..'24':..'34':..}
        """
        comb_label_proba = {}

        for row in range(self.class_num-1):
            for col in range(row+1, self.class_num):
                # 将标签为row和col的样本取出并合并做训练集
                dataset = self.sample_split[row]
                dataset = np.vstack((dataset, self.sample_split[col]))

                y_train = dataset[:, -1]
                x_train = np.delete(dataset, -1, axis=1)

                clf.fit(x_train, y_train)
                proba = clf.predict_proba(self.x_test)
                '''
                当标签row=0，col=1，proba_matrix=[[0.4 0.6 0.  0.  0. ]   proba_matrix.shape:(n_sample of test, class_num)
                                                 [1.  0.  0.  0.  0. ]
                                                 [1.  0.  0.  0.  0. ]
                                                 ...
                                                 [0.9 0.1 0.  0.  0. ]
                                                 [1.  0.  0.  0.  0. ]
                                                 [1.  0.  0.  0.  0. ]]
                '''
                proba_matrix = np.zeros((len(self.x_test), self.class_num))
                proba_matrix[:, [row]] = proba[:, [0]]
                proba_matrix[:, [col]] = proba[:, [1]]

                comb_label_proba.setdefault(str(row)+str(col), proba_matrix)
                # print('-----------------------\n\n\n\n\n')
                # print(clf.predict_proba(self.x_test))
                # print(clf.predict(self.x_test))
                # print(proba_matrix)

                pass
        #print('classififer\n\n\n')
        #print(comb_label_proba)

        sample_proba = []
        for i in range(len(self.x_test)):
            arr = []
            for key in comb_label_proba.keys():
                arr.append(comb_label_proba[key][i])
            arr = np.array(arr)
            sample_proba.append(arr)
            pass
        # print('66666666666666666666666666666666666\n\n\n\n\n')
        # print(sample_proba)

        return sample_proba
        pass

    def adaBoost(self, ):
        ab = AdaBoostClassifier(n_estimators=100)
        proba_matrix = self.ovoclassifier(ab)
        return proba_matrix

    def logisticRegression(self,):
        lr = LogisticRegression(solver='newton-cg', C=1, max_iter=100)
        proba_matrix = self.ovoclassifier(lr)
        return proba_matrix

    def c45(self, ):
        c45 = tree.DecisionTreeClassifier(criterion='entropy')
        proba_matrix = self.ovoclassifier(c45)
        return proba_matrix

    def tree(self, ):
        dt = tree.DecisionTreeClassifier()
        proba_matrix = self.ovoclassifier(dt)
        return proba_matrix

    def gaussiannb(self):
        gauss = GaussianNB()
        proba_matrix = self.ovoclassifier(gauss)
        return proba_matrix

    def knn(self, ):
        knn = KNeighborsClassifier(algorithm='kd_tree', n_neighbors=3)
        proba_matrix = self.ovoclassifier(knn)
        return proba_matrix

    # svm内置有OVO多分类模式
    def svm(self, ):
        clf = svm.SVC(kernel='rbf', probability=True, gamma='auto')  # 创建分类器'linear', 'poly', 'rbf', 'sigmoid', 'precomputed'
        proba_matrix = self.ovoclassifier(clf)
        return proba_matrix

    def ramdomforst(self, ):
        rf = RandomForestClassifier(n_estimators=10, max_depth=None, min_samples_split=2, random_state=0)
        proba_matrix = self.ovoclassifier(rf)
        return proba_matrix

    def input_dataset(self):
        dataSetTrain = np.loadtxt(self.data_train_path)
        dataSetTest = np.loadtxt(self.data_test_path)
        labelArrTrain = dataSetTrain[:, -1]
        labelArrTest = dataSetTest[:, -1]

        # 将训练集按照类别切分存到字典中，sample_split = {label：sample}
        train_sample_split = {}

        for row in range(len(dataSetTrain)):
            if dataSetTrain[row][-1] not in train_sample_split.keys():
                train_sample_split.setdefault(dataSetTrain[row][-1], [])
            else:
                train_sample_split[dataSetTrain[row][-1]].append(dataSetTrain[row])
            pass

        for key in train_sample_split.keys():
            train_sample_split[key] = np.array(train_sample_split[key])
        #print(train_sample_split)

        dataSetTrain = np.delete(dataSetTrain, -1, axis=1)
        dataSetTest = np.delete(dataSetTest, -1, axis=1)

        return train_sample_split, dataSetTest, labelArrTest

        pass

    pass

# 基分类器 adaBoost, logisticRegression, c45, tree, gaussiannb, knn, svm, ramdomforst
multi_class = multi_classification(datatrain, datatest)
proba_matrix = multi_class.proba_matrix_comb()
#multi_class.draw_roc()
print('proba_matrix')
#print(proba_matrix)