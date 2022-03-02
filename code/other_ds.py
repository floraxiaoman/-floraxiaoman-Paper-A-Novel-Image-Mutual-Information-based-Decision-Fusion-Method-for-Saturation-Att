# -*- coding: utf-8 -*
import numpy as np
class other_DS():
    def _DS_envience_calcul(self,DS_input):
        print '=========DS_input======'
        print  len(DS_input)
        # print DS_input
        DS = DS_input
        pr1 = 1
        pr2 = 1
        pr3 = 1
        # %%
        # 证据矩阵
        matrix = []
        K_martix=[]
        for sample_index in range(len(DS)):
            m = np.array(DS[sample_index])
            #print m
            # 优先级矩阵
            pr = np.array([pr1, pr2, pr3])
            # 可信度向量
            crd = self._crd(m)
            discount_evidence = self.discount(m, pr, crd)
            K=1-np.sum(discount_evidence)
            matrix.append(discount_evidence)
            K_martix.append(K)
        return matrix,K_martix
    "Step5.3.1 可信度函数"
    def _crd(self, M):
        # print '=============M=============='
        # print M
        # print "m-shape"
        # print M.shape
        m, n = M.shape
        DM = np.ones((m, m))
        for i in range(m):
            for j in range(m):
                vec1 = M[i, :]
                vec2 = M[j, :]
                DM[i, j] = np.linalg.norm(vec1 - vec2)
        #    SM = ((np.sqrt(2) -DM)/np.sqrt(2))**2
        sm = (DM) / np.sqrt(2) / 2 + 0.5
        e = 4 * sm * (1 - sm)
        b = np.sum(e, axis=1) - 1
        c = sum(b)
        crd = b / c
        return crd
    "Step5.3.2 可信度函数"
    def muti_row(self, m, k):
        row_num = len(k)
        p, q = m.shape
        l = np.ones((p, q))
        for i in range(row_num):
            l[i, :] = m[i, :] * k[i]
        return l
    "Step5.3.3 可信度函数"
    def discount(self, m, pr, crd):
        m_trans = m.T
        d1 = np.sum(m_trans * pr * crd, axis=1)
        d2 = (d1 - min(d1)) / (max(d1) - min(d1))
        d3 = d2 / sum(d2)
        return d3