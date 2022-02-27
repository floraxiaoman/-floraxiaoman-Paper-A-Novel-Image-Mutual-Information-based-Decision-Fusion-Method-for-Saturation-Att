#!/usr/bin/env python
# encoding:utf-8
import itertools
import pickle
import numpy as np
import json
splite1=200 # train
splite2=240  # test

normal_data_train='/home/xianyu/software/openrainbow/xm_data/All_data/train_data/data.normal_4'
H_controller_data_train='/home/xianyu/software/openrainbow/xm_data/All_data/train_data/data.Hcontroller'
L_controller_data_train='/home/xianyu/software/openrainbow/xm_data/All_data/train_data/data.Lcontroller'
H_switch_data_train='/home/xianyu/software/openrainbow/xm_data/All_data/train_data/data.Hswitch'
H_switch_data_train_1='/home/xianyu/software/openrainbow/xm_data/All_data/train_data/data.Hswitch_1'
L_switch_data_train='/home/xianyu/software/openrainbow/xm_data/All_data/train_data/data.Lswitch'
L_switch_data_train_1='/home/xianyu/software/openrainbow/xm_data/All_data/train_data/data.Lswitch_1'
#L_switch_data_train_2='/home/xianyu/software/openrainbow/xm_data/All_data/train_data/data.Lswitch_2'

H_switch_data_train_last='/home/xianyu/software/openrainbow/xm_data/All_data/train_data/data.Hswitch_last'
L_switch_data_train_last='/home/xianyu/software/openrainbow/xm_data/All_data/train_data/data.Lswitch_last'


normal_data_test='/home/xianyu/software/openrainbow/xm_data/All_data/test_data/data.normal_4'
H_controller_data_test='/home/xianyu/software/openrainbow/xm_data/All_data/test_data/data.Hcontroller'
L_controller_data_test='/home/xianyu/software/openrainbow/xm_data/All_data/test_data/data.Lcontroller'
H_switch_data_test='/home/xianyu/software/openrainbow/xm_data/All_data/test_data/data.Hswitch'
H_switch_data_test_1='/home/xianyu/software/openrainbow/xm_data/All_data/test_data/data.Hswitch_1'
L_switch_data_test='/home/xianyu/software/openrainbow/xm_data/All_data/test_data/data.Lswitch'
L_switch_data_test_1='/home/xianyu/software/openrainbow/xm_data/All_data/test_data/data.Lswitch_1'
#L_switch_data_test_2='/home/xianyu/software/openrainbow/xm_data/All_data/test_data/data.Lswitch_2'

H_switch_data_test_last='/home/xianyu/software/openrainbow/xm_data/All_data/test_data/data.Hswitch_last'
L_switch_data_test_last='/home/xianyu/software/openrainbow/xm_data/All_data/train_data/data.Lswitch_last'

test_path='/home/xianyu/software/openrainbow/xm_data/All_data/test_data/test.test'
victimer=['10.0.0.41', '10.0.0.51', '10.0.0.60']

def processing_for_test():

    print ("-----normal----label:0-")
    normal_dataset_train = attain_feature_base_data(normal_data_train, 0)
    normal_dataset_test = attain_feature_base_data(normal_data_test, 0)
    normal_dataset_train = normal_dataset_train[:splite1]
    normal_dataset_test = normal_dataset_test[:splite2]
    print('train num: '+str(normal_dataset_train.shape) + 'test num: '+str(normal_dataset_test.shape))

    print ("------Hcontroller----label:1")
    H_controller_train = attain_feature_base_data(H_controller_data_train, 1)
    H_controller_test = attain_feature_base_data(H_controller_data_test, 1)
    H_controller_train = H_controller_train[:splite1]
    H_controller_test = H_controller_test[:splite2]
    print('train num: '+str(H_controller_train.shape) + 'test num: '+str(H_controller_test.shape))

    print ("------Lcontroller----label:2")
    L_controller_train = attain_feature_base_data(L_controller_data_train, 2)
    L_controller_test = attain_feature_base_data(L_controller_data_test, 2)
    L_controller_train = L_controller_train[:splite1]
    L_controller_test = L_controller_test[:splite2]
    print('train num: '+str(L_controller_train.shape) + 'test num: '+str(L_controller_test.shape))

    print ("-----Hswitch------label:3-")
    H_switch_train = attain_feature_base_data(H_switch_data_train_last, 3)
    H_switch_train_1 = attain_feature_base_data(H_switch_data_train_1, 3)
    H_switch_test = attain_feature_base_data(H_switch_data_test_last, 3)
    H_switch_test_1 = attain_feature_base_data(H_switch_data_test_1, 3)
    H_switch_train = np.vstack((H_switch_train, H_switch_train_1))
    H_switch_test = np.vstack((H_switch_test, H_switch_test_1))

    H_switch_train = H_switch_train[:splite1]
    H_switch_test = H_switch_test[:splite2]
    print('train num: '+str(H_switch_train.shape) + 'test num: '+str(H_switch_test.shape))

    print ("-----Lswitch------label:4-")
    L_switch_train = attain_feature_base_data(L_switch_data_train_last, 4)
    L_switch_train_1 = attain_feature_base_data(L_switch_data_train_1, 4)
    #L_switch_train_2 = attain_feature_base_data(L_switch_data_train_2, 4)
    L_switch_test = attain_feature_base_data(L_switch_data_test_last, 4)
    L_switch_test_1 = attain_feature_base_data(L_switch_data_test_1, 4)
    #L_switch_test_2 = attain_feature_base_data(L_switch_data_test_2, 4)
    #L_switch_test_2 = attain_feature_base_data(L_switch_data_test_2, 4)
    L_switch_train =np.vstack((L_switch_train, L_switch_train_1))
    L_switch_test = np.vstack((L_switch_test, L_switch_test_1))

    L_switch_train = L_switch_train[:splite1]
    L_switch_test = L_switch_test[:splite2]
    print('train num: '+str(L_switch_train.shape) + 'test num: '+str(L_switch_test.shape))

    dataSetTrain = np.vstack((normal_dataset_train, H_controller_train, L_controller_train, H_switch_train, L_switch_train))
    dataSetTest = np.vstack((normal_dataset_test, H_controller_test, L_controller_test, H_switch_test, L_switch_test))

    np.savetxt('/home/xianyu/software/openrainbow/xm_data/All_data/train_data/dataSetTrain.txt', dataSetTrain, fmt='%s')
    np.savetxt('/home/xianyu/software/openrainbow/xm_data/All_data/test_data/dataSetTest.txt', dataSetTest, fmt='%s')



# 将ip：192.168.2.1切分为 19216821
def ip_split(ip):
    ip_str = str(ip).split('.')
    ip = ''.join(itertools.chain(*ip_str))
    return ip
    pass


def attain_feature_base_data(read_path, label):

    # 读取文件，data为字典类型
    data = json.load(open(read_path))
    # 储存数据集dataset,最后一列为label
    dataset = []
    for match in data.keys():
        # 删除两个不需要的特征
        #data[match].pop('eth_dst')
        data[match].pop('time')

        data[match]['ipv4_src'] = ip_split(data[match]['ipv4_src'])
        data[match]['ipv4_dst'] = ip_split(data[match]['ipv4_dst'])
        data[match]['arp_spa'] = ip_split(data[match]['arp_spa'])
        data[match]['arp_tpa'] = ip_split(data[match]['arp_tpa'])

        # 本次实验方式是不同的攻击，正常流量分开收集，一个文件中只有一种类型的数据，直接给标签赋值即可
        # data[match]['label'] = label

        _item = []
        for item in data[match]:
            _item.append(data[match][item])

        # 最后一列是样本标签
        _item.append(label)
        #print _item
        dataset.append(_item)

    # 格式化为array
    dataset = np.array(dataset)
    print(dataset)
    return dataset



processing_for_test()


