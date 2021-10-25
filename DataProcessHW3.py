import numpy as np
import pandas as pd
import os
from tqdm import tqdm
import math, time
from multiprocessing import Process, Pool, cpu_count, Manager

# 记录每句话中词的字典列表，每个字典value记录该句话中出现该词的次数
word_table = []
# 停用词表
stop_word_table = [];
# 是否去掉停用词
flag_remove_stop_word = False
# 语料库，记录整篇文章的全局词典，value记录出现该词的文档数
corpus = {}
# 文档平均长度
avdl = 0

global res_mat

# print(len(word_table))
# print(len(corpus))
# avdl = avdl/len(word_table)
# print(avdl)

def cal_tf_idf(word_dic):
    tf_idf_res_list = []
    dic_len = len(word_dic)
    corpus_len = len(corpus)
    for word in word_dic:
        tf = word_dic[word] / dic_len
        idf = math.log(corpus_len/(corpus[word] + 1))
        tf_idf_res_list.append((word, tf * idf))
    return tf_idf_res_list

def cos_sim(vector_list1, vector_list2):
    vector_list1 = np.mat(vector_list1)
    vector_list2 = np.mat(vector_list2)
    num = float(vector_list1 * vector_list2.T)
    denom = np.linalg.norm(vector_list1) * np.linalg.norm(vector_list2)
    #避免分母为0
    if denom == 0:
        denom = 1
    return num / denom


def cal_cos_sim(word_list1, word_list2):
    word_dic = {}
    word_dic1 = {}
    word_dic2 = {}
    for word in word_list1:
        word_dic[word[0]] = 1
        word_dic1[word[0]] = word[1]
    for word in word_list2:
        word_dic[word[0]] = 1
        word_dic2[word[0]] = word[1]
    #如果两个字典没有重复的关键词，则直接返回相似度为0
    if len(word_dic1) + len(word_dic2) == len(word_dic):
        return float(0)
    vector_list1 = [word_dic1[word] if word in word_dic1 else 0 for word in word_dic]
    vector_list2 = [word_dic2[word] if word in word_dic2 else 0 for word in word_dic]
    return cos_sim(vector_list1, vector_list2)


def work_process(start_index, end_index, word_tf_idf_table, table_len, m_dic):
    print("process:",start_index, end_index, table_len)
    # global res_mat
    for i in tqdm(range(start_index, end_index)):
        vector = np.zeros(table_len)
        for j in range(i, table_len):
            # print(i, j)
            sim = cal_cos_sim(word_tf_idf_table[i],word_tf_idf_table[j])
            vector[j] = sim
        m_dic[i] = vector


if __name__ == "__main__":
    global res_mat
    with open('E:\\软微\\课程\\研一上\\海量数据处理\\199801_clear.txt', 'r', encoding='gbk') as fr:
        lines = fr.readlines()
        sentence = ""
        for line in tqdm(lines):
            # 按空行分文章
            if line != "\n":
                sentence += line
                continue
            ori_list = sentence.split(" ")
            # 记录该句话中word的dic，dic的key为word字符串，value为该词在这句话中出现的次数
            word_dic = {}
            #
            for label_word in ori_list[1:]:
                # 去除空白符和空字符串
                if label_word.strip() != "":
                    # 如果在停用词表中
                    if label_word in stop_word_table:
                        continue
                    label = label_word.split('/')[-1]
                    word = label_word.split('/')[0]
                    # 添加词语到停用词表中
                    if label == 'u' or label == 'w' or label == 'c' or label == 'r' or label == 'd' or label == 'p' or label == 'k':
                        stop_word_table.append(label_word)
                    # 如果不在停用词表，则加入word_list，并更新全局词典
                    if label_word not in stop_word_table:
                        # 如果当前词不在全局字典中，则加入全局字典
                        if label_word not in corpus:
                            corpus[label_word] = 1
                        # 如果当前词在全局字典中
                        else:
                            # 判断在该句话中是否出现过，如果没出现过，则全局字典记录的文档数量+1
                            if label_word not in word_dic:
                                corpus[label_word] += 1
                        # 如果当前词在该句话的word_dic中没有出现过：
                        if label_word not in word_dic:
                            word_dic[label_word] = 1
                        else:
                            word_dic[label_word] += 1
            avdl += len(word_dic)
            word_table.append(word_dic)
            sentence = ""


    # 计算tf-idf频率向量，并排序筛选出前k个重要词，不足k个则全部保留
    k = 50
    word_tf_idf_table = []
    cnt = 0
    for word_dic in word_table:
        tf_idf_res_list = cal_tf_idf(word_dic)
        #     print(tf_idf_res_list)
        sort_res = sorted(tf_idf_res_list, key=lambda x: (x[1], x[0]), reverse=True)
        if len(sort_res) > k:
            word_tf_idf_table.append(sort_res[:k])
        else:
            word_tf_idf_table.append(sort_res)
    # print(word_tf_idf_table[:10])

    # 两两计算余弦相似度
    table_len = len(word_tf_idf_table)
    # table_len = 100
    # 存储相似度矩阵（上三角矩阵）
    res_mat = np.zeros((table_len, table_len))

    print("start")
    # cpus = cpu_count()
    pool = Pool(8)
    start = time.time() # 计算开始时间
    m_dict = Manager().dict()
    for i in range(8):
        pool.apply_async(func=work_process, args=(int(i*table_len/8), int((i+1)*table_len/8), word_tf_idf_table, table_len, m_dict))
    print("yes")
    pool.close()
    pool.join()
    print('计算时间为：',time.time() - start)
    for i in tqdm(range(table_len)):
        res_mat[i,:] = m_dict[i]
    # work_process(0, table_len, word_tf_idf_table, table_len)
    print(res_mat)


    ## 测试单进程
    # for i in tqdm(range(table_len)):
    #     # vector = np.zeros(table_len)
    #     for j in range(i, table_len):
    #         # print(i, j)
    #         res_mat[i][j] = cal_cos_sim(word_tf_idf_table[i],word_tf_idf_table[j])
    # print('计算时间为：', time.time() - start)
    # print(res_mat)

