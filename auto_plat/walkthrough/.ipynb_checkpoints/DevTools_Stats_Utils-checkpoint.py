'''
Author: liulinwei703 liulinwei703@hellobike.com
Date: 2024-06-04 14:22:43
LastEditors: liulinwei703 liulinwei703@hellobike.com
LastEditTime: 2024-06-21 14:23:00
FilePath: /Desktop/DataScience/假设检验工具箱/Stats_Utils.py
'''
from scipy import stats
import numpy as np
from scipy.stats import norm
from scipy.stats import f
from scipy.stats import t
import statsmodels.stats.proportion as proportion


def SRMtest(observed_counts,expected_ratios):
    '''
    description: 用于SRM检验
    input: observed_counts list类型,为观测到的样本量;expected_ratios list类型,为期望的比例
    return (SRM检验统计量,p-val)
    '''    
    observed_counts = np.array(observed_counts) # 实际每组分流人数
    expected_ratios = np.array(expected_ratios)  # 设定的分流比例
    n_total = np.sum(observed_counts) # 实际分流总数
    expected_counts = expected_ratios * n_total # 根据设定比例计算得到的期望每组分流人数
    statistic, pval = stats.chisquare(observed_counts, expected_counts) # 卡方检验
    print("统计量：",round(statistic,3),"\n","pvalue:",round(pval,3),"\n","分配比例出现问题" if pval < 0.05 else "分配比例没有出现问题")
    return round(pval,3), round(statistic,3), 


def Mean_Value_Z_test(mean1, std1, n1, mean2, std2, n2):
    '''
    description: 用于两个均值类型的样本差异Z检验
    input: 样本1的均值,样本1的标准差,样本1的样本量,样本2的均值,样本2的标准差,样本2的均值
    return Z检验的pvalue,Z检验的统计量,置信区间
    '''
        # 计算Z值
    z_score = (mean1 - mean2) / np.sqrt((std1**2 / n1) + (std2**2 / n2))

    # 计算p值
    p_value = 2 * (1 - norm.cdf(np.abs(z_score)))
    
    se =np.sqrt((std1**2 / n1) + (std2**2 / n2))
    ci=[mean1-mean2-1.96*se,mean1-mean2+1.96*se] #1.96为正态分布的0.975分位数

    return p_value, z_score,ci


def Proportion_Z_test(p1, n1, p2, n2):
    '''
    description: 用于两个比例类型的样本差异Z检验
    input: 样本1的比例,样本1的样本量,样本2的比例,样本2的均值
    return Z检验的pvalue,Z检验的统计量
    '''
    z_score, p_value = proportion.proportions_ztest([int(n1*p1), int(n2*p2)], [n1, n2])
    se = (p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2) ** 0.5
    ci=[p1-p2-1.96*se,p1-p2+1.96*se] #1.96为正态分布的0.975分位数

    return p_value, z_score ,ci



def F_test(metric_mean_list, metric_std_list, mertric_cnt_list, center_mean):
    '''
    description: 用于多组样本均值是否相同的检验
    input: 每组样本的均值list, 每组样本的标准差list, 每组样本的样本量list, 总体的均值
    return F检验的pvalue, F检验的统计量
    '''    
    k = len(metric_mean_list) 
    n = sum(mertric_cnt_list)
    grand_mean = center_mean #总均值
    
    group_mean = np.array(metric_mean_list) #各组均值    
    group_std = np.array(metric_std_list) #各组标准差
    ssb = sum(mertric_cnt_list*(np.array(group_mean)-np.array([grand_mean]*k))**2) #组间平方和
    ssw = sum((mertric_cnt_list-1)*(group_std**2)) #组内平方和
    f_stat = (ssb/(k-1))/(ssw/(n-k)) # f统计量
    f_pval_1 =1 - f.cdf(f_stat, k-1, n-k)
    
    return  f_pval_1, f_stat




def M_Distance(mertic1_mean_list, mertic2_mean_list, center_std_list, center_mean_list, cov_xy):
    '''
    description: 用于多个样本点计算到总体中心的马氏距离（目前仅支持两个维度上计算，即支持两个指标）
    input: 每组样本的指标1的均值list, 每组样本的指标2的均值list, 总体上指标的标准差list[指标1的标准差, 指标2的标准差], 总体上指标的均值list[指标1的均值, 指标2的均值], 总体上两个指标的协方差
    return 多个样本点到总体中心的马氏距离之和
    '''    
    std_x = center_std_list[0]
    std_y = center_std_list[1]
    cov_xy = cov_xy
    cov_matrix = np.array([[std_x**2, cov_xy],
                            [cov_xy, std_y**2]])
    inv_cov_matrix = np.linalg.inv(cov_matrix) # 计算协方差矩阵的逆矩阵|
    mahalanobis_distances = np.sum(np.sqrt(np.diag((np.vstack((mertic1_mean_list, mertic2_mean_list)).T- np.array(center_mean_list)) 
                                                   .dot(inv_cov_matrix)
                                                   .dot((np.vstack((mertic1_mean_list, mertic2_mean_list)).T- np.array(center_mean_list)).T))))
    return mahalanobis_distances

def Mean_Value_T_test(n1,n2,mu1,mu2,sigma1,sigma2,alternative=None):
    """
    description: 用于两个均值类型的样本差异T检验  
    input: 样本1的均值,样本1的标准差,样本1的样本量,样本2的均值,样本2的标准差,样本2的数量,拒绝域的方向
    return T检验的pvalue,T检验的统计量  
    """ 
    pooled_variance = ((n1 - 1) * sigma1**2 + (n2 - 1) * sigma2**2) / (n1 + n2 - 2)    
    se = np.sqrt(pooled_variance * (1/n1 + 1/n2))  
  
    t_statistic = (mu1 - mu2) / se  
  
    # 计算自由度（由于方差齐性，使用两个样本大小之和减去2）  
    df = n1 + n2 - 2  

    p_value = 2 * (1 - t.cdf(abs(t_statistic), df=df))   
    if alternative == 'right':
        p_value = 1-t.cdf(abs(t_statistic), df=df)
    if alternative == 'left':
        p_value = t.cdf(abs(t_statistic), df=df)

    return p_value,t_statistic

def Mean_Value_Z_power(mu1,mu2,sd1,sd2,n1,n2,alpha=0.5):
    """
    description: 用于计算两个均值样本的z检验统计功效  
    input: 样本1的均值,样本1的标准差,样本1的样本量,样本2的均值,样本2的标准差,样本2的数量
    return 统计功效
    """ 
    power = norm.cdf((mu1-mu2) / ((sd1**2 / n1 + sd2**2 / n2)**0.5) - norm.ppf(1 - alpha / 2))
    return power


def ratio_variance(mu_x, mu_y, sd_x, sd_y, cov_xy ,n):
    """
    description：利用delta方法,估计比例类指标 M = X / Y 的方差, 用于计算单组内两个指标的组成的比例指标
    input：x、y的均值、标准差、协方差以及样本量n
    output：指标方差
    """
    var = 1/n * (mu_x ** 2 / mu_y ** 2) * (sd_x ** 2 / mu_x ** 2 - 2 * cov_xy / (mu_x * mu_y) + sd_y ** 2 / mu_y ** 2 )
    
    return var





def Relative_Inc_Z_test(n1, n2, mean1, mean2, std1, std2, cov_xy, alternative=None):
    
    var_delta = (1/ (mean2 ** 2)) * (std1 ** 2 / n1) + ((mean1 ** 2)/(mean2 ** 4)) * (std2 ** 2 / n2) - 2 * (mean1) / (mean2 ** 3) * cov_xy
    relative_inc = (mean1 - mean2) / mean2
    se = np.sqrt(var_delta)
    z_score_delta = (relative_inc - 0) / se
    p_value_delta = 2 * (1 - norm.cdf(np.abs(z_score_delta)))
    ci=[relative_inc-1.96*se,relative_inc+1.96*se]

    return p_value_delta, z_score_delta, ci


def Ratio_Metric_Z_test(n1, n2, mean_1_x, mean_1_y, mean_2_x, mean_2_y, std_1_x, std_1_y, std_2_x, std_2_y, cov_1_xy, cov_2_xy):
    """
    description: 用于两个比率类型的样本差异Z检验
    input: 样本1的比例,样本1的样本量,样本2的比例,样本2的均值
    return Z检验的pvalue,Z检验的统计量
    """
    var_delta = ratio_variance(mean_1_x, mean_1_y, std_1_x, std_1_y, cov_1_xy, n1) + ratio_variance(mean_2_x, mean_2_y, std_2_x, std_2_y, cov_2_xy, n2)
    Ratio_Inc = ((mean_1_x/mean_1_y) - (mean_2_x/mean_2_y))
    z_score = Ratio_Inc / np.sqrt(var_delta) 
    p_value = 2 * (1 - norm.cdf(np.abs(z_score)))
    ci = [Ratio_Inc - 1.96 * np.sqrt(var_delta), Ratio_Inc + 1.96 * np.sqrt(var_delta)]
    
    return p_value, z_score, ci


