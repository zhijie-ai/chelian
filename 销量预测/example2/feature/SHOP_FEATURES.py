#----------------------------------------------
# -*- encoding=utf-8 -*-                      #
# __author__:'焉知飞鱼'                        #
# CreateTime:                                 #
#       2019/12/8 22:16                       #
#                                             #
#               天下风云出我辈，                 #
#               一入江湖岁月催。                 #
#               皇图霸业谈笑中，                 #
#               不胜人生一场醉。                 #
#----------------------------------------------

from ..TOOLS.IJCAI2017_TOOL import *

'''
生成商家特征表SHOP_FEATURES.csv，包含平均View/Pay比值，平均每天开店时间，关店时间，
    开店总时长；首次营业日期，非节假日销量中位数，节假日销量中位数，节假日/非节假日销量比值；
    商家类别，人均消费，评分，评论数，门店等级。
'''

# shop_id,city_name,location_id,per_pay,score,comment_cnt,shop_level,cate_1_name,cate_2_name,cate_3_name
#  商家id(000001) ,市名(北京),所在位置编号，位置接近的商家具有相同的编号(001), 人均消费（数值越大消费越高）(3) , 评分（数值越大评分越高）(1) ,
#    评论数（数值越大评论数越多）(2) 门店等级（数值越大门店等级越高） (1),一级品类名称(美食),二级品类名称(小吃),三级品类名称(其他小吃)
SHOP_INFO_EN = pd.read_csv('../data_new/SHOP_INFO_EN.csv')

# %% category infomation
#商店类别
category = pd.read_csv('../data_new/SHOP_CAT.csv')
one_hot = pd.get_dummies(category['CAT'])
category = category.join(one_hot)
SHOP_SJ = map(lambda x: 'SJ' + str(x).zfill(2), np.arange(15))
category.columns = ['SHOP_CA1_EN', 'SHOP_CA2_EN', 'SHOP_CA3_EN', 'Num', 'CAT'] + SHOP_SJ
SHOP_INFO_EN = pd.merge(SHOP_INFO_EN, category, on=['SHOP_CA1_EN', 'SHOP_CA2_EN', 'SHOP_CA3_EN'], how='left')

# %%


SHOP_LOC_N = SHOP_INFO_EN.groupby('SHOP_LOC', as_index=False).count().loc[:, ['SHOP_LOC', 'SHOP_ID']].rename(
    columns={'SHOP_ID': 'SHOP_LOC_N'})
SHOP_INFO_EN = pd.merge(SHOP_INFO_EN, SHOP_LOC_N, on=['SHOP_LOC'], how='left')

SHOP_INFO_EN['SHOP_SCO'] = SHOP_INFO_EN['SHOP_SCO'].fillna(SHOP_INFO_EN['SHOP_SCO'].mean())
SHOP_INFO_EN['SHOP_COM'] = SHOP_INFO_EN['SHOP_COM'].fillna(SHOP_INFO_EN['SHOP_COM'].mean())
SHOP_INFO_EN = SHOP_INFO_EN[
    ['SHOP_ID', 'CITY_EN', 'SHOP_PAY', 'SHOP_SCO', 'SHOP_COM', 'SHOP_LEV', 'SHOP_LOC_N'] + SHOP_SJ]
SHOP_SD = map(lambda x: 'SD' + str(x).zfill(2), np.arange(5))
SHOP_INFO_EN.columns = ['SHOP_ID', 'CITY_EN'] + SHOP_SD + SHOP_SJ

# [shop_id,day,hour,Num_post]
PAYNW = pd.read_csv('../data_new/user_pay_new.csv')#每个商店按天和小时分组支付的总次数，再加一个dayofweek
VIENW = pd.read_csv('../data_new/user_view_new.csv')
# %%
HOLI = pd.read_csv('../additional/HOLI.csv')
HOLI['DATE'] = [(lambda x: str(datetime.datetime.strptime(str(x), '%Y%m%d').date()))(x) for x in HOLI['DATE']]

# %%  calculate top hours   SE,SF
# 支付次数及比重
TOP_N = 1
SHOP_ID = []
SHOP_HOUR_head = []
SHOP_PCT_head = []
for SHOP_IND in range(1, 2001):#SHOP_ID,DATE,HOUR,Num_raw,Num_post,DofW
    tt = PAYNW[PAYNW['SHOP_ID'] == SHOP_IND]
    #算出数据集里每家店按小时分组的支付次数
    tt2 = tt.groupby('HOUR', as_index=False).sum()#算出每家店按小时分组的购买次数(1人买了2次算2次)
    tt3 = tt2.sort_values('Num_post', ascending=False, inplace=False)#降序排序
    tt4 = tt3.head(TOP_N)['HOUR'].values#数据中当前店数据出现的小时(最多24)
    tt5 = tt3.head(TOP_N)['Num_post'].values / tt3['Num_post'].sum()#top1对应的小时的购买次数/数据中的总次数
    SHOP_ID.append(SHOP_IND)
    SHOP_HOUR_head.append(tt4)
    SHOP_PCT_head.append(tt5)
SHOP_ID_df = pd.DataFrame(SHOP_ID)
SHOP_HOUR_head_df = pd.DataFrame(SHOP_HOUR_head)
SHOP_PCT_head_df = pd.DataFrame(SHOP_PCT_head)

#[shop_id,小时最高的购买次数,最高/总的次数]
SELL_INFO = pd.concat([SHOP_ID_df, SHOP_HOUR_head_df, SHOP_PCT_head_df], axis=1)
SHOP_SE = [(lambda x: ('SE' + str(x).zfill(2)))(x) for x in range(TOP_N)]
SHOP_SF = [(lambda x: ('SF' + str(x).zfill(2)))(x) for x in range(TOP_N)]
SELL_INFO.columns = ['SHOP_ID'] + SHOP_SE + SHOP_SF

# %%  calculate top hours
# [shop_id,day,hour,Num_post]
SHOP_ID = []
SHOP_OPEN = []#平均开店时间(第一笔支付算开店时间)
SHOP_CLOSE = []#平均关店时间(最后一笔支付算关店时间)
SHOP_LAST = []
SHOP_MEAN = []
for SHOP_IND in range(1, 2001):
    tt = PAYNW[PAYNW['SHOP_ID'] == SHOP_IND]
    tt2 = tt.groupby('DATE', as_index=False).min().mean()#
    tt3 = tt.groupby('DATE', as_index=False).max().mean()
    tt['MEAN'] = tt['Num_post'] * tt['HOUR']
    SHOP_ID.append(SHOP_IND)
    SHOP_OPEN.append(tt2.HOUR)
    SHOP_CLOSE.append(tt3.HOUR)
    SHOP_LAST.append(tt3.HOUR - tt2.HOUR)
    SHOP_MEAN.append(tt['MEAN'].sum() / tt['Num_post'].sum())
SHOP_ID_df = pd.DataFrame(SHOP_ID)
SHOP_OPEN_df = pd.DataFrame(SHOP_OPEN)
SHOP_CLOSE_df = pd.DataFrame(SHOP_CLOSE)
SHOP_LAST_df = pd.DataFrame(SHOP_LAST)
SHOP_MEAN_df = pd.DataFrame(SHOP_MEAN)
HOUR_INFO = pd.concat([SHOP_ID_df, SHOP_OPEN_df, SHOP_CLOSE_df, SHOP_LAST_df, SHOP_MEAN_df], axis=1)
SHOP_SG = map(lambda x: 'SG' + str(x).zfill(2), np.arange(4))
HOUR_INFO.columns = ['SHOP_ID'] + SHOP_SG

# %%  need to fillna 0
# SHOP_ID,DATE,HOUR,Num_raw,Num_post,DofW
PAYNW_gp = PAYNW.groupby(['DATE', 'SHOP_ID'], as_index=False).sum()
VIENW_gp = VIENW.groupby(['DATE', 'SHOP_ID'], as_index=False).sum()
PAYNW_gp['DofW'] = Datestr2DofW(PAYNW_gp['DATE'])
VIENW_gp['DofW'] = Datestr2DofW(VIENW_gp['DATE'])
PAYNW_gp = pd.merge(PAYNW_gp, HOLI, on=['DATE'], how='left')
VIENW_gp = pd.merge(VIENW_gp, HOLI, on=['DATE'], how='left')

PAYNW_VIENW = pd.merge(PAYNW_gp, VIENW_gp, on=['DATE', 'SHOP_ID'], how='inner')
#Pay/View,每天的比例
PAYNW_VIENW['RATIO'] = PAYNW_VIENW['Num_post_y'] / PAYNW_VIENW['Num_post_x']
SHOP_PAYNW_VIENW = PAYNW_VIENW.groupby('SHOP_ID', as_index=False).mean()
SHOP_SC = 'SC00'  ### view/pay ratio
RATIO_INFO = SHOP_PAYNW_VIENW[['SHOP_ID', 'RATIO']].rename(columns={'RATIO': SHOP_SC})

# %%  first online date,  and online days to count
# 开店天数[shop_id,date,rat_day]
GAP_INFO = PAYNW.groupby('SHOP_ID', as_index=False).min().loc[:, ['SHOP_ID', 'DATE']]
GAP_INFO['RAT_DAY'] = pd.to_datetime(GAP_INFO['DATE']) - datetime.date(2015, 6, 26)
GAP_INFO['RAT_DAY'] = [(lambda x: (x.days))(x) for x in GAP_INFO['RAT_DAY']]
SHOP_SH = [(lambda x: ('SH' + str(x).zfill(2)))(x) for x in range(2)]
GAP_INFO.columns = ['SHOP_ID'] + SHOP_SH

# %%  ratio of weekday and weekend median


PAYNW_gp_wd = PAYNW_gp[PAYNW_gp['HOLI'] == 0].groupby('SHOP_ID', as_index=False).median()
PAYNW_gp_wd = PAYNW_gp_wd[['SHOP_ID', 'Num_post']].rename(columns={'Num_post': 'WD_MED'})
PAYNW_gp_wk = PAYNW_gp[PAYNW_gp['HOLI'] > 0].groupby('SHOP_ID', as_index=False).median()
PAYNW_gp_wk = PAYNW_gp_wk[['SHOP_ID', 'Num_post']].rename(columns={'Num_post': 'WK_MED'})
PAYNW_gp_wdwk = pd.merge(PAYNW_gp_wd, PAYNW_gp_wk, on='SHOP_ID', how='left')

# %%
HOLI_list = [0, 0, 0, 0, 0, 1, 1]
DofW_list = [0, 1, 2, 3, 4, 5, 6]

for holi_ind, DofW_ind in zip(HOLI_list, DofW_list):
    DAYOFWEEk = PAYNW_gp[(PAYNW_gp['HOLI'] == holi_ind) & (PAYNW_gp['DofW'] == DofW_ind)].groupby('SHOP_ID',
                                                                                                  as_index=False).median()
    DAYOFWEEk = DAYOFWEEk[['SHOP_ID', 'Num_post']].rename(columns={'Num_post': 'd' + str(DofW_ind)})
    PAYNW_gp_wdwk = pd.merge(PAYNW_gp_wdwk, DAYOFWEEk, on='SHOP_ID', how='left')

SHOP_SI = [(lambda x: ('SI' + str(x).zfill(2)))(x) for x in range(9)]
PAYNW_gp_wdwk.columns = ['SHOP_ID'] + SHOP_SI
PAYNW_gp_wdwk['FIX'] = PAYNW_gp_wdwk[SHOP_SI].mean(axis=1)
PAYNW_gp_wdwk[SHOP_SI] = PAYNW_gp_wdwk[SHOP_SI].div(PAYNW_gp_wdwk['FIX'], axis=0)
del PAYNW_gp_wdwk['FIX']
PAYNW_gp_wdwk['ratio'] = PAYNW_gp_wdwk['SI00'] / PAYNW_gp_wdwk['SI01']
SHOP_SI = [(lambda x: ('SI' + str(x).zfill(2)))(x) for x in range(10)]
PAYNW_gp_wdwk.columns = ['SHOP_ID'] + SHOP_SI

# %%
SHOP_INFO_EN = pd.merge(SHOP_INFO_EN, SELL_INFO, on='SHOP_ID', how='left')  # SA
SHOP_INFO_EN = pd.merge(SHOP_INFO_EN, HOUR_INFO, on='SHOP_ID', how='left')  # SE,SF
SHOP_INFO_EN = pd.merge(SHOP_INFO_EN, RATIO_INFO, on='SHOP_ID', how='left')  # SC
SHOP_INFO_EN = pd.merge(SHOP_INFO_EN, GAP_INFO, on='SHOP_ID', how='left')  # SH
SHOP_INFO_EN = pd.merge(SHOP_INFO_EN, PAYNW_gp_wdwk, on='SHOP_ID', how='left')  # SI
SHOP_INFO_EN = SHOP_INFO_EN.fillna(0)

# %%
SHOP_INFO_EN.to_csv('SHOP_FEATURES.csv', index=False)