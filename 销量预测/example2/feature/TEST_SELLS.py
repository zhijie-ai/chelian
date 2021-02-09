#----------------------------------------------
# -*- encoding=utf-8 -*-                      #
# __author__:'焉知飞鱼'                        #
# CreateTime:                                 #
#       2019/12/8 22:17                       #
#                                             #
#               天下风云出我辈，                 #
#               一入江湖岁月催。                 #
#               皇图霸业谈笑中，                 #
#               不胜人生一场醉。                 #
#----------------------------------------------
# 异常值修复，他这生成测试集时，把所有的date变成column了
# 生成测试集历史过去三周销量表格，修正异常销量，以历史过去14天销量的μ±2σ 为限制，其中μ为均值，σ为均方根
from example2.TOOLS.IJCAI2017_TOOL import *

PAYNW = pd.read_csv('../data_new/user_pay_new.csv')

# index为shop_id，column为date
PAYNW_TAB = pd.pivot_table(PAYNW, values=['Num_post'], index=['SHOP_ID'], columns=['DATE'], aggfunc=np.sum)
PAYNW_TAB = pd.concat([PAYNW_TAB[PAYNW_TAB.columns[0:169:1]], pd.DataFrame({'A': [np.nan], }, index=np.arange(1, 2001)),
                       PAYNW_TAB[PAYNW_TAB.columns[169::1]]], axis=1)
PAYNW_TAB.columns = [str((datetime.datetime.strptime('20150626', '%Y%m%d') + datetime.timedelta(days=x)).date()) for x
                     in range(PAYNW_TAB.shape[1])]
inspect_cols = [str((datetime.datetime.strptime('20161009', '%Y%m%d') + datetime.timedelta(days=x)).date()) for x in
                range(23)]
print(PAYNW_TAB.head())
#以历史3周的销量数据
PAYNW_TAB_OCT = PAYNW_TAB.loc[:, '2016-10-09':'2016-10-31']
PAYNW_TAB_OCT.reset_index(level=0, inplace=True)
print(PAYNW_TAB_OCT.head())

SHOP_MELT = pd.melt(PAYNW_TAB_OCT, id_vars=['SHOP_ID'], value_vars=inspect_cols)
SHOP_MELT = SHOP_MELT.rename(columns={'variable': 'DATE'})

# %% find all the shops with small value, substitude with the mininum value of day of week in 4 weeks
# 以销量少的为目标，取前后2周的数据,销量少，次数也少的shop
SMALL_SHOP = SHOP_MELT.loc[SHOP_MELT['value'] <= 10, :]# value即为Num_post
SMALL_SHOP = SMALL_SHOP.sort_values(by=['SHOP_ID', 'DATE'])
SMALL_count = SMALL_SHOP.groupby(by=['SHOP_ID'], as_index=False).count()
SMALL_count = SMALL_count[SMALL_count['DATE'] <= 2]#每个商铺的支付的时间跨度不大于2天
SMALL_SHOP = SMALL_SHOP[SMALL_SHOP.SHOP_ID.isin(SMALL_count.SHOP_ID)]# 2个条件
print('AAAA',SMALL_SHOP)
SMALL_SHOP.index = np.arange(len(SMALL_SHOP))
Substitude_list = []
for ind, value in SMALL_SHOP.iterrows():
    SHOP_ID = value.SHOP_ID
    DATE = value.DATE
    Day_shift_list = [-14, -7, 7, 14]
    Shop_sub_list = []
    for shift_ind in Day_shift_list:
        DATE_shift = str((datetime.datetime.strptime(DATE, '%Y-%m-%d') +
                          datetime.timedelta(days=shift_ind)).date())
        try:
            value_append = PAYNW_TAB.loc[SHOP_ID, DATE_shift]
        except:
            value_append = np.nan
        Shop_sub_list.append(value_append)
    Substitude_list.append(Shop_sub_list)

# 先取出销量很少的shop_id，然后以每条数据为参考，取出前后2周的销量数据
Substitude_list = pd.DataFrame(Substitude_list)
SMALL_SHOP = pd.concat([SMALL_SHOP, Substitude_list], axis=1)
SMALL_SHOP['Num_post'] = SMALL_SHOP[np.arange(4)].min(axis=1)

for ind, value in SMALL_SHOP.iterrows():
    SHOP_ID = value.SHOP_ID
    DATE = value.DATE
    PAYNW_TAB.loc[SHOP_ID, DATE] = value.Num_post

# %% substitude with fill oct


PAYNW_TAB_FIX = pd.read_csv('../data_new/FillOct.csv')
PAYNW_TAB_FIX['DATE'] = [(lambda x: str(datetime.datetime.strptime(x, '%Y/%m/%d').date()))(x) for x in
                         PAYNW_TAB_FIX['DATE']]

for ind, value in PAYNW_TAB_FIX.iterrows():
    SHOP_ID = value.SHOP_ID
    DATE = value.DATE
    PAYNW_TAB.loc[SHOP_ID, DATE] = value.Num_post

TRN_N = 21
TST_N = 14

TEST = pd.DataFrame()
TRN_END = datetime.datetime.strptime('2016-10-31', '%Y-%m-%d')
TRN_STA = (TRN_END - datetime.timedelta(days=(TRN_N - 1)))#三周的数据
TST_STA = (TRN_END + datetime.timedelta(days=(1)))
TST_END = (TRN_END + datetime.timedelta(days=(TST_N)))#2周作为测试
test_date_zip = zip([str(TRN_STA.date())], [str(TRN_END.date())], [str(TST_STA.date())], [str(TST_END.date())])
TEST = PAYNW_TAB.loc[:, str(TRN_STA.date()):str(TRN_END.date())]
TEST.reset_index(level=0, inplace=True)
end_date = datetime.datetime.strptime('2016-10-31', '%Y-%m-%d')
TEST.loc[:, 'TRN_STA'] = str(TRN_STA.date())
TEST.loc[:, 'TRN_END'] = str(TRN_END.date())
TEST.loc[:, 'TST_STA'] = str(TST_STA.date())
TEST.loc[:, 'TST_END'] = str(TST_END.date())
TEST_TRN_C = map(lambda x: 'SA' + str(x).zfill(2), np.arange(TRN_N))
TEST.columns = ['SHOP_ID'] + TEST_TRN_C + ['TRN_STA', 'TRN_END', 'TST_STA', 'TST_END']

# %%

TEST.to_csv('TEST_SELLS.csv', index=False)