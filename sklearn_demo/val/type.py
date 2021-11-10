#----------------------------------------------
# -*- encoding=utf-8 -*-                      #
# __author__:'焉知飞鱼'                        #
# CreateTime:                                 #
#       2020/8/10 17:58                       #
#                                             #
#               天下风云出我辈，                 #
#               一入江湖岁月催。                 #
#               皇图霸业谈笑中，                 #
#               不胜人生一场醉。                 #
#----------------------------------------------
import numpy as np

TYPES = {
    'gender':np.int8,
    'state':np.int8,
    'bizuserdisplayclass':np.int8,
    'addsourcetype':np.int8,
    'isbiggoodsuser':np.int8,

    'isagencyuser':np.int8,
    'iscontractuser':np.int8,
    'isretailuser':np.int8,
    'is_staff':np.int8,
    'logindays_offset':np.float32,
    'createtime_offset':np.float32,
    'modifydays_offset':np.float32,

    'login_cnt_30':np.float32,
    'order_cnt_30':np.float32,
    'order_date_cnt_30':np.float32,

    'sum_sku_num_30':np.float32,
    'sum_total_amount_30':np.float32,
    'turnover_order_cnt_30':np.float32,

    'sum_turnover_amount_30':np.float32,
    'sum_turnover_num_30':np.float32,
    'visit_cnt_30':np.float32,

    'complaint_cnt_30':np.float32,
    'login_cnt_90':np.float32,
    'order_cnt_90':np.float32,
    'order_date_cnt_90':np.float32,

    'sum_sku_num_90':np.float32,
    'sum_total_amount_90':np.float32,
    'turnover_order_cnt_90':np.float32,

    'turnover_order_cnt_90.1':np.float32,
    'sum_turnover_num_90':np.float32,
    'visit_cnt_90':np.float32,

    'complaint_cnt_90':np.float32,


    'productsku_state':np.int8,
    'producttype':np.int8,
    'is_unpacksale':np.int8,
    'shop_type':np.int8,
    'firstonline_date':np.float32,

    'monthofshelf_life':np.float32,
    'productskuunit_costprice':np.float32,
    'spec_quantity':np.float32,

    'stat_order_num_15':np.float32,
    'stat_sum_sku_num_15':np.float32,
    'stat_sum_total_amount_15':np.float32,

    'stat_original_sum_total_amount_15':np.float32,
    'stat_turnover_order_num_15':np.float32,

    'stat_sum_turnover_num_15':np.float32,
    'stat_sum_turnover_amount_15':np.float32,

'stat_avg_actualsell_price_15':np.float32,
'stat_user_num_15':np.float32,

'stat_repurchase_rate_15':np.float32,
'stat_order_num_30':np.float32,
'stat_sum_sku_num_30':np.float32,

'stat_sum_total_amount_30':np.float32,
'stat_original_sum_total_amount_30':np.float32,

'stat_turnover_order_num_30':np.float32,
'stat_sum_turnover_num_30':np.float32,

'stat_sum_turnover_amount_30':np.float32,
'stat_avg_actualsell_price_30':np.float32,

'stat_user_num_30':np.float32,
'stat_repurchase_rate_30':np.float32,
'stat_order_num_90':np.float32,

'stat_sum_sku_num_90':np.float32,
'stat_sum_total_amount_90':np.float32,

'stat_original_sum_total_amount_90':np.float32,
'stat_turnover_order_num_90':np.float32,

'stat_sum_turnover_num_90':np.float32,
'stat_sum_turnover_amount_90':np.float32,

'stat_avg_actualsell_price_90':np.float32,
'stat_user_num_90':np.float32,

'stat_repurchase_rate_90':np.float32,
'in_specquantity':np.float32,
'in_order_num_15':np.float32,

'in_avg_bare_price_15':np.float32,
'in_sum_total_amount_15':np.float32,
'in_sum_sku_num_15':np.float32,

'in_sum_minuint_num_15':np.float32,
'in_avg_price_15':np.float32,
'in_order_num_30':np.float32,

'in_avg_bare_price_30':np.float32,
'in_sum_total_amount_30':np.float32,
'in_sum_sku_num_30':np.float32,

'in_sum_minuint_num_30':np.float32,
'in_avg_price_30':np.float32,
'in_order_num_90':np.float32,

'in_avg_bare_price_90':np.float32,
'in_sum_total_amount_90':np.float32,
'in_sum_sku_num_90':np.float32,

'in_sum_minuint_num_90':np.float32,
'in_avg_price_90':np.float32,
'out_specquantity':np.float32,

'out_order_num_15':np.float32,
'out_avg_minunit_price_15':np.float32,

'out_sum_total_amount_15':np.float32,
'out_sum_sku_num_15':np.float32,

'out_sum_minuint_num_15':np.float32,
'out_avg_sku_price_15':np.float32,

'out_avg_cost_price_15':np.float32,
'out_avg_sku_cost_price_15':np.float32,

'out_avg_profit_price_15':np.float32,
'out_avg_sku_profit_price_15':np.float32,
'out_order_num_30':np.float32,

'out_avg_minunit_price_30':np.float32,
'out_sum_total_amount_30':np.float32,

'out_sum_sku_num_30':np.float32,
'out_sum_minuint_num_30':np.float32,
'out_avg_sku_price_30':np.float32,

'out_avg_cost_price_30':np.float32,
'out_avg_sku_cost_price_30':np.float32,

'out_avg_profit_price_30':np.float32,
'out_avg_sku_profit_price_30':np.float32,
'out_order_num_90':np.float32,

'out_avg_minunit_price_90':np.float32,
'out_sum_total_amount_90':np.float32,

'out_sum_sku_num_90':np.float32,
'out_sum_minuint_num_90':np.float32,
'out_avg_sku_price_90':np.float32,

'out_avg_cost_price_90':np.float32,
'out_avg_sku_cost_price_90':np.float32,

'out_avg_profit_price_90':np.float32,
'out_avg_sku_profit_price_90':np.float32
}