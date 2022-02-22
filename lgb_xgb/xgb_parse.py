# ----------------------------------------------
# -*- encoding=utf-8 -*-                      #
# __author__:'焉知飞鱼'                        #
# CreateTime:                                 #
#       2021/12/1 15:35                       #
#                                             #
#               天下风云出我辈，                 #
#               一入江湖岁月催。                 #
#               皇图霸业谈笑中，                 #
#               不胜人生一场醉。                 #
# ----------------------------------------------
import pandas as pd

def _xgb_tree_leaf_parse(xgbtree,nodeid_leaf):
    '''给定叶子节点，查找 xgbtree 树的路径    '''
    leaf_ind=list(nodeid_leaf)
    result=xgbtree.loc[(xgbtree.ID.isin(leaf_ind)),:]
    result['Tag']='Leaf'
    node_id=list(result.ID)
    while len(node_id)>0:
        tmp1=xgbtree.loc[(xgbtree.Yes.isin(node_id)),:]
        tmp2=xgbtree.loc[(xgbtree.No.isin(node_id)),:]
        tmp1['Tag']='Yes'
        tmp2['Tag']='No'
        node_id=list(tmp1.ID)+list(tmp2.ID)
        result=pd.concat([result,tmp1,tmp2],axis=0)
        return result


def xgb_parse(model, xgbtree, feature=None):
    '''给定模型和单个样本，返回该样本的xgbtree树路径以及该样本的特征重要度    '''
    feature_names=model.get_booster().feature_names
    #missing_value=model.get_params()['missing']
    f0=pd.DataFrame({'GainTotal':model.feature_importances_,'Feature':feature_names})
    f0=f0[['Feature','GainTotal']]
    xgbtree=model.get_booster().trees_to_dataframe()
    if feature is None:
        return xgbtree,f0.sort_values(by='GainTotal',ascending=False).reset_index(drop=True)
        ind=model.get_booster().predict(xgb.DMatrix(feature),validate_features=False,pred_leaf=True)[0]
        ind=pd.Series(np.arange(model.n_estimators)).astype(np.str)+'-'+pd.Series(ind).astype(np.str)
        result=_xgb_tree_leaf_parse(xgbtree,ind)
        result=result.sort_values(by=['Tree','Node'])
        loc=int(np.where(result.columns=='Feature')[0][0])+1
        result.insert(loc,'FeatureValue',result.Feature.replace(feature.to_dict()))
        #result.loc[(result.FeatureValue==missing_value)|(result.FeatureValue.isnull()),'Tag']='Missing'
        result=result[['Tree','Node','ID','Feature','FeatureValue','Split','Tag','Yes','No','Missing','Gain','Cover']]
        f_=result.groupby('Feature')['Gain'].mean().drop('Leaf',axis=0)
        f_=f_/f_.sum()
        f_=pd.DataFrame(f_).reset_index()
        f=pd.merge(f0[['Feature','GainTotal']],f_[['Feature','Gain']],on='Feature',how='left').fillna(0)
        f['diff']=np.round((f['Gain']-f['GainTotal'])/f['GainTotal'],2)
        f=f[['Feature','Gain','GainTotal','diff']].sort_values(by='Gain',ascending=False).reset_index(drop=True)
    return result,f

if __name__ == '__main__':
    import pickle
    model = pickle.load(open('xgb_model_param.pickle', 'rb'))
    xgbtree = model.get_booster().trees_to_dataframe()
