# 导入Flask类
from flask import Flask
from flask import render_template,request
import pickle
import numpy as np


# 实例化，可视为固定格式
app = Flask(__name__)

poly_reg = pickle.load(open('./data/poly_reg.pickle','rb'))
Ridge_model = pickle.load(open('./data/ridge_model.pickle','rb'))

poly_reg__e2r = pickle.load(open('./data/poly_reg5_exp2rank.pickle','rb'))
Ridge_model_e2r = pickle.load(open('./data/ridge_model_exp2rank.pickle','rb'))

# route()方法用于设定路由；类似spring路由配置
#http://127.0.0.1:5000/rank2exp?rank=3
@app.route('/rank2exp')
def rank2exp():
    rank = request.args.get('q')
    rank = int(float(rank))
    if rank < 0 :
        return "请输入合法的排名"
    data = np.array([[rank]])
    exp = Ridge_model.predict(poly_reg.fit_transform(data))
    if exp < 0:
        exp = 0
    return str(int(exp[0]))

# http://127.0.0.1:5000/exp2rank?exp=8000
@app.route('/exp2rank')
def exp2rank():
    exp = request.args.get('q')
    exp = float(exp)
    if exp < 0.0 or exp > 10000.0:
        return "请输入在0-10000之间的指数"
    data = np.array([[exp]])
    rank = Ridge_model_e2r.predict(poly_reg__e2r.fit_transform(data))
    return str(int(rank[0]))

if __name__ == '__main__':
    # app.run(host, port, debug, options)
    # 默认值：host=127.0.0.1, port=5000, debug=false
    app.run(debug=False)
    # print(Ridge_model.predict(poly_reg.fit_transform([[600]])))
    # print(Ridge_model.predict(poly_reg.fit_transform([[590]])))
    # print(Ridge_model.predict(poly_reg.fit_transform([[620]])))
    # print('A'*20)
    # print(Ridge_model_e2r.predict(poly_reg__e2r.fit_transform([[0]])))
    # print(Ridge_model_e2r.predict(poly_reg__e2r.fit_transform([[17]])))
    # print(Ridge_model_e2r.predict(poly_reg__e2r.fit_transform([[10000]])))
    # print(Ridge_model_e2r.predict(poly_reg__e2r.fit_transform([[10200]])))
