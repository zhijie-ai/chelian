#-----------------------------------------------
# -*- encoding=utf-8 -*-                       #
# __author__:'焉知飞鱼'                         #
# CreateTime:                                  #
#       2021/3/1 13:58                         #
#                                              #
#               天下风云出我辈，                 #
#               一入江湖岁月催。                 #
#               皇图霸业谈笑中，                 #
#               不胜人生一场醉。                 #
#-----------------------------------------------
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc
import grpc
import tensorflow as tf
import numpy as np

encoder_inp = np.array([[1,2,3,0,0],[2,3,4,0,0]])
decoder_inp = np.array([[3],[3]])


with grpc.insecure_channel('localhost:8500') as channel:
    stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)
    request = predict_pb2.PredictRequest()
    request.model_spec.name='model'
    request.model_spec.signature_name='test_signature'
    request.inputs["encoder_input"].CopyFrom(
        tf.contrib.util.make_tensor_proto(encoder_inp, shape=list(encoder_inp.shape)))
    request.inputs["decoder_input"].CopyFrom(
        tf.contrib.util.make_tensor_proto(decoder_inp, shape=list(decoder_inp.shape)))
    response = stub.Predict(request, 5.0)#5 secs timeout
    res_from_server_np = tf.make_ndarray(response.outputs['y_pred'])
    print(res_from_server_np)