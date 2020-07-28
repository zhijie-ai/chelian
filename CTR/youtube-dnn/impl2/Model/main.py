#-----------------------------------------------
# -*- encoding=utf-8 -*-                       #
# __author__:'焉知飞鱼'                         #
# CreateTime:                                  #
#       2020/7/27 11:53                         #
#                                              #
#               天下风云出我辈，                 #
#               一入江湖岁月催。                 #
#               皇图霸业谈笑中，                 #
#               不胜人生一场醉。                 #
#-----------------------------------------------
from Data.data import *
from Model.embedding import *
from Model.model import *


class config():
    # batch_size = 128
    embedding_size = 256
    skip_window = 1  # How many words to consider left and right.
    num_skips = 2  # How many times to reuse an input to generate a context.
    num_sampled = 64
    num_steps = 100001
    DNN_learning_rate = 1.0
    DNN_drop_rate = 0.95
    DNN_layer1_size = 1024
    DNN_layer2_size = 512


def main():
    # Load data
    print("Start loading data...")
    user, business = load_grouped_data()
    print("Data loading finished...")

    # Extract Embedding Data
    print("Start Extracting for embedding...")
    user_bus_data, user_bus_dict, reverse_user_bus_dict, user_bus_count = extract_embedding_user_business(user)
    business_cat_data, business_cat_count = extract_embedding_catagory(business)
    print("Extracting finished...")

    print("Start Embedding...")
    user_bus_embed = Embedding(user_bus_data, user_bus_count)

    bus_cat_embed = Embedding(business_cat_data, business_cat_count)

    #TODO:get input & label, business number for DNN
    business_size = 0
    inputs = ()
    labels = ()

    #training
    rev_embed_size = config.embedding_size #TODO:GET EMDEDDINGSIZES FOR CAT & BUSINESS ,256
    cat_embed_size = config.embedding_size #,256

    model = DNNmodel(rev_embed_size, cat_embed_size, business_size)
    model.train(inputs, labels)






if __name__ == '__main__':
    main()