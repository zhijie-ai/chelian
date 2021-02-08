#----------------------------------------------
# -*- encoding=utf-8 -*-                      #
# __author__:'xiaojie'                        #
# CreateTime:                                 #
#       2019/8/15 18:26                       #
#                                             #
#               天下风云出我辈，                 #
#               一入江湖岁月催。                 #
#               皇图霸业谈笑中，                 #
#               不胜人生一场醉。                 #
#----------------------------------------------
import tensorflow_hub as hub


elmo = hub.Module("https://tfhub.dev/google/elmo/2", trainable=True)

# just a random sentence

x = ["Roasted ants are a popular snack in Columbia"]

# Extract ELMo features

embeddings = elmo(x, signature="default", as_dict=True)["elmo"]