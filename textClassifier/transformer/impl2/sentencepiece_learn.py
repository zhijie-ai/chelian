#-----------------------------------------------
# -*- encoding=utf-8 -*-                       #
# __author__:'焉知飞鱼'                         #
# CreateTime:                                  #
#       2021/2/22 13:49                         #
#                                              #
#               天下风云出我辈，                 #
#               一入江湖岁月催。                 #
#               皇图霸业谈笑中，                 #
#               不胜人生一场醉。                 #
#-----------------------------------------------
import jieba

def sentence_piece():
    import sentencepiece as spm
    sp = spm.SentencePieceProcessor()

    text="讲解了一下机器学习领域中常见的深度学习LSTM+CRF的网络拓扑 17：了解CRF层添加的好处 18：\
    EmissionScore TransitionScore 19：CRF的目标函数 20：计算CRF真实路径的分数"

    print("*****************text****************")
    print(text)
    print("*****************jieba****************")
    print(' '.join(jieba.cut(text)))

    print("*****************XLNET sentencepiece****************")
    sp.Load("/home/deploy/pre_training/spiece.model")
    print(sp.EncodeAsPieces(text))

    print("***************** unigram_sentencepiece****************")
    sp.Load("/home/deploy/pre_training/sentencepiece/mypiece_unigram.model")
    print(sp.EncodeAsPieces(text))

    print("***************** bpe_sentencepiece****************")
    sp.Load("/home/deploy/pre_training/sentencepiece/mypiece_pbe.model")
    print(sp.EncodeAsPieces(text))


def sentence_piece2():
    import sentencepiece as spm
    import os

    os.makedirs("iwslt2016/segmented2", exist_ok=True)
    train = '--input=iwslt2016/prepro/train.de --pad_id=0 --unk_id=1 \
             --bos_id=2 --eos_id=3\
             --model_prefix=iwslt2016/segmented2/unigram --vocab_size={} \
             --model_type=unigram'.format(32000)
    spm.SentencePieceTrainer.Train(train)

    sp = spm.SentencePieceProcessor()
    sp.Load("iwslt2016/segmented2/unigram.model")
    sent = 'Most of the planet is ocean water'
    pieces = sp.EncodeAsPieces(sent)
    print(pieces)

if __name__ == '__main__':
    sentence_piece2()