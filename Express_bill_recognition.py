import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import paddlenlp
from paddlenlp.datasets import load_dataset
from functools import partial
import pandas as pd
import numpy as np
import random
from paddlenlp.transformers import ErnieTokenizer,ErnieForTokenClassification,LinearDecayWithWarmup
from paddlenlp.metrics import ChunkEvaluator
paddle.seed(1234)
np.random.seed(1234)
random.seed(1234)

#读取数据
def read_data(data_path):
    with open(data_path, 'r', encoding='utf-8') as fp:
        next(fp)  # 跳过列名
        for line in fp.readlines():
            words, labels = line.strip('\n').split('\t')
            words = words.split('\002')
            labels = labels.split('\002')
            yield words, labels
train_ds = load_dataset(read_data,data_path = 'train.txt',lazy=False)
dev_ds = load_dataset(read_data,data_path = 'dev.txt',lazy=False)
# print(train_ds[0])

#将数据序列化
#标签转数字
label_vocab = {'P-B': 0,
 'P-I': 1,
 'T-B': 2,
 'T-I': 3,
 'A1-B': 4,
 'A1-I': 5,
 'A2-B': 6,
 'A2-I': 7,
 'A3-B': 8,
 'A3-I': 9,
 'A4-B': 10,
 'A4-I': 11,
 'O': 12,
 'pad':13}

def convert_example(example,tokenizer,label_vocab):
    tokens, labels = example
    #is_split_into_words代表传入的是已经分割好的数组
    tokenized_input = tokenizer(tokens, return_length=True,is_split_into_words=True,max_seq_len=400)
    # Token '[CLS]' and '[SEP]' will get label 'O'
    labels = ['O'] + labels + ['O']
    #将标签转换成数字
    tokenized_input['labels'] = [label_vocab[x] for x in labels]
    return tokenized_input['input_ids'], tokenized_input['token_type_ids'], tokenized_input['seq_len'], tokenized_input['labels']

tokenizer = ErnieTokenizer.from_pretrained('ernie-1.0')

trans_func = partial(convert_example, tokenizer=tokenizer, label_vocab=label_vocab)

train_ds.map(trans_func)
dev_ds.map(trans_func)
# print (train_ds[0])


#构建dataloader
from paddle.io import DistributedBatchSampler,DataLoader
from paddlenlp.data import Tuple,Stack,Pad

#补齐函数，补齐后，input_ids,token_type_ids,labels三者长度一样
batchify_fn = lambda samples, fn=Tuple(
    Pad(axis=0, pad_val=tokenizer.pad_token_id),  # input_ids
    Pad(axis=0, pad_val=tokenizer.pad_token_type_id),  # token_type_ids
    Stack(),  # seq_len
    Pad(axis=0, pad_val=label_vocab.get('pad',13),dtype='int64')  #用pad对应的id补齐label，后续计算损失时忽略这部分
): fn(samples)
batch_size = 64
train_sampler = DistributedBatchSampler(train_ds,batch_size=batch_size,shuffle=True)
dev_sample = DistributedBatchSampler(dev_ds,batch_size=batch_size,shuffle=False)
train_loader = DataLoader(train_ds,batch_sampler=train_sampler,collate_fn=batchify_fn,return_list=True)
dev_loader = DataLoader(dev_ds,batch_sampler=dev_sample,collate_fn=batchify_fn,return_list=True)


#构建测试集
def read_test_data():
    with open('test.txt','r') as f:
        next(f)
        for i in f:
            words,_ = i.split('\t')
            words = words.split('\002')
            yield words
test_ds = load_dataset(read_test_data,lazy=False)
def convert_test_example(example,tokenizer):
    #is_split_into_words代表传入的是已经分割好的数组
    tokenized_input = tokenizer(example, return_length=True,is_split_into_words=True)
    return tokenized_input['input_ids'], tokenized_input['token_type_ids'], tokenized_input['seq_len']

test_trans_func = partial(convert_test_example, tokenizer=tokenizer)
test_ds.map(test_trans_func)
test_batchify_fn = lambda samples, fn=Tuple(
    Pad(axis=0, pad_val=tokenizer.pad_token_id),  # input_ids
    Pad(axis=0, pad_val=tokenizer.pad_token_type_id),  # token_type_ids
    Stack()  # seq_len
): fn(samples)
batch_size = 64
test_sampler = DistributedBatchSampler(test_ds,batch_size=batch_size,shuffle=False)
test_loader = DataLoader(test_ds,batch_sampler=test_sampler,collate_fn=test_batchify_fn,return_list=True)
id_to_label = dict(zip(label_vocab.values(),label_vocab.keys()))


#ernie+fc训练模型
#定义训练参数
ignore_label = label_vocab.get('pad',13)
model = ErnieForTokenClassification.from_pretrained("ernie-1.0", num_classes=len(label_vocab))
metric = ChunkEvaluator(label_list=label_vocab.keys(), suffix=True)
loss_fn = paddle.nn.loss.CrossEntropyLoss(ignore_index=ignore_label)#忽略-1部分的损失
optimizer = paddle.optimizer.AdamW(learning_rate=2e-5, parameters=model.parameters())


def evaluate(model, metric, data_loader,optimizer,best_f1):
    model.eval()
    metric.reset()
    for input_ids, seg_ids, lens, labels in data_loader:
        logits = model(input_ids, seg_ids)
        preds = paddle.argmax(logits, axis=-1)
        n_infer, n_label, n_correct = metric.compute(None, lens, preds, labels)
        metric.update(n_infer.numpy(), n_label.numpy(), n_correct.numpy())
        precision, recall, f1_score = metric.accumulate()
    print("eval precision: %f - recall: %f - f1: %f" %
          (precision, recall, f1_score))
    if f1_score>best_f1:
        best_f1 = f1_score
        paddle.save(model.state_dict(),'best_ernie_model.bin')
        paddle.save(optimizer.state_dict(),'ernie_optimizer_state.bin')
        paddlenlp.utils.log.logger.info('the best model have been saved,best_f1:%.3f'%best_f1)
    model.train()
    return best_f1
step = 0
best_f1=0


#模型训练与评估
for epoch in range(10):
    for idx, (input_ids, token_type_ids, length, labels) in enumerate(train_loader):
        logits = model(input_ids, token_type_ids)
        loss = paddle.mean(loss_fn(logits, labels))
        loss.backward()
        optimizer.step()
        optimizer.clear_grad()
        step += 1
        if step%10 == 0:
            print("epoch:%d - step:%d - loss: %f" % (epoch, step, loss))
    best_f1 = evaluate(model, metric,dev_loader,optimizer,best_f1)

#预测
model.eval()

pre_list = []
len_list = []
input_id_list = []
for input_ids, seg_ids, lens in test_loader:
    logits = model(input_ids, seg_ids)
    preds = paddle.argmax(logits, axis=-1)
    pre_list.append(preds.numpy())
    len_list.append(lens.numpy())
    input_id_list.append(input_ids.numpy())

final_pre = []
tag_labels = {'A1':'省','A2':'市','A3':'县','A4':'详细地址','P':'姓名','T':'电话'}
for batch_pre,batch_input_id,batch_lens in zip(pre_list,input_id_list,len_list):
    for pre,input_id,lens in zip(batch_pre,batch_input_id,batch_lens):
        pre = [id_to_label[i] for i in pre]
        sent_out = []
        tags_out = []
        words = ''
        for s, t in zip(input_id[:lens], pre[:lens]):
            s = tokenizer.convert_ids_to_tokens(s)
            if t.endswith('-B'):
                if len(words):
                    sent_out.append(words)
                tags_out.append(tag_labels[t.split('-')[0]])
                words = s
            elif t.endswith('-I'):
                words += s
        sent_out.append(words)
        final_pre.append((sent_out,tags_out))

print(final_pre)

