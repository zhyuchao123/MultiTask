import torch
import torch.autograd.variable as Variable
import torch.nn as nn
import torch.utils
# import mt_bluebert.blue_metrics as blue_metrics
# from multitask import blue_metrics
# import blue_metrics
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import classification_report,f1_score,accuracy_score,recall_score,precision_score

class biosses_sts(nn.Module):
    def __init__(self, in_features, out_features, drop_p=0, max_sen_size=256):
        super(biosses_sts, self).__init__()
        self.infeatures = in_features
        self.out_features = out_features

        self.Linear = nn.Linear(in_features, out_features)
        # self.Dropout = nn.Dropout(drop_p)
        self.Relu = nn.LeakyReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_tensor):
        # input_tensor:(16, 256, 768) (batch_size, max_seq_len,embed_size)

        # out = input_tensor.mean(axis=1)
        out = input_tensor[:, 0, :]
        # out:(16,768)
        out = self.Linear(out)
        # out:(16,1)
        # out = self.Dropout(out)
        out = self.Relu(out)
        out = out.squeeze(-1)
        # out:(16)

        return self.sigmoid(out)  # out:(16)


class chemprot_mul_classication(nn.Module):
    def __init__(self, max_sen_size, word_embed_size=768, num_labels=6, drop_p=0):
        super(chemprot_mul_classication, self).__init__()
        self.word_embed_size = word_embed_size
        self.num_labels = num_labels
        # self.Pooling = nn.AvgPool1d(kernel_size=word_embed_size, stride=1)
        self.max_sen_size = max_sen_size
        self.Linear = nn.Linear(word_embed_size, num_labels)
        # self.Dropout = nn.Dropout(drop_p)
        self.Relu = nn.LeakyReLU()

        self.softmax = nn.Softmax(-1)

    def forward(self, input_tensor):
        # input_tensor:(16, 256, 768) (batch_size, max_seq_len,embed_size)
        # (16,1,768)
        # (16,768)
        # out = self.Pooling(input_tensor)
        out = input_tensor[:, 0, :]
        # out = self.Dropout(out)
        out = out.squeeze(1)


        out = self.Linear(out)
        out = self.Relu(out)

        return self.softmax(out)


class mednli_entail(nn.Module):
    def __init__(self, max_sen_size, word_embed_size=768, num_labels=3, drop_p=0):
        super(mednli_entail, self).__init__()
        self.word_embed_size = word_embed_size
        self.num_labels = num_labels
        # self.Pooling = nn.AvgPool1d(kernel_size=word_embed_size, stride=1)
        self.max_sen_size = max_sen_size
        self.Linear = nn.Linear(word_embed_size, num_labels)
        # self.Dropout = nn.Dropout(drop_p)
        self.Relu = nn.LeakyReLU()

        self.softmax = nn.Softmax(-1)

    def forward(self, input_tensor):
        # out = self.Pooling(input_tensor)
        # out = self.Dropout(out)
        out = input_tensor[:, 0, :]

        out = out.squeeze(1)
        out = self.Linear(out)
        out = self.Relu(out)
        return self.softmax(out)


class hoc_doc_classification(nn.Module):
    def __init__(self, max_sen_size, word_embed_size=768, num_labels=11, drop_p=0):
        super(hoc_doc_classification, self).__init__()
        self.word_embed_size = word_embed_size
        self.num_labels = num_labels
        self.max_sen_size = max_sen_size
        self.Linears = nn.ModuleList([ nn.Linear(word_embed_size, 1) for lb in range(num_labels)])

        self.Relu = nn.LeakyReLU()
        self.sigmoid = nn.Sigmoid()
        self.Tanh = nn.Tanh()
    def forward(self, input_tensor):
        # out = self.Pooling(input_tensor)
        # out = self.Dropout(out)
        out = input_tensor[:, 0, :]
        out = out.squeeze(1)
        out = self.Tanh(out)

        out = torch.cat([linearfc(out) for linearfc in self.Linears],dim=1)
        out = self.Relu(out)

        return out


class ddi2010_mul_classification(nn.Module):
    def __init__(self, max_sen_size, word_embed_size=768, num_labels=5, drop_p=0):
        super(ddi2010_mul_classification, self).__init__()
        self.word_embed_size = word_embed_size
        self.num_labels = num_labels
        self.max_sen_size = max_sen_size
        self.Linear = nn.Linear(word_embed_size, num_labels)
        self.Dropout = nn.Dropout(drop_p)
        self.Relu = nn.LeakyReLU()
        self.softmax = nn.Softmax(-1)

    def forward(self, input_tensor):
        # out = self.Pooling(input_tensor)
        # out = self.Dropout(out)
        out = input_tensor[:, 0, :]

        out = out.squeeze(1)
        out = self.Linear(out)
        out = self.Dropout(out)
        out = self.Relu(out)
        return out


class multi_task_model(nn.Module):
    def __init__(self, base_model, frozen=True, device='cuda', max_sen_size=512):
        """
        2020.9.30 Yuchao Zhang
        base_model means pre-trained model
        base model in train if frozen is true

        """
        super(multi_task_model, self).__init__()
        self.device = device
        self.base_model = base_model
        """
        see the structure of 
        """

        self.base_model.train() if frozen == False else self.base_model.eval()
        self.biosses_sts = biosses_sts(max_sen_size=max_sen_size, in_features=768, out_features=1)
        self.chemprot_mul_classication = chemprot_mul_classication(max_sen_size=max_sen_size, word_embed_size=768,
                                                                   num_labels=6)
        self.mednli_entail = mednli_entail(max_sen_size=max_sen_size, word_embed_size=768, num_labels=3)

        self.hoc_doc_classification = hoc_doc_classification(max_sen_size=max_sen_size, word_embed_size=768,
                                                             num_labels=11)

        self.ddi2010_mul_classification = ddi2010_mul_classification(max_sen_size=max_sen_size, word_embed_size=768,
                                                                     num_labels=5, drop_p=0)
        """
        add more task head below
        """

    def forward(self, feature, dataset_name):
        if feature is None:
            # when reach the edge of the dataLoader
            # return None
            return None
        input_ids = feature['input_ids'].to(self.device)
        # input_ids = Variable(input_ids)
        input_mask = feature['input_mask'].to(self.device)
        segment_ids = feature['segment_ids'].to(self.device)
        if dataset_name == 'biosses':
            return self.biosses_forward(input_ids=input_ids, token_type_ids=segment_ids, attention_mask=input_mask), \
                   feature['score']
        elif dataset_name == "chemprot":
            return self.chemprot_forward(input_ids=input_ids, token_type_ids=segment_ids, attention_mask=input_mask), \
                   feature['label']
        elif dataset_name == "mednli":
            return self.mednli_forward(input_ids=input_ids, token_type_ids=segment_ids, attention_mask=input_mask), \
                   feature['label']
        elif dataset_name == 'hoc':
            return self.hoc_forward(input_ids=input_ids, token_type_ids=segment_ids, attention_mask=input_mask), \
                   feature['label']
        elif dataset_name == 'ddi2010':
            return self.ddi2010_forward(input_ids=input_ids, token_type_ids=segment_ids, attention_mask=input_mask), \
                   feature['label']

    def biosses_forward(self, input_ids, token_type_ids, attention_mask):
        if input_ids is None:
            return None
        else:

            digit, hidden = self.base_model(output_hidden_states=True,
                                            input_ids=input_ids,
                                            token_type_ids=token_type_ids,
                                            attention_mask=attention_mask)

            # temp_result = self.biosses_sts(hidden[-1])
            return self.biosses_sts(hidden[-1])

    def chemprot_forward(self, input_ids, token_type_ids, attention_mask):
        if input_ids is None:
            return None
        else:
            digit, hidden = self.base_model(output_hidden_states=True,
                                            input_ids=input_ids,
                                            token_type_ids=token_type_ids,
                                            attention_mask=attention_mask)
            return self.chemprot_mul_classication(hidden[-1])

    def mednli_forward(self, input_ids, token_type_ids, attention_mask):
        if input_ids is None:
            return None
        else:
            digit, hidden = self.base_model(output_hidden_states=True,
                                            input_ids=input_ids,
                                            token_type_ids=token_type_ids,
                                            attention_mask=attention_mask)
            return self.mednli_entail(hidden[-1])

    def hoc_forward(self, input_ids, token_type_ids, attention_mask):
        if input_ids is None:
            return None
        else:
            digit, hidden = self.base_model(output_hidden_states=True,
                                            input_ids=input_ids,
                                            token_type_ids=token_type_ids,
                                            attention_mask=attention_mask)
            return self.hoc_doc_classification(hidden[-1])

    def ddi2010_forward(self, input_ids, token_type_ids, attention_mask):
        if input_ids is None:
            return None
        else:
            digit, hidden = self.base_model(output_hidden_states=True,
                                            input_ids=input_ids,
                                            token_type_ids=token_type_ids,
                                            attention_mask=attention_mask)
            return self.ddi2010_mul_classification(hidden[-1])


class MultiTaskLossWrapper(nn.Module):
    def __init__(self, model, task_num, device='cuda', fine_tune=False):
        """
        model is the base model
        task_num is the number of tasks
        if fine_tune is True, the back propogation will not flow back to the model
        """

        super(MultiTaskLossWrapper, self).__init__()
        self.model = model.to(device)
        self.task_num = task_num
        """define Loss calculators"""
        self.MSE_loss = nn.MSELoss(reduction="mean")
        # different task use different loss setting
        # chemprot loss use weight according to the train data
        # false    14757
        # CPR:3      768
        # CPR:4     2251
        # CPR:5      173
        # CPR:6      235
        # CPR:9      727

        # define loss for chemprot
        t_chemprot = 1 / torch.tensor([14757, 768, 2251, 173, 235, 727], dtype=torch.float32)
        w_chemprot = (t_chemprot / t_chemprot.sum())
        self.CrossEntropy_loss_chemprot = nn.CrossEntropyLoss(w_chemprot.to(device))

        # define loss for mednli
        t_mednli = 1 / torch.tensor([1, 1, 1], dtype=torch.float32)
        w_mednli = (t_mednli / t_mednli.sum())
        self.CrossEntropy_loss_mednli = nn.CrossEntropyLoss(w_mednli.to(device))

        # loss for hoc
        # t_hoc = torch.tensor([7389, 723, 563, 596, 346, 213, 164, 238, 264, 458, 148], dtype=torch.float)
        t_hoc = torch.tensor([13000, 723, 563, 596, 346, 213, 164, 238, 264, 458, 148], dtype=torch.float)
        w_hoc =  (1 / t_hoc) / (1 / t_hoc).sum()

        self.MultiLabel_loss_hoc = nn.MultiLabelSoftMarginLoss(weight=w_hoc.to(device), reduction='sum')

        # loss for ddi2010
        t_ddi2010 = 1 / torch.tensor([15842,946,1212,633,146], dtype=torch.float32)
        w_ddi2010 = (t_ddi2010 / t_ddi2010.sum())
        self.CrossEntropy_loss_ddi2010 = nn.CrossEntropyLoss(w_ddi2010.to(device))

        self.device = device
        self.fine_tune = fine_tune

    def forward(self, features, dataset_names):
        losses = []
        for feature, dataset_name in zip(features, dataset_names):
            if feature is None:
                continue
            if self.fine_tune == True:
                with torch.no_grad():
                    predict, target = self.model(feature, dataset_name)
            else:
                predict, target = self.model(feature, dataset_name)

            # predict = predict.to('cpu')
            loss = self.get_train_loss(predict, target, dataset_name)
            # torch.cuda.empty_cache()
            # loss = loss.cpu()
            # torch.cuda.empty_cache()

            losses.append(loss)
        # return a list
        return losses

    def get_train_loss(self, predict, target, dataset_name):
        # Reuse the loss defined by Yifan Peng
        # refered link: https://github.com/ncbi-nlp/bluebert/blob/master/mt-bluebert/mt_bluebert/blue_metrics.py
        if dataset_name == 'biosses':
            # biosses can use pearsonr
            return self.MSE_loss(predict.to(self.device), target.to(self.device))
        elif dataset_name == "chemprot":
            return self.CrossEntropy_loss_chemprot(predict.to(self.device), target.to(self.device))

        elif dataset_name == "mednli":
            return self.CrossEntropy_loss_mednli(predict.to(self.device), target.to(self.device))

        elif dataset_name == "hoc":
            return self.MultiLabel_loss_hoc(predict.to(self.device), target.to(self.device))

        elif dataset_name == "ddi2010":
            return self.CrossEntropy_loss_ddi2010(predict.to(self.device), target.to(self.device))

        """add more metrics below"""
    def get_eval_loss(self, predict, target, dataset_name):
        # Reuse the loss defined by Yifan Peng
        # refered link: https://github.com/ncbi-nlp/bluebert/blob/master/mt-bluebert/mt_bluebert/blue_metrics.py
        if dataset_name == 'biosses':
            # biosses can use pearsonr
            return pearsonr(predict, target)[0]
        elif dataset_name == "chemprot":
            # _value, predict = predict.max(1)
            return f1_score(predict, target,average='micro')
        elif dataset_name == "mednli":
            # _value, predict = predict.max(1)

            return accuracy_score(predict, target)
        elif dataset_name == "hoc":
            return f1_score(predict, target,average='micro')
        elif dataset_name == "ddi2010":
            return f1_score(predict, target,average='micro')
        """add more metrics below"""

    def get_multiTask_model(self):
        return self.model

    def evaluation_result(self, feature, dataset_name):
        if dataset_name == 'biosses':
            # biosses can use pearsonr
            return self.model(feature, dataset_name)
        elif dataset_name == "chemprot":
            predicts, target = self.model(feature, dataset_name)
            _, predicts_max = predicts.max(1)

            return predicts_max, target
        elif dataset_name == "mednli":
            predicts, target = self.model(feature, dataset_name)
            _, predicts_max = predicts.max(1)

            return predicts_max, target
        elif dataset_name == "hoc":
            predicts, target = self.model(feature, dataset_name)
            predicts_ = torch.zeros_like(predicts)
            predicts = predicts_.masked_fill(predicts>=0.5,value=1)
            return predicts, target
        elif dataset_name == "ddi2010":
            predicts, target = self.model(feature, dataset_name)
            _, predicts_max = predicts.max(1)
            return predicts_max, target

# from torch.autograd import Variable
# #
# # loss = nn.CrossEntropyLoss()
# output = Variable(torch.FloatTensor([[0,0,0,1],[1,0,0,0]]))
# # target = Variable(torch.LongTensor([3,0]))
# # print(target)
# # #
# # print(loss(output,target))
# values , predict_labels = (output.max(axis=1))
# print(predict_labels )
