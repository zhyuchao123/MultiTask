import torch
import os
import logging
from tqdm import tqdm
import transformers
from transformers import AutoTokenizer, AutoModelWithLMHead
# import transformers.tokenization_bert as tokenizer
from transformers import BertTokenizer, BertConfig, BertModel
import logger
import pandas as pd
# from mt_bluebert.module.task_heads import multi_task_model,MultiTaskLossWrapper
from module.task_heads import multi_task_model,MultiTaskLossWrapper
# the multi_task_model and MultiTaskLossWrapper is defined by Yuchao Zhang 2020.09
# import mt_bluebert.module.my_eval_online_model as my_eval
from module import my_eval_online_model as my_eval
import torch.utils.data as data
import itertools
from torch.optim.lr_scheduler import ReduceLROnPlateau
################################################################################
class MyDataset(torch.utils.data.Dataset):
    def __init__(self, features,dataset_name):
        self.features = features
        self.len = len(features)
        self.dataset_name = dataset_name
    def __getitem__(self, index):
        feature = self.features[index]
        if self.dataset_name == "biosses":
            return {"input_ids": torch.tensor(feature['input_ids'], dtype=torch.int64),
                    "input_mask": torch.tensor(feature['input_mask'], dtype=torch.int64),
                    "segment_ids": torch.tensor(feature['segment_ids'], dtype=torch.int64),
                    "score": torch.tensor(feature['score'], dtype=torch.float,requires_grad=True)
                # ,"old_index": torch.tensor(feature['old_index'], dtype=torch.int64)
                # ,"dataset_name": "biosses"
                    }
        elif self.dataset_name== "mednli" :
            return {"input_ids": torch.tensor(feature['input_ids'], dtype=torch.int64),
                    "input_mask": torch.tensor(feature['input_mask'], dtype=torch.int64),
                    "segment_ids": torch.tensor(feature['segment_ids'], dtype=torch.int64),
                    "label": torch.tensor(feature['label'],dtype=torch.int64).detach()

                # ,"old_index": torch.tensor(feature['old_index'], dtype=torch.int64)
                # ,"dataset_name": "mednli"
                    }
        elif self.dataset_name== "chemprot":
            return {"input_ids": torch.tensor(feature['input_ids'], dtype=torch.int64),
                    "input_mask": torch.tensor(feature['input_mask'], dtype=torch.int64),
                    "segment_ids": torch.tensor(feature['segment_ids'], dtype=torch.int64),
                    "label": torch.tensor(feature['label'],dtype=torch.int64).detach()
                # ,"old_index": torch.tensor(feature['old_index'], dtype=torch.int64)
                # ,"dataset_name": "chemprot"
                    }
        elif self.dataset_name == "hoc":
            return {"input_ids": torch.tensor(feature['input_ids'], dtype=torch.int64),
                    "input_mask": torch.tensor(feature['input_mask'], dtype=torch.int64),
                    "segment_ids": torch.tensor(feature['segment_ids'], dtype=torch.int64),
                    "label": torch.tensor(feature['label'], dtype=torch.int64).detach()
                    # ,"old_index": torch.tensor(feature['old_index'], dtype=torch.int64)
                    # ,"dataset_name": "chemprot"
                    }
        elif self.dataset_name == "ddi2010":
            return {"input_ids": torch.tensor(feature['input_ids'], dtype=torch.int64),
                    "input_mask": torch.tensor(feature['input_mask'], dtype=torch.int64),
                    "segment_ids": torch.tensor(feature['segment_ids'], dtype=torch.int64),
                    "label": torch.tensor(feature['label'], dtype=torch.int64).detach()
                    # ,"old_index": torch.tensor(feature['old_index'], dtype=torch.int64)
                    # ,"dataset_name": "chemprot"
                    }
    def get_dataset_name(self):
        return self.dataset_name
    def __len__(self):
        return self.len


################################################################################
# testing functions
def printlines(text, n=5):
    # function for print first n lines(default 5)
    limit = len(text)
    for i, t in enumerate(text):
        if i < n and i < limit:
            print(t)
            print("###sep###")


################################################################################
def loadmodel(model_dir=None,from_net=True,frozen=True,task_num = 5):
    """
    return: a base model
    model dir is where the parameters and config of the model in
    from_net means the pre-trained model download from transformer library

    """
    model = None
    if from_net == True and model_dir is None:
        # load
        pretrained_model = AutoModelWithLMHead.from_pretrained("dmis-lab/biobert-v1.1")
        model_ = multi_task_model(base_model=pretrained_model,frozen=frozen,device='cuda')
        # model =  MultiTaskLossWrapper(model=model_,task_num=task_num)
        return model_


    """
    
    
    """
    if model_dir == None:
        modeldir = os.path.abspath(os.path.dirname(__file__)+ "/../../")
        modelpath = os.path.join(modeldir, r"models\NCBI_BERT_pubmed_mimic_uncased_L-12_H-768_A-12\bert_model2.pt")

    checkpoint = torch.load(modelpath, map_location=lambda storage, loc: storage)
    state = checkpoint["state"]
    config_ = checkpoint["config"]
    config = transformers.PretrainedConfig()
    config.update(config_)
    model = transformers.BertModel(config=config)
    model.state_dict(state)
    return model




def convert_single_example(df, max_seq_length, tokenizer,is_sentence_pair=True,dataset_name="biosses"):
    # dataframe ['old_index''sentence1','sentence2','score']

    is_sentence_pair = True
    labels = {}
    if dataset_name == "biosses":
        is_sentence_pair = True
    elif dataset_name == "chemprot":
        is_sentence_pair = False
        # labels = {"false":0,"CPR:3":1,"CPR:4":2,"CPR:5":3,"CPR:6":4,"CPR:9":5}
        labels = {"false":0,"CPR:3":1,"CPR:4":2,"CPR:5":3,"CPR:6":4,"CPR:9":5}

    elif dataset_name == "mednli":
        is_sentence_pair = True
        # labels = {"entailment":0,"neutral":1,"contradiction":2}
        labels = {"entailment": 0, "neutral": 1, "contradiction": 2}
    elif dataset_name == 'hoc':
        is_sentence_pair = False
        labels = list(map(int, df.iloc[0,df.columns.get_loc('label')]))
    elif dataset_name == 'ddi2010':
        is_sentence_pair = False
        labels = {'DDI-false': 0, 'DDI-mechanism': 1, 'DDI-effect': 2, 'DDI-advise': 3, 'DDI-int': 4}

    s1_attr_name = "sentence1" if is_sentence_pair else "sentence"
    s1 = df[s1_attr_name].values[0]

    s2 = df['sentence2'].values[0] if is_sentence_pair else None

    tokens_length =  len(tokenizer.encode(s1,s2))
    encoder_result = tokenizer.encode_plus(s1,s2,truncation='longest_first',max_length=max_seq_length,padding= 'max_length')
    input_ids = encoder_result['input_ids']
    input_mask = encoder_result['attention_mask']
    segment_ids = encoder_result['token_type_ids']
    if dataset_name == "biosses":
        feature = InputFeatures(
            input_ids=input_ids,
            input_mask=input_mask,
            segment_ids=segment_ids,
            score=df['score'].values[0],
            label= None,
            old_index=df['old_index'].values[0],
            tokens_length=tokens_length)
    elif dataset_name == "chemprot":
        feature = InputFeatures(
            input_ids=input_ids,
            input_mask=input_mask,
            segment_ids=segment_ids,
            score= None,
            label=labels[str(df['label'].values[0])],
            old_index=df['index'].values[0],
            tokens_length=tokens_length)
    elif dataset_name == "mednli":
        feature = InputFeatures(
            input_ids=input_ids,
            input_mask=input_mask,
            segment_ids=segment_ids,
            score=None,
            label= labels[str(df['label'].values[0])] ,
            old_index=df['index'].values[0],
            tokens_length=tokens_length)
    elif dataset_name == 'hoc':
        feature = InputFeatures(
            input_ids=input_ids,
            input_mask=input_mask,
            segment_ids=segment_ids,
            score=None,
            label=labels,
            old_index=df['index'].values[0],
            tokens_length=tokens_length
        )
    elif dataset_name == 'ddi2010':
        feature = InputFeatures(
            input_ids=input_ids,
            input_mask=input_mask,
            segment_ids=segment_ids,
            score=None,
            label=labels[str(df['label'].values[0])],
            old_index=df['index'].values[0],
            tokens_length=tokens_length
        )
    assert feature is not None, "dataset_name is not specified or invalid"
    return feature


class InputFeatures(object):
    def __init__(self, input_ids, input_mask, segment_ids, score, old_index, label, tokens_length):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.score = score
        self.old_index = old_index
        self.label = label
        self.tokens_length = tokens_length
    def __getitem__(self, item):
        table = {"input_ids": self.input_ids,
                 "input_mask": self.input_mask,
                 "segment_ids": self.segment_ids,
                 "score": self.score,
                 "old_index": self.old_index,
                 "label":self.label,
                 "tokens_length" : self.tokens_length}
        return table[item]


def loadData(path=None, dataset_name="biosses", data_type='train'):
    # load data from datasets
    # data_types = ["train","test","dev"]
    if path == None:
        # default data type is training data
        path = os.path.join(os.path.abspath(os.path.dirname(__file__) + "/datasets/" + dataset_name))

    else:
        path = os.path.join(path, data_type)

    # df = pd.read_csv(filepath_or_buffer=os.path.join(path, data_type + ".tsv"), sep='\t')
    # df has attributes
    # print(df.head(2))
    # [genre,filename,year,old_index,source1,source2,sentence1,sentence2,score]
    if dataset_name == "biosses":
        df = pd.read_csv(filepath_or_buffer=os.path.join(path, data_type + ".tsv"), sep='\t')

        return df[['old_index', 'sentence1', 'sentence2', 'score']]
    elif dataset_name == "chemprot":
        df = pd.read_csv(filepath_or_buffer=os.path.join(path, data_type + ".tsv"), sep='\t')

        return df[['index', 'sentence', 'label']]
    elif dataset_name == "mednli":
        df = pd.read_csv(filepath_or_buffer=os.path.join(path, data_type + ".tsv"), sep='\t')

        return df[['index', 'sentence1', 'sentence2', 'label']]
    elif dataset_name == 'hoc':
        df = pd.read_csv(filepath_or_buffer=os.path.join(path, data_type + ".tsv"), sep='\t',dtype='str')
        return df[['index', 'sentence', 'label']]
    elif dataset_name == 'ddi2010':
        df = pd.read_csv(filepath_or_buffer=os.path.join(path, data_type + ".tsv"), sep='\t',dtype='str')
        return df[['index', 'sentence', 'label']]

class MultiTask_dataLoader(object):
    def __init__(self,datasets:MyDataset):

        self.dataLoaders = {dataset.get_dataset_name():torch.utils.data.DataLoader(dataset,shuffle=True) for name,dataset in datasets}
        self.one_epoch_data_count = max(map(len,datasets)) # the data quantity depends on the largest dataset
    def __iter__(self):
        pass
        # try:

class train():
    def __init__(self, base_model, data_Sets, lr=1e-5, device= 'cuda' , batch_size= 2, shuffle=False,task_num=3, accumulate_iter=10):
        # get the trained model's path
        self.dir = os.path.abspath(os.path.dirname(__file__))
        self.dir = os.path.join(self.dir, "models\model_out")
        # list the model in that directory
        dirs = os.listdir(self.dir)
        self.time_for_sumloss = accumulate_iter
        self.optimizer = None
        self.epoch_losses = []
        def findRecentModel(name):
            # sort key function
            # print(name.split("_"))
            return int(name.split('_')[1])
        self.lr = lr
        # self.device = 'cpu' if not use_cuda or not torch.cuda.is_available() else 'cuda'
        self.device = device
        self.schedule = None
        if len(dirs)==0:
            # when there's no pre-trained model
            self.data_set_Names = [data_set.get_dataset_name() for data_set in data_Sets]
            self.data_Loaders = [torch.utils.data.DataLoader(dataset=data_set, batch_size=batch_size, shuffle=shuffle)
                                 for data_set in data_Sets]

            # define task specified layers start

            self.model = MultiTaskLossWrapper(model=base_model, task_num=task_num,fine_tune=False)
            self.optimizer = torch.optim.Adam(self.model.parameters(),
                                              lr=self.lr, eps=1e-12, weight_decay=1e-8)
            # self.model.train()
            self.schedule = ReduceLROnPlateau(self.optimizer,patience=2,verbose=True,factor=0.2)
            # If the total loss does not decreaes for 2 epoch,reduce the learning rate by 0.2* self.lr
            self.origin_stamp = 0
            # define task specified layers end
        else:
            dirs.sort(key=findRecentModel)
            content = torch.load(os.path.join(self.dir,dirs[-1]))
            self.data_set_Names = content["data_set_Names"]
            if len(data_Sets)==0:
                self.data_Loaders = content["data_Loaders"]
            else:
                self.data_Loaders = [
                    torch.utils.data.DataLoader(dataset=data_set, batch_size=batch_size, shuffle=shuffle)
                    for data_set in data_Sets]
            # define task specified layers start

            modeldict = content["model"]
            self.model = MultiTaskLossWrapper(model=base_model, task_num=task_num,fine_tune=False)
            modeldict.pop("CrossEntropy_loss_chemprot.weight")
            modeldict.pop('CrossEntropy_loss_mednli.weight')
            modeldict.pop('MultiLabel_loss_hoc.weight')
            modeldict.pop('CrossEntropy_loss_ddi2010.weight')

            self.model.load_state_dict(modeldict,strict=False)
            self.model.train()

            self.optimizer = torch.optim.Adam(self.model.parameters())
            self.optimizer.load_state_dict(content["optimizer"])

            self.schedule = ReduceLROnPlateau(self.optimizer, patience=2, verbose=True,factor=0.2)
            self.origin_stamp = content["stamp"]
            # setattr(self.model,"fine_tune",False)
    def save_model(self, stamp,loss):

        content = {"data_set_Names":self.data_set_Names,
                   "data_Loaders":self.data_Loaders,
                   "model" : self.model.state_dict(),
                   "optimizer":self.optimizer.state_dict(),
                   "stamp":stamp,
                   "loss":loss}

        torch.save(content,f=os.path.join(self.dir, "model_{}_.pt".format(stamp)))
    def run(self, epoch=None):
        if epoch is None:
            epoch = 5
        logger.logger.info("start for training ......")

        total_iters = max([len(dl) for dl in self.data_Loaders])
        for i in range(epoch):
            loss_list = []
            loops = tqdm(enumerate(itertools.zip_longest(*self.data_Loaders,fillvalue=None)),total=total_iters,leave=False)
            # loops = tqdm(range(total_iters),total=total_iters,leave=False)

            count = 0
            self.epoch_losses = []
            if i % 5 == 0:
                self.save_model(i + self.origin_stamp, self.epoch_losses)
            # myiters = [iter(dl) for dl in self.data_Loaders]
            for ind, features in loops:
            # for ind in loops:

                # logger.logger.info("start interating on dataloaders ......")
                # torch.cuda.empty_cache()

            # for iter_count in range(total_iters):
                if count % self.time_for_sumloss == 0:
                    # clear accumulated loss
                    self.optimizer.zero_grad()
                # for dataset_id, eachiter in enumerate(myiters):
                #     try:
                #         features = next(eachiter)
                #     except StopIteration:
                #         myiters[dataset_id] = iter(self.data_Loaders[dataset_id])



                temp_loss = self.model(features, self.data_set_Names)
                # loss_list.extend(temp_loss)

                # if count % self.time_for_sumloss == 0:
                # _avg_loss = torch.mean(torch.tensor(temp_loss, requires_grad=True))
                # temp_loss[-1].retain_grad()
                _avg_loss = sum(temp_loss)/len(temp_loss)
                _avg_loss.backward()

                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1)
                loss_list.append(_avg_loss.item())
                self.epoch_losses.append(_avg_loss.item())
                # print(_avg_loss.grad)
                # print(self.model.model.biosses_sts.Linear.weight.grad)
                # print(self.model.model.chemprot_mul_classication.Linear.weight.grad)

                # print(self.model.model.base_model.bert.embeddings.word_embeddings.weight.grad)
                # temp = self.model.model.base_model.bert
                # print(self.model.model.base_model.bert.encoder.layer[-2].output.dense.weight.grad)
                # print(self.model.model.base_model.bert.pooler.dense.weight.grad)
                #
                # _avg_loss.backward()
                # for l in temp_loss:
                #     l.backward()
                # _avg_loss = sum(temp_loss)
                # self.epoch_losses.append(_avg_loss.item())
                # _avg_loss.backward()
                if count % self.time_for_sumloss == 0:
                    # print(self.model["model.biosses_sts.Linear.weight"].grad)
                    self.optimizer.step()


                    # count = 0
                # loss_list = []
                loops.set_description(f"Epoch {i+self.origin_stamp} [{count}/{total_iters}]")
                loops.set_postfix(losses=["{:.6f}".format(err.item()) for err in temp_loss])
                torch.cuda.empty_cache()
                count += 1 # count is for loss accumulation

            self.schedule.step(sum(loss_list) / len(loss_list))
            # to check if the loss decrease after a epoch, if not, decrease the learning rate



def evaluation(model, model_dict_dir ,eval_dataSets,device,batch_size):
    my_eval.evaluation(model=model, model_dict_path=model_dict_dir, eval_dataSets=eval_dataSets,device=device,batch_size=batch_size)
    

def get_features(df,max_seq_length, tokenizer , dataset_name, stop_size=100,):
    features = []
    logger.logger.info("extract features for {}".format(dataset_name))
    total = len(df)
    logger.logger.info("totally "+str(total)+" examples for "+dataset_name)
    loop = tqdm(range(total),total=total,leave=False)
    for index in loop:

        _feature = convert_single_example(df=df[index:index + 1], max_seq_length=max_seq_length,
                                          tokenizer=tokenizer, dataset_name=dataset_name)
        # if _feature['tokens_length'] > max_seq_length:
        #     continue
        # if dataset_name == 'hoc':
        #     _feature['label'] == '100000000'
        features.append(_feature)
        loop.set_description(f"{dataset_name} [{index}/{total}]")
        # if index>=stop_size:
        #     return features
    logger.logger.info("finally "+str(len(features))+" examples for "+dataset_name)

    return features

# def main():
    # torch.set_num_threads(6)
    # logging.info(torch.__config__.parallel_info())
    # device = "cuda" if torch.cuda.is_available() else "cpu"
    # dir = os.path.abspath(os.path.dirname(__file__)+ "/../../")
    # model_dir = os.path.abspath(os.path.dirname(__file__)+ "/../../")
    # model_dir = os.path.join(model_dir, r"models\model_out")
    # models_in_file = os.listdir(model_dir)
    # train_epoch = 1000
    # task_num = 5 # define the number of tasks
    #
    # def findRecentModel(name):
    #     return int(name.split('_')[1])
    #
    # max_seq_length = 256 # define the maximum length for input
    #
    # # load online tokenizer
    # tokenizer = AutoTokenizer.from_pretrained("dmis-lab/biobert-v1.1")
    #
    #
    # do_train = False
    # if len(models_in_file)== 0 and do_train:
    # # if do_train:
    #     """
    #     The model has not been trained(determined by detecting the model in the file)
    #     """
    #     model = loadmodel(from_net=True,frozen=False,task_num= task_num)  # load default model
    #
    #
    #
    #     # start data preparation
    #     biosses_dataframe = loadData(dataset_name="biosses",data_type="train")  # load biosses data
    #     biosses_features = get_features(df=biosses_dataframe,max_seq_length=max_seq_length, tokenizer= tokenizer , dataset_name = "biosses")
    #
    #     chemprot_dataframe = loadData(dataset_name="chemprot",data_type="train")
    #     chemprot_features = get_features(df=chemprot_dataframe,max_seq_length=max_seq_length, tokenizer= tokenizer , dataset_name = "chemprot")
    #
    #     mednli_dataframe = loadData(dataset_name="mednli",data_type="train")
    #     mednli_features = get_features(df=mednli_dataframe,max_seq_length=max_seq_length, tokenizer= tokenizer , dataset_name = "mednli")
    #
    #     hoc_dataframe = loadData(dataset_name='hoc',data_type='train')
    #     hoc_features = get_features(df=hoc_dataframe,max_seq_length=max_seq_length, tokenizer= tokenizer , dataset_name = "hoc")
    #
    #     ddi2010_dataframe = loadData(dataset_name='ddi2010',data_type='train')
    #     ddi2010_features = get_features(df=ddi2010_dataframe,max_seq_length=max_seq_length, tokenizer= tokenizer , dataset_name = "ddi2010")
    #
    #     biosses_train_data = MyDataset(features=biosses_features,dataset_name="biosses")  # inherit from torch.utils.data.Dataset
    #     chemprot_train_data = MyDataset(features=chemprot_features,dataset_name="chemprot")
    #     mednli_train_data = MyDataset(features=mednli_features,dataset_name="mednli")
    #     hoc_train_data = MyDataset(features=hoc_features,dataset_name='hoc')
    #     ddi2010_train_data = MyDataset(features=ddi2010_features,dataset_name='ddi2010')
    #
    #     # # start training
    #     # trainer = train(base_model=model, data_Sets=[biosses_train_data, chemprot_train_data, mednli_train_data],
    #     #                 device='cuda', shuffle=True,task_num= task_num)
    #     trainer = train(base_model=model, data_Sets=[biosses_train_data,chemprot_train_data,mednli_train_data,hoc_train_data,ddi2010_train_data],batch_size=2,
    #                     device='cuda', shuffle=True, task_num=task_num,accumulate_iter=3)
    #     # trainer = train(base_model=model, data_Sets=[hoc_train_data],
    #     #                 batch_size=12,
    #     #                 device='cuda', shuffle=True, task_num=task_num, accumulate_iter=2)
    #     trainer.run(train_epoch)
    # elif do_train:
    #     trainer = train(base_model=None, data_Sets=[],
    #                     device='cuda', shuffle=True,task_num=task_num)
    #     trainer.run(train_epoch)
    #
    # # my_eval.evaluate_train(r"F:\NLP\NLP\blue_bert_master\models\model_out_flowback\model_2000_.pt")
    # # exit(1)
    # do_eval = True
    # models_in_file = os.listdir(model_dir)
    #
    # if do_eval and len(models_in_file)>0:
    #     models_in_file.sort(key=findRecentModel)
    #     latest_model = os.path.join(model_dir, models_in_file[-1])
    #     # start data preparation
    #
    #     biosses_dataframe = loadData(dataset_name="biosses", data_type="test")  # load biosses data
    #     biosses_features = get_features(df=biosses_dataframe, max_seq_length=max_seq_length, tokenizer=tokenizer,
    #                                     dataset_name="biosses")
    #
    #     chemprot_dataframe = loadData(dataset_name="chemprot", data_type="test")
    #     chemprot_features = get_features(df=chemprot_dataframe, max_seq_length=max_seq_length, tokenizer=tokenizer,
    #                                      dataset_name="chemprot")
    #
    #     mednli_dataframe = loadData(dataset_name="mednli", data_type="test")
    #     mednli_features = get_features(df=mednli_dataframe, max_seq_length=max_seq_length, tokenizer=tokenizer,
    #                                    dataset_name="mednli")
    #
    #     hoc_dataframe = loadData(dataset_name='hoc', data_type='test')
    #     hoc_features = get_features(df=hoc_dataframe, max_seq_length=max_seq_length, tokenizer=tokenizer,
    #                                 dataset_name="hoc")
    #
    #     ddi2010_dataframe = loadData(dataset_name='ddi2010', data_type='test')
    #     ddi2010_features = get_features(df=ddi2010_dataframe, max_seq_length=max_seq_length, tokenizer=tokenizer,
    #                                     dataset_name="ddi2010")
    #
    #     biosses_test_data = MyDataset(features=biosses_features,dataset_name="biosses")  # inherit from torch.utils.data.Dataset
    #     chemprot_test_data = MyDataset(features=chemprot_features, dataset_name="chemprot")
    #     mednli_test_data = MyDataset(features=mednli_features, dataset_name="mednli")
    #
    #     hoc_test_data = MyDataset(features=hoc_features,dataset_name="hoc")
    #
    #     ddi2010_test_data = MyDataset(features=ddi2010_features,dataset_name="ddi2010")
    #     # # start training
    #
    #     # for model in models_in_file:
    #     #     recent_model = os.path.join(model_dir,model)
    #     #     # evaluation(model_dir=recent_model, eval_dataSets=[biosses_test_data, chemprot_test_data, mednli_test_data])
    #     #     evaluation(model_dir=recent_model, eval_dataSets=[chemprot_test_data],device=device, batch_size=20)
    #     evaluation(model_dir=os.path.join(model_dir,models_in_file[-3]), eval_dataSets=[biosses_test_data, chemprot_test_data, mednli_test_data,hoc_test_data,ddi2010_test_data],device=device,batch_size=20)

    # end training


# if __name__ == '__main__':
#     main()