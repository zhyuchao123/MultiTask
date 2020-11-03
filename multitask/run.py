import torch
import logging
import os
# for idle
# from multitask.my_blue_classifier_20200930 import AutoTokenizer,loadmodel,loadData,get_features,MyDataset,train,evaluation
# for terminal
from my_blue_classifier_20200930 import AutoTokenizer,loadmodel,loadData,get_features,MyDataset,train,evaluation,InputFeatures
from module.task_heads import multi_task_model,MultiTaskLossWrapper


def main():
    # import torch
    #
    # ck = torch.load(r'F:\multiTask\multitask\models\MultiTaskModels\model_35_.pt')
    #
    # print(ck)

    torch.set_num_threads(6)
    logging.info(torch.__config__.parallel_info())
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dir = os.path.abspath(os.path.dirname(__file__) )
    model_dir = os.path.abspath(os.path.dirname(__file__))
    model_dir = os.path.join(model_dir, r"models\model_out")
    # model_dir = os.path.join(model_dir, r"models\chemprot02")

    models_in_file = os.listdir(model_dir)
    train_epoch = 1000
    task_num = 5  # define the number of tasks

    def findRecentModel(name):
        return int(name.split('_')[1])

    max_seq_length = 256  # define the maximum length for input

    # load online tokenizer
    tokenizer = AutoTokenizer.from_pretrained("dmis-lab/biobert-v1.1")

    do_train = True
    if len(models_in_file) == 0 and do_train:
        # if do_train:
        """
        The model has not been trained(determined by detecting the model in the file)
        """
        model = loadmodel(from_net=True, frozen=False, task_num=task_num)  # load default model

        # start data preparation
        # biosses_dataframe = loadData(dataset_name="biosses", data_type="train")  # load biosses data
        # biosses_features = get_features(df=biosses_dataframe, max_seq_length=max_seq_length, tokenizer=tokenizer,
        #                                 dataset_name="biosses")

        # chemprot_dataframe = loadData(dataset_name="chemprot", data_type="train")
        # chemprot_features = get_features(df=chemprot_dataframe, max_seq_length=max_seq_length, tokenizer=tokenizer,
        #                                  dataset_name="chemprot")
        #
        # mednli_dataframe = loadData(dataset_name="mednli", data_type="train")
        # mednli_features = get_features(df=mednli_dataframe, max_seq_length=max_seq_length, tokenizer=tokenizer,
        #                                dataset_name="mednli")

        hoc_dataframe = loadData(dataset_name='hoc', data_type='train')
        hoc_features = get_features(df=hoc_dataframe, max_seq_length=max_seq_length, tokenizer=tokenizer,
                                    dataset_name="hoc")
        #
        # ddi2010_dataframe = loadData(dataset_name='ddi2010', data_type='train')
        # ddi2010_features = get_features(df=ddi2010_dataframe, max_seq_length=max_seq_length, tokenizer=tokenizer,
        #                                 dataset_name="ddi2010")

        # biosses_train_data = MyDataset(features=biosses_features,
        #                                dataset_name="biosses")  # inherit from torch.utils.data.Dataset
        # chemprot_train_data = MyDataset(features=chemprot_features, dataset_name="chemprot")
        # mednli_train_data = MyDataset(features=mednli_features, dataset_name="mednli")
        hoc_train_data = MyDataset(features=hoc_features, dataset_name='hoc')
        # ddi2010_train_data = MyDataset(features=ddi2010_features, dataset_name='ddi2010')

        # # start training
        # trainer = train(base_model=model, data_Sets=[chemprot_train_data],
        #                 device='cuda', shuffle=True,task_num= task_num)
        # trainer = train(base_model=model,
        #                 data_Sets=[biosses_train_data, chemprot_train_data, mednli_train_data, hoc_train_data,
        #                            ddi2010_train_data], batch_size=2,
        #                 device='cuda', shuffle=True, task_num=task_num, accumulate_iter=3)
        trainer = train(base_model=model, data_Sets=[hoc_train_data],
                        batch_size=12,
                        device='cuda', shuffle=True, task_num=task_num, accumulate_iter=2)
        trainer.run(train_epoch)
    elif do_train:
        base_model = loadmodel(from_net=True, frozen=False, task_num=task_num)  # load default model
        biosses_dataframe = loadData(dataset_name="biosses", data_type="train")  # load biosses data
        biosses_features = get_features(df=biosses_dataframe, max_seq_length=max_seq_length, tokenizer=tokenizer,
                                        dataset_name="biosses")
        biosses_train_data = MyDataset(features=biosses_features,
                                       dataset_name="biosses")  # inherit from torch.utils.data.Dataset
        trainer = train(base_model=base_model, data_Sets=[],
                        device='cuda', shuffle=True, task_num=1)
        trainer.run(train_epoch)

    # my_eval.evaluate_train(r"F:\NLP\NLP\blue_bert_master\models\model_out_flowback\model_2000_.pt")
    # exit(1)
    do_eval = True
    models_in_file = os.listdir(model_dir)

    if do_eval and len(models_in_file) > 0:
        models_in_file.sort(key=findRecentModel)
        latest_model = os.path.join(model_dir, models_in_file[-1])
        # start data preparation
        #
        biosses_dataframe = loadData(dataset_name="biosses", data_type="test")  # load biosses data
        biosses_features = get_features(df=biosses_dataframe, max_seq_length=max_seq_length, tokenizer=tokenizer,
                                        dataset_name="biosses")

        chemprot_dataframe = loadData(dataset_name="chemprot", data_type="test")
        chemprot_features = get_features(df=chemprot_dataframe, max_seq_length=max_seq_length, tokenizer=tokenizer,
                                         dataset_name="chemprot")

        mednli_dataframe = loadData(dataset_name="mednli", data_type="test")
        mednli_features = get_features(df=mednli_dataframe, max_seq_length=max_seq_length, tokenizer=tokenizer,
                                       dataset_name="mednli")

        hoc_dataframe = loadData(dataset_name='hoc', data_type='test')
        hoc_features = get_features(df=hoc_dataframe, max_seq_length=max_seq_length, tokenizer=tokenizer,
                                    dataset_name="hoc")

        ddi2010_dataframe = loadData(dataset_name='ddi2010', data_type='test')
        ddi2010_features = get_features(df=ddi2010_dataframe, max_seq_length=max_seq_length, tokenizer=tokenizer,
                                        dataset_name="ddi2010")

        biosses_test_data = MyDataset(features=biosses_features,dataset_name="biosses")  # inherit from torch.utils.data.Dataset
        chemprot_test_data = MyDataset(features=chemprot_features, dataset_name="chemprot")
        mednli_test_data = MyDataset(features=mednli_features, dataset_name="mednli")

        hoc_test_data = MyDataset(features=hoc_features, dataset_name="hoc")

        ddi2010_test_data = MyDataset(features=ddi2010_features, dataset_name="ddi2010")
        # # start training
        base_model = loadmodel(from_net=True, frozen=False, task_num=task_num)
        initial_model = MultiTaskLossWrapper(model=base_model, task_num=task_num, fine_tune=False)
        for model in models_in_file:

            recent_model_dict = os.path.join(model_dir,model)
            evaluation(model=initial_model,model_dict_dir=recent_model_dict, eval_dataSets=[biosses_test_data, chemprot_test_data, mednli_test_data,ddi2010_test_data,hoc_test_data],batch_size=20,device='cuda')
            # evaluation(model=initial_model,model_dict_dir=recent_model_dict, eval_dataSets=[hoc_test_data],device=device, batch_size=20)
        # evaluation(model_dir=os.path.join(model_dir, models_in_file[-3]),
        #            eval_dataSets=[biosses_test_data, chemprot_test_data, mednli_test_data, hoc_test_data,
        #                           ddi2010_test_data], device=device, batch_size=20)
    do_dev = False
    if do_dev and len(models_in_file) > 0:
        models_in_file.sort(key=findRecentModel)
        latest_model = os.path.join(model_dir, models_in_file[-1])
        # start data preparation

        biosses_dataframe = loadData(dataset_name="biosses", data_type="dev")  # load biosses data
        biosses_features = get_features(df=biosses_dataframe, max_seq_length=max_seq_length, tokenizer=tokenizer,
                                        dataset_name="biosses")
        biosses_test_data = MyDataset(features=biosses_features,
                                      dataset_name="biosses")  # inherit from torch.utils.data.Dataset
        for model in models_in_file:
            recent_model = os.path.join(model_dir, model)
            # evaluation(model_dir=recent_model, eval_dataSets=[biosses_test_data, chemprot_test_data, mednli_test_data])
            evaluation(model_dir=recent_model, eval_dataSets=[biosses_test_data], device=device, batch_size=20)
if __name__ == '__main__':
    main()