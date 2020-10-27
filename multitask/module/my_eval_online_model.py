import torch
import os
import time
import numpy as np
from logger import logger
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import classification_report,f1_score,accuracy_score,recall_score,precision_score
from tqdm import tqdm
import pandas as pd
from multitask.module.task_heads import MultiTaskLossWrapper,multi_task_model # for model load
from sklearn.metrics import confusion_matrix

def confusion_matrix_to_pandas(data,labels):
    table = pd.DataFrame(
        data,
        columns=labels,
        index=labels)
    return table
def evaluation(model,model_dict_path, eval_dataSets, device="cuda", batch_size=10):
    """
    to evaluation the model
    input paras:
    model_path: the path for a model
    eval_dataSets: a list of evaluation datasets in dataset form
    devi

    batch_size: batch size for evaluation, can adjust for faster evaluation
    Yuchao Zhang 2020.10.09
    """

    # environment setting for evaluation
    seed = 32
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


    report_path = os.path.join(os.path.abspath(os.path.join(os.getcwd(), "reports")))

    checkpoint = torch.load(model_dict_path)
    model_dict = checkpoint['model']
    model.load_state_dict(model_dict)
    model = model.eval()
    model = model.to(device)
    # base_model = model.get_multiTask_model().eval()
    data_set_Names = [data_set.get_dataset_name() for data_set in eval_dataSets]
    data_Loaders = [torch.utils.data.DataLoader(dataset=data_set, batch_size=batch_size, shuffle=False)
                    for data_set in eval_dataSets]
    model_name = model_dict_path.split("\\")[-1].split('.')[0]

    for dataset_name,data_Loaders in zip(data_set_Names,data_Loaders):
        targets = []
        predicts = []
        loops = tqdm(enumerate(data_Loaders),total=len(data_Loaders),leave=False)
        for ind,features in loops:
            with torch.no_grad():
                pred, target = model.evaluation_result(feature=features, dataset_name=dataset_name)

            # target = features['score']
            pred = pred.cpu().detach().numpy()
            target = target.cpu().detach().numpy()
            torch.cuda.empty_cache()
            predicts.extend(pred)
            targets.extend(target)
            loops.set_description(f"{dataset_name}:[{ind}/{len(data_Loaders)}]")
        # end of evaluation calculation
        # start to get finally result

        if dataset_name == "biosses":
            # coff_pearson = model.get_eval_loss(predicts,targets,dataset_name)
            coff_pearson = pearsonr(targets, predicts)[0]
            # model_name = model_path.split("\\")[-1].split('.')[0]
            logger.info("Pearson coefficiency of biosses is: {:.4f} {:s}".format(coff_pearson,model_name))
        elif dataset_name == "mednli":
            # coff = model.get_eval_loss(predicts,targets,dataset_name)
            # coff = (np.array(predicts)==np.array(targets)).sum()/len(targets)
            coff = accuracy_score(targets,predicts)
            logger.info("Accuracy of mednli is: {:.4f}".format(coff))
            report = classification_report(targets,predicts,target_names=["entailment", "neutral", "contradiction"])
            cmatrics = confusion_matrix(targets,predicts)
            mtx = confusion_matrix_to_pandas(cmatrics,["entailment", "neutral", "contradiction"])
            with open(os.path.join(report_path,model_name+"mednli.txt"),'w') as f:
                f.writelines(str(report))
                f.writelines(str(mtx))
        elif dataset_name == "chemprot":
            # coff = model.get_eval_loss(predicts,targets,dataset_name)
            #
            # logger.info("micro_f1 of chemprot is: {:.4f}".format(coff))
            coff = f1_score(targets,predicts,average="micro")
            logger.info("micro_f1 of chemprot is: {:.4f}".format(coff))
            report = classification_report(targets,predicts,target_names=["false","CPR:3","CPR:4","CPR:5","CPR:6","CPR:9"])
            cmatrics = confusion_matrix(targets,predicts)
            mtx = confusion_matrix_to_pandas(cmatrics,["false","CPR:3","CPR:4","CPR:5","CPR:6","CPR:9"])
            with open(os.path.join(report_path,model_name+"chemprot.txt"),'w') as f:
                f.writelines(str(report))
                f.writelines(str(mtx))

        elif dataset_name == "hoc":
            # to merge multiple numpy matrix to one
            targets = np.vstack(targets)
            predicts = np.vstack(predicts)

            coff = f1_score(targets, predicts, average="micro")
            logger.info("micro_f1 of hoc is: {:.4f}".format(coff))
            report = classification_report(targets,predicts,target_names=['None','sustaining proliferative signaling', 'genomic instability and mutation', 'resisting cell death', 'tumor promoting inflammation', 'enabling replicative immortality', 'cellular energetics', 'inducing angiogenesis', 'evading growth suppressors', 'activating invasion and metastasis', 'avoiding immune destruction'])
            with open(os.path.join(report_path,model_name+"hoc.txt"),'w') as f:
                f.writelines(str(report))
        elif dataset_name == "ddi2010":
            coff = f1_score(targets, predicts, average="micro")
            logger.info("micro_f1 of ddi2010 is: {:.4f}".format(coff))
            report = classification_report(targets, predicts,
                                           target_names=['DDI-false', 'DDI-mechanism', 'DDI-effect', 'DDI-advise', 'DDI-int'])
            cmatrics = confusion_matrix(targets, predicts)
            mtx = confusion_matrix_to_pandas(cmatrics, ['DDI-false', 'DDI-mechanism', 'DDI-effect', 'DDI-advise', 'DDI-int'])
            with open(os.path.join(report_path, model_name + "ddi2010.txt"), 'w') as f:
                f.writelines(str(report))
                f.writelines(str(mtx))
