import wget
import tempfile
import os


def download():
    url = "http://mengchao.site/"
    dataset_names = ['biosses', 'chemprot', 'mednli', 'ddi2010', 'hoc']
    dataset_types = ['train.tsv', 'test.tsv', 'dev.tsv']

    dataset_path = os.path.join(os.path.abspath(__file__ + "/../"), 'datasets')
    for name in dataset_names:

        for tp in dataset_types:
            print("download...... ", name, " ", tp)
            target_path = os.path.join(dataset_path, name)
            os.makedirs(target_path, exist_ok=True)
            target = os.path.join(dataset_path, *[name, tp])
            single_url = url + name + '/' + tp
            if not os.path.exists(target_path):
                wget.download(single_url, out=target_path)
            print("finish download of", name, "/", tp)

    print("complete!")