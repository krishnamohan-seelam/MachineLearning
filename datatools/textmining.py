import pandas as pd 
import os
from mlsettings.settings import load_app_config, get_datafolder_path
import pyprind
import numpy as np

def load_imdb():
    load_app_config()
    input_path = get_datafolder_path()
    directory ="aclImdb"
    imdb_path = os.path.join(input_path,directory)
    outfile = os.path.join(imdb_path,"movie_data.csv")
    print("input files path:{0}".format(imdb_path))
    print("output file path:{0}".format(outfile))
    pbar = pyprind.ProgBar(50000)
    labels = {'pos': 1, 'neg': 0}
    df = pd.DataFrame()
    for s in ('test', 'train'):
        for l in ('pos', 'neg'):
            path = os.path.join(imdb_path, s, l)
            for file in os.listdir(path):
                with open(os.path.join(path, file),'r', encoding='utf-8') as infile:
                    txt = infile.read()
                    df = df.append([[txt, labels[l]]],ignore_index=True)
                    pbar.update()
    df.columns = ['review', 'sentiment']
    df = df.reindex(np.random.permutation(df.index))
    df.to_csv(outfile, index=False, encoding='utf-8')
def main():
    load_imdb()


if __name__ == '__main__':
    main()