#%%
import pandas as pd
from constants import INMET_PATH
from glob import glob


csv_paths = glob(INMET_PATH+'csvs/*.csv')

def read_inmet_csv(fpath):

    df = pd.read_csv(fpath, sep=';', decimal=',', header=9)
    if df.empty:
        return
    df = df.rename({'Data Medicao': 'date', 'TEMPERATURA MAXIMA, DIARIA(°C)': 'tmax'},
                axis=1)
    sr = df.set_index('date')['tmax'].squeeze()
    

    if sr.isna().sum() == len(sr):
        return


    meta = pd.read_csv(fpath,
                    sep=':', decimal=',', header=None, nrows=9, index_col=0).T
    
    meta = meta.rename({'Nome': 'name', 'Codigo Estacao': 'station_id',
                        'Latitude':'lat', 'Longitude':'lon', 'Alititude':'elevation',
                        'Data Inicial':'start_date', 'Data Final':'end_date'}, axis=1)
    meta['station_id'] = meta['station_id'].astype(str).str.strip()

    meta = meta.set_index('station_id').T.squeeze()
    sr.name = meta.name
    return sr, meta

def load_inmet_data():

    dfs = []
    metas = []
    for fpath in csv_paths:
        res = read_inmet_csv(fpath)
        if res is not None:
            sr, meta = res
            dfs.append(sr)
            metas.append(meta)

    df = pd.DataFrame(dfs).T
    meta = pd.DataFrame(metas)
    meta.index.name = 'station_id'
    return df, meta
