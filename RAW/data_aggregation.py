import pandas as pd
import numpy as np
import sys, getopt, os, re
from datetime import datetime
from math import cos, asin, sqrt
import numpy as np
import pickle

def apply_windowing(X, 
                    initial_time_step, 
                    max_time_step, 
                    window_size, 
                    idx_target,
                    only_y_not_nan = False,
                    only_y_gt_zero = False,
                    only_X_not_nan = False):

  assert idx_target >= 0 and idx_target < X.shape[1]
  assert initial_time_step >= 0
  assert max_time_step >= initial_time_step

  start = initial_time_step
    
  sub_windows = (
        start +
        # expand_dims converts a 1D array to 2D array.
        np.expand_dims(np.arange(window_size), 0) +
        np.expand_dims(np.arange(max_time_step + 1), 0).T
  )
  X_temp, y_temp = X[sub_windows], X[window_size:(max_time_step+window_size+1):1, idx_target]

  if only_y_not_nan and only_y_gt_zero and only_X_not_nan:
    y_train_not_nan_idx = np.where(~np.isnan(y_temp))[0]
    y_train_gt_zero_idx = np.where(y_temp>0)[0]
    x_train_is_nan_idx = np.unique(np.where(np.isnan(X_temp)))
    idxs = np.intersect1d(y_train_not_nan_idx, y_train_gt_zero_idx)
    idxs = np.setdiff1d(idxs, x_train_is_nan_idx)
    X_temp, y_temp = X_temp[idxs], y_temp[idxs]

  return X_temp, y_temp

cor_est = ['alto_da_boa_vista','guaratiba','iraja','jardim_botanico','riocentro','santa_cruz','sao_cristovao','vidigal']

def myFunc2(e):
  return e[:][1]

def prox(nome, num):
  lugar = []
  result = []
  aux1 = pd.read_csv('../data/estacoes_local.csv')
  aux = aux1[~aux1['files'].isin([nome])]
  del aux['Unnamed: 0']
  for loc in aux['DC_NOME']:
      p = 0.017453292519943295
      alvo = aux1[aux1['files'] == nome]
      est = aux[aux['DC_NOME'] == loc]

      hav = 0.5 - cos((est.VL_LATITUDE.iloc[0]-alvo.VL_LATITUDE.iloc[0])*p)/2 + cos(alvo.VL_LATITUDE.iloc[0]*p)*cos(est.VL_LATITUDE.iloc[0]*p) * (1-cos((est.VL_LONGITUDE.iloc[0]-alvo.VL_LONGITUDE.iloc[0])*p)) / 2
      
      dist = 12742 * asin(sqrt(hav))
      lugar = lugar + [(est.files.iloc[0],dist)]
      lugar.sort(key=myFunc2) 
      lugar = lugar[0:num]  
      result = [i[0] for i in lugar]
  print(result)
  return result

def aggregation(f, n, inic, fim):
    df = pd.read_csv('../data/' + f + '.csv')
    df['data'] = pd.to_datetime(df['DT_MEDICAO'] + ' '+ df['HR_MEDICAO'].apply(lambda x: '{0:0>4}'.format(x)).str.slice(0, 2) + ':' + df['HR_MEDICAO'].apply(lambda x: '{0:0>4}'.format(x)).str.slice(2, 4) + ':00', format='%Y-%m-%d%H:%M:%S', infer_datetime_format=True)
    del df['Unnamed: 0']    
    df_aux = df
    if inic != 0 and fim != 0:    
        mask = (df['data'] > inic + '-1-1') & (df['data'] <= fim + '-1-1')
        df_aux = df.loc[mask].reset_index(drop=True)
    else:
        df_aux = df.reset_index(drop=True)

        
    n = len(df)
    train_df = df[0:int(n*0.7)]
    val_df = df[int(n*0.7):int(n*0.9)]
    test_df = df[int(n*0.9):]

    result = prox('RIO DE JANEIRO - FORTE DE COPACABANA_1997_2022',int(n))
    count = 0
    for s in result:
        
        if s in cor_est:
            fonte = '../data/'+s+'.csv'
        else:
            fonte = '../data/'+s+'_ERA5_RAD_VENT_TEMP.csv'
        df1 = pd.read_csv(fonte)
        #df1 = df1.fillna(0)
        del df1['Unnamed: 0']
        
        if s in cor_est:
            df1['data'] = pd.to_datetime(df1['Dia'] +' '+ df1['Hora'], format='%Y-%m-%d%H:%M:%S', infer_datetime_format=True)
        else:
            df1['data'] = pd.to_datetime(df1['DT_MEDICAO'] + ' '+ df1['HR_MEDICAO'].apply(lambda x: '{0:0>4}'.format(x)).str.slice(0, 2) + ':' + df1['HR_MEDICAO'].apply(lambda x: '{0:0>4}'.format(x)).str.slice(2, 4) + ':00', format='%Y-%m-%d%H:%M:%S', infer_datetime_format=True)
        
        if inic != 0 and fim != 0:    
            mask = (df1['data'] > inic + '-1-1') & (df1['data'] <= fim + '-1-1')
            df1 = df1.loc[mask].reset_index(drop=True)
        

        suf = str(count)
        count += 1
        
        #df1 = df1[df1['data'].isin(df['data'])]
        
        
        train_df1 = df1[df1['data'].isin(train_df['data'])]
        val_df1 = df1[df1['data'].isin(val_df['data'])]
        test_df1 =  df1[df1['data'].isin(test_df['data'])]

        print(df1)
        if s in cor_est:
            df1['CHUVA'] = df1['Chuva']
            df1['VEN_DIR'] = df1['DirVento']
            df1['VEN_VEL'] = df1['VelVento']
            df1['TEM_INS'] = df1['Temperatura']
            df1['PRE_INS'] = df1['Pressao']
            df1['UMD_INS'] = df1['Umidade']


            df1 = df1.drop(columns=['Dia','Hora','estacao','HBV','Chuva','DirVento','VelVento','Temperatura','Pressao','Umidade'])
            
            train_df1 = train_df1.drop(columns=['Dia','Hora','estacao','HBV','Chuva','DirVento','VelVento','Temperatura','Pressao','Umidade'])
            val_df1 = val_df1.drop(columns=['Dia','Hora','estacao','HBV','Chuva','DirVento','VelVento','Temperatura','Pressao','Umidade'])
            test_df1 =  test_df1.drop(columns=['Dia','Hora','estacao','HBV','Chuva','DirVento','VelVento','Temperatura','Pressao','Umidade'])

        else:
            df1 = df1.drop(columns=['DC_NOME','UF','DT_MEDICAO','CD_ESTACAO','VL_LATITUDE','VL_LONGITUDE','HR_MEDICAO','TEM_SEN','PRE_MAX','RAD_GLO','PTO_INS','TEM_MIN','UMD_MIN','PTO_MAX','PRE_MIN','UMD_MAX','PTO_MIN','TEM_MAX','TEN_BAT','VEN_RAJ','TEM_CPU'])
    
            train_df1 = train_df1.drop(columns=['DC_NOME','UF','DT_MEDICAO','CD_ESTACAO','VL_LATITUDE','VL_LONGITUDE','HR_MEDICAO','TEM_SEN','PRE_MAX','RAD_GLO','PTO_INS','TEM_MIN','UMD_MIN','PTO_MAX','PRE_MIN','UMD_MAX','PTO_MIN','TEM_MAX','TEN_BAT','VEN_RAJ','TEM_CPU'])
            val_df1 = val_df1.drop(columns=['DC_NOME','UF','DT_MEDICAO','CD_ESTACAO','VL_LATITUDE','VL_LONGITUDE','HR_MEDICAO','TEM_SEN','PRE_MAX','RAD_GLO','PTO_INS','TEM_MIN','UMD_MIN','PTO_MAX','PRE_MIN','UMD_MAX','PTO_MIN','TEM_MAX','TEN_BAT','VEN_RAJ','TEM_CPU'])
            test_df1 =  test_df1.drop(columns=['DC_NOME','UF','DT_MEDICAO','CD_ESTACAO','VL_LATITUDE','VL_LONGITUDE','HR_MEDICAO','TEM_SEN','PRE_MAX','RAD_GLO','PTO_INS','TEM_MIN','UMD_MIN','PTO_MAX','PRE_MIN','UMD_MAX','PTO_MIN','TEM_MAX','TEN_BAT','VEN_RAJ','TEM_CPU'])
        
        train_df = pd.concat([train_df,train_df1], ignore_index=True)
        train_df.sort_values(by='data', inplace = True)
        val_df = pd.concat([val_df,val_df1], ignore_index=True)
        val_df.sort_values(by='data', inplace = True)

    print('Treinamento: ')
    print(train_df)
    print('Validação: ')
    print(val_df)
    if inic == 0 and fim == 0:
        df_final = df_aux[df_aux['data'].isin(df['data'])]
    else:
        df_final = df_aux

    del df_final['data']
    df_final.to_csv('../data/'+ f + '_' + str(count) + '.csv')    


def myfunc(argv):
    arg_file = ""
    arg_num = 0
    arg_inic = 0
    arg_fim = 0
    
    arg_help = "{0} -f <file> -n <number> -b <begin> -e <end>".format(argv[0])
    
    try:
        opts, args = getopt.getopt(argv[1:], "hf:n:b:e:", ["help", "file=", "number=", "begin=", "end="])
    except:
        print(arg_help)
        sys.exit(2)
    
    for opt, arg in opts:
        if opt in ("-h", "--help"):
            print(arg_help)  # print the help message
            sys.exit(2)
        elif opt in ("-f", "--file"):
            arg_file = arg
        elif opt in ("-n", "--number"):
            arg_num = arg
        elif opt in ("-b", "--begin"):
            arg_inic = arg
        elif opt in ("-e", "--end"):
            arg_fim = arg
    aggregation(arg_file,arg_num,arg_inic,arg_fim)


if __name__ == "__main__":
    myfunc(sys.argv)