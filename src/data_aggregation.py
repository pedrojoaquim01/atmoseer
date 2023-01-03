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
        
        if s in cor_est:
            df1 = df1.drop(columns=['Dia','Hora','estacao','HBV','data'])
        else:
            df1 = df1.drop(columns=['DC_NOME','UF','DT_MEDICAO','CD_ESTACAO','VL_LATITUDE','VL_LONGITUDE','HR_MEDICAO','data','showalter'])

        if not df1.empty:
            df_arr = np.array(df1)

            WS = 6 # size of window to use
            IDX_TARGET = 1 # index position of the target variable
            if s in cor_est:
                IDX_TARGET = df1.columns.get_loc("Chuva")
            else:
                IDX_TARGET = df1.columns.get_loc("CHUVA")


            X, y = apply_windowing(df_arr, 
                                    initial_time_step=0, 
                                    max_time_step=len(df_arr)-WS-1, 
                                    window_size = WS, 
                                    idx_target = IDX_TARGET,
                                    only_y_not_nan = True,
                                    only_y_gt_zero = True,
                                    only_X_not_nan = True)
            print(s)
            print(X)
            print(y)

            if len(X) > 0:
                df_aux['X_' + suf] =  pd.Series(X.tolist())
                df_aux['y_' + suf] =  pd.Series(y.tolist())
                #df_out= pd.DataFrame()
                outfile = open('../data/Janelamento_'+ s+'_X','wb')
                pickle.dump(X,outfile)
                
                
                outfile = open('../data/Janelamento_'+ s+'_y','wb')
                pickle.dump(y,outfile)

                infile = open('../data/Janelamento_'+ s+'_X','rb')
                #df_out['X'] =  pd.Series(X.tolist())
                #df_out['y'] =  pd.Series(y.tolist())
                #df_out.to_csv('../data/Forte_Copacabana - ' + s + '.csv')
                new_dict = pickle.load(infile)
                print(new_dict)

            #df1 = df1.add_suffix('_' + suf)
            #df1 = df1.rename(columns={"data_" + suf : "data"})
            
        # df_aux = pd.merge(df_aux,df1, how = 'outer')
            
            #df_aux = df_aux.sort_values(by=['DT_MEDICAO','HR_MEDICAO'])

    if inic == 0 and fim == 0:
        df_final = df_aux[df_aux['data'].isin(df['data'])]
    else:
        df_final = df_aux

    print(df_final.head())
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