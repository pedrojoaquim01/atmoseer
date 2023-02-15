
import pandas as pd
import numpy as np
from Utils.near_stations import prox

cor_est = ['alto_da_boa_vista','guaratiba','iraja','jardim_botanico','riocentro','santa_cruz','sao_cristovao','vidigal']

def apply_windowing(X, 
                    initial_time_step, 
                    max_time_step, 
                    window_size, 
                    idx_target,
                    only_y_not_nan = True,
                    only_y_gt_zero = True,
                    only_X_not_nan = True):

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

def generate_windowed_split(df, id_target = 'CHUVA', window_size = 6, stations = 0, aux_nome = ''):
  n = len(df)
  train_df = df[0:int(n*0.7)]
  val_df = df[int(n*0.7):int(n*0.9)]
  test_df = df[int(n*0.9):]

  if stations != 0:
    df = df.drop(columns=['TEM_SEN','PRE_MAX','RAD_GLO','PTO_INS','TEM_MIN','UMD_MIN','PTO_MAX','PRE_MIN','UMD_MAX','PTO_MIN','TEM_MAX','TEN_BAT','VEN_RAJ','TEM_CPU'])
    train_df = train_df.drop(columns=['TEM_SEN','PRE_MAX','RAD_GLO','PTO_INS','TEM_MIN','UMD_MIN','PTO_MAX','PRE_MIN','UMD_MAX','PTO_MIN','TEM_MAX','TEN_BAT','VEN_RAJ','TEM_CPU'])
    val_df = val_df.drop(columns=['TEM_SEN','PRE_MAX','RAD_GLO','PTO_INS','TEM_MIN','UMD_MIN','PTO_MAX','PRE_MIN','UMD_MAX','PTO_MIN','TEM_MAX','TEN_BAT','VEN_RAJ','TEM_CPU'])
    test_df = test_df.drop(columns=['TEM_SEN','PRE_MAX','RAD_GLO','PTO_INS','TEM_MIN','UMD_MIN','PTO_MAX','PRE_MIN','UMD_MAX','PTO_MIN','TEM_MAX','TEN_BAT','VEN_RAJ','TEM_CPU'])
    
    result = prox('RIO DE JANEIRO - FORTE DE COPACABANA_1997_2022',int(stations))
    count = 0
    sufix = aux_nome.replace('RIO DE JANEIRO - FORTE DE COPACABANA_1997_2022','')
    for s in result:
        
        if s in cor_est:
            fonte = '../data/'+ s + sufix +'.csv'
        else:
            fonte = '../data/'+ s + sufix +'.csv'
        df1 = pd.read_csv(fonte)
        df1 = df1.fillna(0)
        del df1['Unnamed: 0']
        
        if s in cor_est:
            df1['data'] = pd.to_datetime(df1['Dia'] +' '+ df1['Hora'], format='%Y-%m-%d%H:%M:%S', infer_datetime_format=True)
        else:
            df1['data'] = pd.to_datetime(df1['DT_MEDICAO'] + ' '+ df1['HR_MEDICAO'].apply(lambda x: '{0:0>4}'.format(x)).str.slice(0, 2) + ':' + df1['HR_MEDICAO'].apply(lambda x: '{0:0>4}'.format(x)).str.slice(2, 4) + ':00', format='%Y-%m-%d%H:%M:%S', infer_datetime_format=True)

        suf = str(count)
        count += 1
        
        #df1 = df1[df1['data'].isin(df['data'])]
        
        print(df1)
        try:
          train_df1 = df1[df1['data'].isin(train_df['data'])]
        except:
          continue
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
        train_df = train_df.fillna(0)
        val_df = pd.concat([val_df,val_df1], ignore_index=True)
        val_df.sort_values(by='data', inplace = True)
        val_df = val_df.fillna(0)
        
  
  if stations != 0:
    train_df = train_df.drop(columns=['data'])
    val_df = val_df.drop(columns=['data'])
    test_df = test_df.drop(columns=['data'])
  train_arr = np.array(train_df)
  val_arr = np.array(val_df)
  test_arr = np.array(test_df)

  idx_target = train_df.columns.get_loc(id_target)
  print(idx_target)

  TIME_WINDOW_SIZE = window_size
  IDX_TARGET = id_target
      
  X_train, y_train = apply_windowing(train_arr, 
                                    initial_time_step=0, 
                                    max_time_step=len(train_arr)-TIME_WINDOW_SIZE-1, 
                                    window_size = TIME_WINDOW_SIZE, 
                                    idx_target = idx_target)
  y_train = y_train.reshape(-1,1)

  X_val, y_val = apply_windowing(val_arr, 
                                initial_time_step=0, 
                                max_time_step=len(val_arr)-TIME_WINDOW_SIZE-1, 
                                window_size = TIME_WINDOW_SIZE, 
                                idx_target = idx_target)
  y_val = y_val.reshape(-1,1)

  X_test, y_test = apply_windowing(test_arr, 
                                  initial_time_step=0, 
                                  max_time_step=len(test_arr)-TIME_WINDOW_SIZE-1, 
                                  window_size = TIME_WINDOW_SIZE, 
                                  idx_target = idx_target)
  y_test = y_test.reshape(-1,1)

  return X_train, y_train, X_val, y_val, X_test, y_test
