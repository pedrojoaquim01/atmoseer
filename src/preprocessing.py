import pandas as pd
import numpy as np
from pathlib import Path
from math import cos, asin, sqrt
import sys
import getopt
import xarray as xr
from utils.near_stations import prox
from datetime import datetime, timedelta

def pre_proc(arquivo, use_sounding_as_data_source = 0, use_numerical_model_as_data_source = 0, log_Vento = 1, log_Tempo = 1, num_neighbors = 0):

    arq_pre_proc = arquivo + '_E'
    if use_numerical_model_as_data_source:
        arq_pre_proc = arq_pre_proc + '-N'
    if use_sounding_as_data_source:
        arq_pre_proc = arq_pre_proc + '-R'
    if(num_neighbors > 0):
        arq_pre_proc = arq_pre_proc + '_EI+' + str(num_neighbors) + 'NN'
    else:
        arq_pre_proc = arq_pre_proc + '_EI'

    dado_proc = Path('../data/' + arq_pre_proc + '.csv')

    if dado_proc.is_file():
        fonte = '../data/' + arq_pre_proc + '.csv'
        df = pd.read_csv(fonte)
        if 'Unnamed: 0' in df.columns:
            del df['Unnamed: 0']
        df = df.fillna(0)
        df = df.reindex(sorted(df.columns), axis=1)
        return df
    else:
        fonte = '../data/' + arquivo + '.csv'
        df = pd.read_csv(fonte)
        if 'Unnamed: 0' in df.columns:
            del df['Unnamed: 0']
        df = df.fillna(0)
        df = df.reindex(sorted(df.columns), axis=1)

        cor_est = ['alto_da_boa_vista','guaratiba','iraja','jardim_botanico','riocentro','santa_cruz','sao_cristovao','vidigal']
        
        if use_numerical_model_as_data_source:
            ano = list(map(str,range(1998,2022)))           
            ds = xr.open_dataset('../data/ERA-5/RJ_1997.nc')
            for i in ano:
                ds_aux = xr.open_dataset('../data/ERA-5/RJ_'+ i +'.nc')
                ds = ds.merge(ds_aux) 
            
            ano2 = [['1997','1998'],['1999','2000'],['2001','2002'],['2003','2004'],['2005','2006'],['2007','2008'],['2009','2010'],['2011','2012'],['2013','2014'],['2015','2016'],['2017','2018'],['2019','2020']]
            ds2 = xr.open_dataset('../data/ERA-5/RJ_2021_200.nc')
            for i in ano2:
                ds_aux2 = xr.open_dataset('../data/ERA-5/RJ_'+ i[0]+'_'+i[1] +'_200.nc')
                ds2 = ds2.merge(ds_aux2)

            df_estacoes = pd.read_csv('../data/estacoes_local.csv')
            df_estacoes = df_estacoes[df_estacoes['files'] == arquivo]
            latitude_aux = df_estacoes['VL_LATITUDE'].iloc[0]
            
            longitude_aux = df_estacoes['VL_LONGITUDE'].iloc[0]

            test = ds.sel(level = 1000, longitude = longitude_aux, latitude = latitude_aux, method = 'nearest')
            test2 = ds.sel(level = 700, longitude = longitude_aux, latitude = latitude_aux, method = 'nearest')
            test3 = ds2.sel(longitude = longitude_aux, latitude = latitude_aux, method = 'nearest')
            
            df_era = pd.DataFrame({'time': test.time,'Geopotential_1000': test.z, 'Humidity_1000': test.r,'Temperature_1000': test.t, 'WindU_1000': test.u, 'WindV_1000': test.v,'Geopotential_700': test2.z, 'Humidity_700': test2.r,'Temperature_700': test2.t, 'WindU_700': test2.u, 'WindV_700': test2.v,'Geopotential_200': test3.z, 'Humidity_200': test3.r,'Temperature_200': test3.t, 'WindU_200': test3.u, 'WindV_200': test3.v})
            
            if arquivo in cor_est:
                df['time'] = pd.to_datetime(df['Dia'] +' '+ df['Hora'], format='%Y-%m-%d%H:%M:%S', infer_datetime_format=True)
            else:
                df['time'] = pd.to_datetime(df['DT_MEDICAO'] + ' '+ df['HR_MEDICAO'].apply(lambda x: '{0:0>4}'.format(x)).str.slice(0, 2) + ':' + df['HR_MEDICAO'].apply(lambda x: '{0:0>4}'.format(x)).str.slice(2, 4) + ':00', format='%Y-%m-%d%H:%M:%S', infer_datetime_format=True)
            
            df_era['time'] = pd.to_datetime(df_era['time'].astype(str))
            df_era['time'] = df_era['time'] - timedelta(hours=3)

            df = df.merge(df_era,on='time',how='left')

            df['Geopotential_1000'] = df['Geopotential_1000'].interpolate(method='linear')
            df['Humidity_1000'] = df['Humidity_1000'].interpolate(method='linear')
            df['Temperature_1000'] = df['Temperature_1000'].interpolate(method='linear')
            df['WindU_1000'] = df['WindU_1000'].interpolate(method='linear')
            df['WindV_1000'] = df['WindV_1000'].interpolate(method='linear')
            
            df['Geopotential_700'] = df['Geopotential_700'].interpolate(method='linear')
            df['Humidity_700'] = df['Humidity_700'].interpolate(method='linear')
            df['Temperature_700'] = df['Temperature_700'].interpolate(method='linear')
            df['WindU_700'] = df['WindU_700'].interpolate(method='linear')
            df['WindV_700'] = df['WindV_700'].interpolate(method='linear')

            df['Geopotential_200'] = df['Geopotential_200'].interpolate(method='linear')
            df['Humidity_200'] = df['Humidity_200'].interpolate(method='linear')
            df['Temperature_200'] = df['Temperature_200'].interpolate(method='linear')
            df['WindU_200'] = df['WindU_200'].interpolate(method='linear')
            df['WindV_200'] = df['WindV_200'].interpolate(method='linear')

            df = df.fillna(0)
            del df['time']
            print('Log: Variavel ERA5 sem problema')

        if use_sounding_as_data_source:
            df_rs = pd.read_csv('../data/SBGL_indices_1997-01-01_2022-12-31.csv')
            df_rs['log_hr'] = ''
            df_rs['log_hr'] = df_rs['time'].map(lambda x: '0' if x[11:19] == '00:00:00' else '12')
           
            if arquivo in cor_est:
                df['time'] = pd.to_datetime(df['Dia'] +' '+ df['Hora'], format='%Y-%m-%d%H:%M:%S', infer_datetime_format=True)
            else:
                df['time'] = pd.to_datetime(df['DT_MEDICAO'] + ' '+ df['HR_MEDICAO'].apply(lambda x: '{0:0>4}'.format(x)).str.slice(0, 2) + ':' + df['HR_MEDICAO'].apply(lambda x: '{0:0>4}'.format(x)).str.slice(2, 4) + ':00', format='%Y-%m-%d%H:%M:%S', infer_datetime_format=True)
            
            df_rs_aux = df_rs[['time', 'CAPE', 'CIN', 'showalter', 'lift', 'k', 'total_totals']]
            df_rs_aux = df_rs_aux.drop_duplicates()
            df_rs_aux['time'] = pd.to_datetime(df_rs_aux['time'].astype(str))
            df_rs_aux['time'] = df_rs_aux['time'] + timedelta(hours=4)
            df = df.merge(df_rs_aux,on='time',how='left')
            
            df['CAPE'] = df['CAPE'].interpolate(method='linear')
            df['CIN'] = df['CIN'].interpolate(method='linear')
            df['showalter'] = df['showalter'].interpolate(method='linear')
            df['lift_index'] = df['lift_index'].interpolate(method='linear')
            df['k_index'] = df['k_index'].interpolate(method='linear')
            df['total_totals'] = df['total_totals'].interpolate(method='linear')
            df = df.fillna(0)
            del df['time']
            print('Log: Variavel CAPE sem problema')

        if log_Vento:
            if arquivo in cor_est:
                wv = df['VelVento'] / 3.6
                wd_rad = df['DirVento']*np.pi / 180

                df['Wx'] = wv*np.cos(wd_rad)
                df['Wy'] = wv*np.sin(wd_rad)
            else:
                wv = df['VEN_VEL']
                wd_rad = df['VEN_DIR']*np.pi / 180

                df['Wx'] = wv*np.cos(wd_rad)
                df['Wy'] = wv*np.sin(wd_rad)
        
        if log_Tempo:
            if arquivo in cor_est:
                date_time = pd.to_datetime(df['Dia'] +' '+ df['Hora'], format='%Y-%m-%d%H:%M:%S', infer_datetime_format=True)
            else:
                date_time = pd.to_datetime(df['DT_MEDICAO'] + ' '+ df['HR_MEDICAO'].apply(lambda x: '{0:0>4}'.format(x)).str.slice(0, 2) + ':' + df['HR_MEDICAO'].apply(lambda x: '{0:0>4}'.format(x)).str.slice(2, 4) + ':00', format='%Y-%m-%d%H:%M:%S', infer_datetime_format=True)
            
            timestamp_s = date_time.map(pd.Timestamp.timestamp)
            day = 24*60*60
            year = (365.2425)*day

            df['Day sin'] = np.sin(timestamp_s * (2 * np.pi / day))
            df['Day cos'] = np.cos(timestamp_s * (2 * np.pi / day))
            df['Year sin'] = np.sin(timestamp_s * (2 * np.pi / year))
            df['Year cos'] = np.cos(timestamp_s * (2 * np.pi / year))
            
        # AGREGAÇÃO E SEPARAÇÃO DOS DADOS
        cor_est = ['alto_da_boa_vista', 'guaratiba', 'iraja', 'jardim_botanico', 'riocentro', 'santa_cruz', 'sao_cristovao', 'vidigal']
        df = df.fillna(0)

        if arquivo in cor_est:
            if num_neighbors != 0:
                data_aux = pd.to_datetime(df['DT_MEDICAO'] + ' '+ df['HR_MEDICAO'].apply(lambda x: '{0:0>4}'.format(x)).str.slice(0, 2) + ':' + df['HR_MEDICAO'].apply(lambda x: '{0:0>4}'.format(x)).str.slice(2, 4) + ':00', format='%Y-%m-%d%H:%M:%S', infer_datetime_format=True)
        else:    
            if num_neighbors != 0:
                data_aux = pd.to_datetime(df['Dia'] + ' '+ df['Hora'].apply(lambda x: '{0:0>4}'.format(x)).str.slice(0, 2) + ':' + df['Hora'].apply(lambda x: '{0:0>4}'.format(x)).str.slice(2, 4) + ':00', format='%Y-%m-%d%H:%M:%S', infer_datetime_format=True)
        
        if arquivo in cor_est:
            df1 = df.drop(columns=['Dia','Hora','estacao','HBV'])
            col_target = 'Chuva'
        else:
            df1 = df.drop(columns=['DC_NOME','UF','DT_MEDICAO','CD_ESTACAO','VL_LATITUDE','VL_LONGITUDE','HR_MEDICAO'])
            col_target = 'CHUVA'


        assert df1.isnull().values.any() == False

        target = df1[col_target].copy()

        df2 = ((df1-df1.min())/(df1.max()-df1.min()))

        df2[col_target] = target

        df2 = df2.fillna(0)
        
        if num_neighbors != 0:
            df2['data'] = data_aux

        assert df2.isnull().values.any() == False

        n = len(df2)
        train_df = df2[0:int(n*0.7)]
        val_df = df2[int(n*0.7):int(n*0.9)]
        test_df = df2[int(n*0.9):]

        if num_neighbors != 0:
            if s not in cor_est:
                df2 = df2.drop(columns=['TEM_SEN','PRE_MAX','RAD_GLO','PTO_INS','TEM_MIN','UMD_MIN','PTO_MAX','PRE_MIN','UMD_MAX','PTO_MIN','TEM_MAX','TEN_BAT','VEN_RAJ','TEM_CPU'])
                train_df = train_df.drop(columns=['TEM_SEN','PRE_MAX','RAD_GLO','PTO_INS','TEM_MIN','UMD_MIN','PTO_MAX','PRE_MIN','UMD_MAX','PTO_MIN','TEM_MAX','TEN_BAT','VEN_RAJ','TEM_CPU'])
                val_df = val_df.drop(columns=['TEM_SEN','PRE_MAX','RAD_GLO','PTO_INS','TEM_MIN','UMD_MIN','PTO_MAX','PRE_MIN','UMD_MAX','PTO_MIN','TEM_MAX','TEN_BAT','VEN_RAJ','TEM_CPU'])
                test_df = test_df.drop(columns=['TEM_SEN','PRE_MAX','RAD_GLO','PTO_INS','TEM_MIN','UMD_MIN','PTO_MAX','PRE_MIN','UMD_MAX','PTO_MIN','TEM_MAX','TEN_BAT','VEN_RAJ','TEM_CPU'])
                
            result = prox(arquivo,int(num_neighbors))
            count = 0
            for s in result:
                fonte = '../data/'+ s +'.csv'
                df3 = pd.read_csv(fonte)
                df3 = df1.fillna(0)
                del df3['Unnamed: 0']
                
                if s in cor_est:
                    df3['data'] = pd.to_datetime(df3['Dia'] +' '+ df3['Hora'], format='%Y-%m-%d%H:%M:%S', infer_datetime_format=True)
                else:
                    df3['data'] = pd.to_datetime(df3['DT_MEDICAO'] + ' '+ df3['HR_MEDICAO'].apply(lambda x: '{0:0>4}'.format(x)).str.slice(0, 2) + ':' + df3['HR_MEDICAO'].apply(lambda x: '{0:0>4}'.format(x)).str.slice(2, 4) + ':00', format='%Y-%m-%d%H:%M:%S', infer_datetime_format=True)

                suf = str(count)
                count += 1
                
                try:
                    train_df3 = df3[df3['data'].isin(train_df['data'])]
                except:
                    continue

                val_df3 = df3[df3['data'].isin(val_df['data'])]
                test_df3 =  df3[df3['data'].isin(test_df['data'])]

                if s in cor_est:
                    df3['CHUVA'] = df3['Chuva']
                    df3['VEN_DIR'] = df3['DirVento']
                    df3['VEN_VEL'] = df3['VelVento'] / 3.6
                    df3['TEM_INS'] = df3['Temperatura']
                    df3['PRE_INS'] = df3['Pressao']
                    df3['UMD_INS'] = df3['Umidade']

                    df3 = df3.drop(columns=['Dia','Hora','estacao','HBV','Chuva','DirVento','VelVento','Temperatura','Pressao','Umidade'])
                    
                    train_df3 = train_df3.drop(columns=['Dia','Hora','estacao','HBV','Chuva','DirVento','VelVento','Temperatura','Pressao','Umidade'])
                    val_df3 = val_df3.drop(columns=['Dia','Hora','estacao','HBV','Chuva','DirVento','VelVento','Temperatura','Pressao','Umidade'])
                    test_df3 =  test_df3.drop(columns=['Dia','Hora','estacao','HBV','Chuva','DirVento','VelVento','Temperatura','Pressao','Umidade'])

                else:
                    df1 = df1.drop(columns=['DC_NOME','UF','DT_MEDICAO','CD_ESTACAO','VL_LATITUDE','VL_LONGITUDE','HR_MEDICAO','TEM_SEN','PRE_MAX','RAD_GLO','PTO_INS','TEM_MIN','UMD_MIN','PTO_MAX','PRE_MIN','UMD_MAX','PTO_MIN','TEM_MAX','TEN_BAT','VEN_RAJ','TEM_CPU'])
            
                    train_df3 = train_df3.drop(columns=['DC_NOME','UF','DT_MEDICAO','CD_ESTACAO','VL_LATITUDE','VL_LONGITUDE','HR_MEDICAO','TEM_SEN','PRE_MAX','RAD_GLO','PTO_INS','TEM_MIN','UMD_MIN','PTO_MAX','PRE_MIN','UMD_MAX','PTO_MIN','TEM_MAX','TEN_BAT','VEN_RAJ','TEM_CPU'])
                    val_df3 = val_df3.drop(columns=['DC_NOME','UF','DT_MEDICAO','CD_ESTACAO','VL_LATITUDE','VL_LONGITUDE','HR_MEDICAO','TEM_SEN','PRE_MAX','RAD_GLO','PTO_INS','TEM_MIN','UMD_MIN','PTO_MAX','PRE_MIN','UMD_MAX','PTO_MIN','TEM_MAX','TEN_BAT','VEN_RAJ','TEM_CPU'])
                    test_df3 =  test_df3.drop(columns=['DC_NOME','UF','DT_MEDICAO','CD_ESTACAO','VL_LATITUDE','VL_LONGITUDE','HR_MEDICAO','TEM_SEN','PRE_MAX','RAD_GLO','PTO_INS','TEM_MIN','UMD_MIN','PTO_MAX','PRE_MIN','UMD_MAX','PTO_MIN','TEM_MAX','TEN_BAT','VEN_RAJ','TEM_CPU'])
                
                train_df = pd.concat([train_df,train_df3], ignore_index=True)
                train_df.sort_values(by='data', inplace = True)
                train_df = train_df.fillna(0)
                val_df = pd.concat([val_df,val_df3], ignore_index=True)
                val_df.sort_values(by='data', inplace = True)
                val_df = val_df.fillna(0)
                
        
        if num_neighbors != 0:
            train_df = train_df.drop(columns=['data'])
            val_df = val_df.drop(columns=['data'])
            test_df = test_df.drop(columns=['data'])

        #Exportação
        train_df.to_csv('../data/'+ arq_pre_proc + '_train.csv')  
        val_df.to_csv('../data/'+ arq_pre_proc + '_val.csv')  
        test_df.to_csv('../data/'+ arq_pre_proc + '_test.csv')    
        print('Para o script de treinamento do modelo utilize arquivo : ' + arq_pre_proc)

def main(argv):
    arg_file = ""
    use_sounding_as_data_source = 0
    use_numerical_model_as_data_source = 0
    num_neighbors = 0
    help_message = "Usage: {0} -f <file> -d <data_source_spec> -n <num_neighbors>".format(argv[0])
    
    try:
        opts, args = getopt.getopt(argv[1:], "hf:d:n:", ["help", "file=", "datasources=", "neighbors="])
    except:
        print(help_message)
        sys.exit(2)
    
    for opt, arg in opts:
        if opt in ("-h", "--help"):
            print(help_message)  # print the help message
            sys.exit(2)
        elif opt in ("-f", "--file"):
            arg_file = arg
        elif opt in ("-d", "--datasources"):
            if arg.find('R') != -1:
                use_sounding_as_data_source = 1
            if arg.find('N') != -1:
                use_numerical_model_as_data_source = 1
        elif opt in ("-n", "--neighbors"):
            num_neighbors = arg

    pre_proc(arg_file, use_sounding_as_data_source, use_numerical_model_as_data_source, num_neighbors = num_neighbors)


if __name__ == "__main__":
    main(sys.argv)