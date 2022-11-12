import pandas as pd
import numpy as np
from pathlib import Path
from math import cos, asin, sqrt
import sys
import getopt


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

def pre_proc(arquivo,log_CAPE = 0,log_Vento = 0,log_Tempo = 0,mes_min = 0,mes_max = 0, sta = 0):
    sta = int(sta)
    arq_pre_proc = arquivo
    if(log_CAPE):
        arq_pre_proc = arq_pre_proc + '_CAPE'
    if(log_Vento):
        arq_pre_proc = arq_pre_proc + '_VENT'
    if(log_Tempo):
        arq_pre_proc = arq_pre_proc + '_TEMP'
    if  mes_min != 0 and mes_max != 0:
        arq_pre_proc = arq_pre_proc + '_' + str(mes_min) + '_' + str(mes_max)
    if  sta > 0:
        arq_pre_proc = arq_pre_proc + '_' + str(sta)

    dado_proc = Path('../data/'+arq_pre_proc+'.csv')

    if dado_proc.is_file():
        fonte = '../data/'+arq_pre_proc+'.csv'
        df = pd.read_csv(fonte)
        del df['Unnamed: 0']
        df = df.fillna(0)
        df = df.reindex(sorted(df.columns), axis=1)
        return df
    else:
        fonte = '../data/'+arquivo+'.csv'
        df = pd.read_csv(fonte)
        del df['Unnamed: 0']
        df = df.fillna(0)
        df = df.reindex(sorted(df.columns), axis=1)
        
        print('Log: Importou sem problema')

        cor_est = ['alto_da_boa_vista','guaratiba','iraja','jardim_botanico','riocentro','santa_cruz','sao_cristovao','vidigal']
        
        if(log_CAPE):
            df_rs = pd.read_csv('../data/sondas_CAPE_CIN.csv')
            df_rs['log_hr'] = ''
            df_rs['log_hr'] = df_rs['time'].map(lambda x: '0' if x[11:19] == '00:00:00' else '12')

            df['CAPE'] = np.nan
            df['CIN'] = np.nan

            if arquivo in cor_est:
                for i in df_rs.date.unique():
                    if df_rs[(df_rs['date'] == i) & (df_rs['log_hr'] == '0')]['CAPE'].unique().size == 0:
                        df.loc[(df['Dia'] == i) & (df['Hora'] == '00:00:00'),'CAPE'] = np.nan
                    else:
                        df.loc[(df['Dia'] == i) & (df['Hora'] == '00:00:00'),'CAPE'] = df_rs[(df_rs['date'] == i) & (df_rs['log_hr'] == '0')]['CAPE'].unique()[0]
                    
                    if df_rs[(df_rs['date'] == i) & (df_rs['log_hr'] == '0')]['CIN'].unique().size == 0:
                        df.loc[(df['Dia'] == i) & (df['Hora'] == '00:00:00'),'CIN'] = np.nan
                    else:
                        df.loc[(df['Dia'] == i) & (df['Hora'] == '00:00:00'),'CIN']  = df_rs[(df_rs['date'] == i) & (df_rs['log_hr'] == '0')]['CIN'].unique()[0]
                        

                    if df_rs[(df_rs['date'] == i) & (df_rs['log_hr'] == '12')]['CAPE'].unique().size == 0:    
                        df.loc[(df['Dia'] == i) & (df['Hora'] == '12:00:00'),'CAPE'] = np.nan
                    else:
                        df.loc[(df['Dia'] == i) & (df['Hora'] == '12:00:00'),'CAPE'] = df_rs[(df_rs['date'] == i) & (df_rs['log_hr'] == '12')]['CAPE'].unique()[0]

                    if df_rs[(df_rs['date'] == i) & (df_rs['log_hr'] == '12')]['CIN'].unique().size == 0:    
                        df.loc[(df['Dia'] == i) & (df['Hora'] == '12:00:00'),'CIN'] = np.nan
                    else:
                        df.loc[(df['Dia'] == i) & (df['Hora'] == '12:00:00'),'CIN']  = df_rs[(df_rs['date'] == i) & (df_rs['log_hr'] == '12')]['CIN'].unique()[0]

            else:    
                for i in df_rs.date.unique():
                    if df_rs[(df_rs['date'] == i) & (df_rs['log_hr'] == '0')]['CAPE'].unique().size == 0:
                        df.loc[(df['DT_MEDICAO'] == i) & (df['HR_MEDICAO'] == 0),'CAPE'] = np.nan
                    else:
                        df.loc[(df['DT_MEDICAO'] == i) & (df['HR_MEDICAO'] == 0),'CAPE'] = df_rs[(df_rs['date'] == i) & (df_rs['log_hr'] == '0')]['CAPE'].unique()[0]
                    
                    if df_rs[(df_rs['date'] == i) & (df_rs['log_hr'] == '0')]['CIN'].unique().size == 0:
                        df.loc[(df['DT_MEDICAO'] == i) & (df['HR_MEDICAO'] == 0),'CIN'] = np.nan
                    else:
                        df.loc[(df['DT_MEDICAO'] == i) & (df['HR_MEDICAO'] == 0),'CIN']  = df_rs[(df_rs['date'] == i) & (df_rs['log_hr'] == '0')]['CIN'].unique()[0]
                        

                    if df_rs[(df_rs['date'] == i) & (df_rs['log_hr'] == '12')]['CAPE'].unique().size == 0:    
                        df.loc[(df['DT_MEDICAO'] == i) & (df['HR_MEDICAO'] == 1200),'CAPE'] = np.nan
                    else:
                        df.loc[(df['DT_MEDICAO'] == i) & (df['HR_MEDICAO'] == 1200),'CAPE'] = df_rs[(df_rs['date'] == i) & (df_rs['log_hr'] == '12')]['CAPE'].unique()[0]

                    if df_rs[(df_rs['date'] == i) & (df_rs['log_hr'] == '12')]['CIN'].unique().size == 0:    
                        df.loc[(df['DT_MEDICAO'] == i) & (df['HR_MEDICAO'] == 1200),'CIN'] = np.nan
                    else:
                        df.loc[(df['DT_MEDICAO'] == i) & (df['HR_MEDICAO'] == 1200),'CIN']  = df_rs[(df_rs['date'] == i) & (df_rs['log_hr'] == '12')]['CIN'].unique()[0]
            df['CAPE'][0] = 0
            df['CIN'][0] = 0
            df = df.interpolate(method='linear')
            print('Log: Variavel CAPE sem problema')

        if(log_Vento):
            if arquivo in cor_est:
                wv = df['VelVento']
                wd_rad = df['DirVento']*np.pi / 180

                df['Wx'] = wv*np.cos(wd_rad)
                df['Wy'] = wv*np.sin(wd_rad)
            else:
                wv = df['VEN_VEL']
                wd_rad = df['VEN_DIR']*np.pi / 180

                df['Wx'] = wv*np.cos(wd_rad)
                df['Wy'] = wv*np.sin(wd_rad)
        
            print('Log: Variavel Vento sem problema')
        
        if(log_Tempo):
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
            
            print('Log: Variavel Tempo sem problema')

        if arquivo in cor_est:
            df['data'] = pd.to_datetime(df['Dia'] +' '+ df['Hora'], format='%Y-%m-%d%H:%M:%S', infer_datetime_format=True)
        else:
            df['data'] = pd.to_datetime(df['DT_MEDICAO'] + ' '+ df['HR_MEDICAO'].apply(lambda x: '{0:0>4}'.format(x)).str.slice(0, 2) + ':' + df['HR_MEDICAO'].apply(lambda x: '{0:0>4}'.format(x)).str.slice(2, 4) + ':00', format='%Y-%m-%d%H:%M:%S', infer_datetime_format=True)
            

        if  mes_min != 0 and mes_max != 0:
            if int(mes_min) > int(mes_max):
                print('Min > Max')
                df = df[~((df.data.dt.month > int(mes_max)) & (df.data.dt.month < int(mes_min)))]
            else:
                print('Min < Max')
                df = df[(df.data.dt.month > int(mes_min)) & (df.data.dt.month < int(mes_max))]
            
            del df['data']
        
        if(sta > 0):
            print('Entrou sta')
            result = prox(arquivo,sta)
            for s in result:
                fonte = '../data/'+s+'.csv'
                df1 = pd.read_csv(fonte)
                df1 = df1.fillna(0)
                del df1['Unnamed: 0']
                
                if s in cor_est:
                    df1['data'] = pd.to_datetime(df1['Dia'] +' '+ df1['Hora'], format='%Y-%m-%d%H:%M:%S', infer_datetime_format=True)
                else:
                    df1['data'] = pd.to_datetime(df1['DT_MEDICAO'] + ' '+ df1['HR_MEDICAO'].apply(lambda x: '{0:0>4}'.format(x)).str.slice(0, 2) + ':' + df1['HR_MEDICAO'].apply(lambda x: '{0:0>4}'.format(x)).str.slice(2, 4) + ':00', format='%Y-%m-%d%H:%M:%S', infer_datetime_format=True)
                
                if s in cor_est:
                    df1 = df1.drop(columns=['Dia','Hora','estacao','HBV'])
                else:
                    df1 = df1.drop(columns=['DC_NOME','UF','DT_MEDICAO','CD_ESTACAO','VL_LATITUDE','VL_LONGITUDE','HR_MEDICAO'])
                df1 = df1.add_suffix('_' + s)
                df1 = df1.rename(columns={"data_" + s : "data"})
                
                df = pd.merge(df,df1, how = 'outer')
                df = df.fillna(0)
                
                if arquivo in cor_est:
                  df = df.sort_values(by=['Dia','Hora'])
                else:
                  df = df.sort_values(by=['DT_MEDICAO','HR_MEDICAO'])
                  
        del df['data']
        df.to_csv('../data/'+arq_pre_proc + '.csv')    
        return df


def myfunc(argv):
    arg_file = ""
    arg_CAPE = 0
    arg_Tempo = 0
    arg_Vento = 0
    arg_min = 0
    arg_max = 0
    arg_sta = 0
    arg_help = "{0} -f <file> -c <log_CAPE> -t <log_Time> -w <log_Wind> -s <sta>".format(argv[0])
    
    try:
        opts, args = getopt.getopt(argv[1:], "hf:c:t:w:i:a:s:", ["help", "file=", 
        "cape=", "time=", "wind=", "min=", "max=", "sta="])
    except:
        print(arg_help)
        sys.exit(2)
    
    for opt, arg in opts:
        if opt in ("-h", "--help"):
            print(arg_help)  # print the help message
            sys.exit(2)
        elif opt in ("-f", "--file"):
            arg_file = arg
        elif opt in ("-c", "--cape"):
            arg_CAPE = arg
        elif opt in ("-t", "--time"):
            arg_Tempo = arg
        elif opt in ("-w", "--wind"):
            arg_Vento = arg
        elif opt in ("-i", "--min"):
            arg_min = arg
        elif opt in ("-a", "--max"):
            arg_max = arg
        elif opt in ("-s", "--sta"):
            arg_sta = arg

    pre_proc(arg_file,arg_CAPE,arg_Tempo,arg_Vento,arg_min,arg_max,arg_sta)


if __name__ == "__main__":
    myfunc(sys.argv)