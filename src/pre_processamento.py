import pandas as pd
import numpy as np
from pathlib import Path



def pre_proc(arquivo,log_CAPE = 0,log_Vento = 0,log_Tempo = 0,mes_min = 0,mes_max = 0):
    
    arq_pre_proc = arquivo
    if(log_CAPE):
        arq_pre_proc = arq_pre_proc + '_CAPE'
    if(log_Vento):
        arq_pre_proc = arq_pre_proc + '_VENT'
    if(log_Tempo):
        arq_pre_proc = arq_pre_proc + '_TEMP'
    if  mes_min != 0 and mes_max != 0:
        arq_pre_proc = arq_pre_proc + '_' + str(mes_min) + '_' + str(mes_max)

    dado_proc = Path('../Dados/'+arq_pre_proc+'.csv')

    if dado_proc.is_file():
        fonte = '../Dados/'+arq_pre_proc+'.csv'
        df = pd.read_csv(fonte)
        del df['Unnamed: 0']
        df = df.fillna(0)
        df = df.reindex(sorted(df.columns), axis=1)
        return df
    else:
        fonte = '../Dados/'+arquivo+'.csv'
        df = pd.read_csv(fonte)
        del df['Unnamed: 0']
        df = df.fillna(0)
        df = df.reindex(sorted(df.columns), axis=1)
        
        print('Log: Importou sem problema')

        cor_est = ['alto_da_boa_vista','guaratiba','iraja','jardim_botanico','riocentro','santa_cruz','sao_cristovao','vidigal']
        
        if(log_CAPE):
            df_rs = pd.read_csv('../Dados/sondas_CAPE_CIN.csv')
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

        

        if  mes_min != 0 and mes_max != 0:
            if arquivo in cor_est:
                df['data'] = pd.to_datetime(df['Dia'] +' '+ df['Hora'], format='%Y-%m-%d%H:%M:%S', infer_datetime_format=True)
            else:
                df['data'] = pd.to_datetime(df['DT_MEDICAO'] + ' '+ df['HR_MEDICAO'].apply(lambda x: '{0:0>4}'.format(x)).str.slice(0, 2) + ':' + df['HR_MEDICAO'].apply(lambda x: '{0:0>4}'.format(x)).str.slice(2, 4) + ':00', format='%Y-%m-%d%H:%M:%S', infer_datetime_format=True)
            

            if int(mes_min) > int(mes_max):
                print('Min > Max')
                df = df[~((df.data.dt.month > int(mes_max)) & (df.data.dt.month < int(mes_min)))]
            else:
                print('Min < Max')
                df = df[(df.data.dt.month > int(mes_min)) & (df.data.dt.month < int(mes_max))]
            
            del df['data']

        df.to_csv('../Dados/'+arq_pre_proc + '.csv')    
        return df