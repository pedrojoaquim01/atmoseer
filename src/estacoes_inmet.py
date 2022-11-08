import pandas as pd
import xarray as xr
import numpy as np
import sys, getopt, os, re



def processamento(estacao):
    anos = list(range(1998, 2022))
    estacoes_inmet = pd.read_json('https://apitempo.inmet.gov.br/estacoes/T')
    estacoes_inmet = estacoes_inmet[estacoes_inmet['SG_ESTADO'] == 'RJ']
    estacoes = estacoes_inmet[estacoes_inmet['CD_ESTACAO'] == estacao]
    dfnew2 = pd.read_json('https://apitempo.inmet.gov.br/estacao/1997-01-01/1998-01-01/' + estacao)
    for i in anos:
        novo = pd.read_json('https://apitempo.inmet.gov.br/estacao/' + str(i) + '-01-01/' + str(i+1) + '-01-01/' + estacao)
        uniao = [dfnew2, novo]
        dfnew2 = pd.concat(uniao)
    dfnew2.to_csv(estacoes['DC_NOME'].iloc[0] + '_1997_2022.csv')


def myfunc(argv):
    arg_file = ""
    arg_help = "{0} -s <station> -a <all>".format(argv[0])
    
    try:
        opts, args = getopt.getopt(argv[1:], "hs:a:", ["help", "sta=", "all="])
    except:
        print(arg_help)
        sys.exit(2)
    
    for opt, arg in opts:
        if opt in ("-h", "--help"):
            print(arg_help)  # print the help message
            sys.exit(2)
        elif opt in ("-s", "--sta"):
            arg_file = arg
        elif opt in ("-a", "--all"):
            arg_file = arg

    if arg_file == '':
        print('Digite alguma das estações : alto_da_boa_vista, guaratiba, iraja, jardim_botanico, riocentro, santa_cruz, sao_cristovao, vidigal')
    else:
        processamento(arg_file)


if __name__ == "__main__":
    myfunc(sys.argv)


