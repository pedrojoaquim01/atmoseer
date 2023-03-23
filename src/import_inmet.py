import pandas as pd
import numpy as np
import sys, getopt, os, re



def processamento(estacao, all = 0,arg_begin = 1997,arg_end = 2022):
    anos = list(range(arg_begin, arg_end))
    estacoes_inmet = pd.read_json('https://apitempo.inmet.gov.br/estacoes/T')
    estacoes_inmet = estacoes_inmet[estacoes_inmet['SG_ESTADO'] == 'RJ']
    if all < 1:
        estacoes = estacoes_inmet[estacoes_inmet['CD_ESTACAO'] == estacao]
        dfnew2 = pd.read_json('https://apitempo.inmet.gov.br/estacao/'+ str(arg_begin) +'-01-01/'+ str(arg_begin + 1) +'01-01/' + estacao)
        for i in anos:
            novo = pd.read_json('https://apitempo.inmet.gov.br/estacao/' + str(i) + '-01-01/' + str(i+1) + '-01-01/' + estacao)
            uniao = [dfnew2, novo]
            dfnew2 = pd.concat(uniao)
        dfnew2.to_csv(estacoes['DC_NOME'].iloc[0] + '_'+ str(arg_begin) +'_'+ str(arg_end) +'.csv')
    else:
        estacoes = estacoes_inmet[estacoes_inmet['CD_ESTACAO'].isin(('A636', 'A621', 'A602', 'A652'))]
        for j in list(range(0, len(estacoes))):
            dfnew2 = pd.read_json(
                'https://apitempo.inmet.gov.br/estacao/'+ str(arg_begin) +'-01-01/'+ str(arg_begin + 1) +'-01-01/' + estacoes['CD_ESTACAO'].iloc[j])
            for i in anos:
                novo = pd.read_json('https://apitempo.inmet.gov.br/estacao/' + str(i) +
                                    '-01-01/' + str(i+1) + '-01-01/' + estacoes['CD_ESTACAO'].iloc[j])
                uniao = [dfnew2, novo]
                dfnew2 = pd.concat(uniao)
            dfnew2.to_csv(estacoes['DC_NOME'].iloc[j] + '.csv')


def myfunc(argv):
    arg_file = ""
    arg_all = 0
    arg_begin = 1997
    arg_end = 2022
    arg_help = "{0} -s <station> -a <all> -b <begin> -e <end>".format(argv[0])
    
    try:
        opts, args = getopt.getopt(argv[1:], "hs:a:b:e:", ["help", "sta=", "all=","begin=","end="])
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
            arg_all = 1
        elif opt in ("-b", "--begin"):
            arg_begin = arg
        elif opt in ("-e", "--end"):
            arg_end = arg

    if arg_file == '' and arg_all == 0:
        print('Digite alguma das estações : A652 (Forte de Copacabana), A636 (Jacarepagua), A621 (Vila Militar), A602 (Marambaia)')
    else:
        processamento(arg_file,arg_all,arg_begin,arg_end)


if __name__ == "__main__":
    myfunc(sys.argv)


