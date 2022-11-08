import pandas as pd

estacoes_inmet = pd.read_json('https://apitempo.inmet.gov.br/estacoes/T')
estacoes_inmet = estacoes_inmet[estacoes_inmet['SG_ESTADO'] == 'RJ']
 
estacoes_inmet.head()
 
lista = estacoes_inmet
estacoes = estacoes_inmet[estacoes_inmet['CD_ESTACAO'] == 'A652']
anos = list(range(1998, 2022))
 
dfnew2 = pd.read_json('https://apitempo.inmet.gov.br/estacao/1997-01-01/1998-01-01/A652')
for i in anos:
    novo = pd.read_json('https://apitempo.inmet.gov.br/estacao/' + str(i) + '-01-01/' + str(i+1) + '-01-01/A652')
    uniao = [dfnew2, novo]
    dfnew2 = pd.concat(uniao)
dfnew2.to_csv('RIO DE JANEIRO - FORTE DE COPACABANA_1997_2022.csv')