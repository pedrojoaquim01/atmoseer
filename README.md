# AtmoSeer

## Sobre a ferramenta
Projeto de TCC, que tem o objetivo de realizar previsões de chuva utilizando Redes Neurais Convolucionais, com diferentes fontes de dados.

## Aplicação

### Requerimentos
Para a execução do projeto é preciso que as bibliotecas disponíveis no documento requeriments.txt estejam baixadas.

### Execução
O projeto possue 3 tipos de código que podem ser executados, Importação de dados, Pré processamento e Geração do modelo. Para acessar os códigos é necessários estar na pasta `./src` .

#### Importação de Dados
No projeto existem 3 diferentes script de importação de dados, para estações do COR, estações do INMET e estações de Radiossonda. Eles são responsáveis por gerar os conjuntos de dados que serão utilizados para o treinamento do modelo.

O script **_estacoes_cor.py_** possui quatro argumentos `-s` ou `--sta` que define qual estação será selecionada, o argumento `-a` ou `--all` que caso seja preenchido com 1 indica que serão importados os dados de todas as estações, e por fim `-b` ou `--begin` e `-e` ou `--end` que pode ser preenchido com o intervalo de anos para a importação dos dados (O intervalo padrão de importação dos dados é de 1997 até 2022). É preciso escolher entre as estações meteorológicas através do nome: alto_da_boa_vista, guaratiba, iraja, jardim_botanico, riocentro, santa_cruz, sao_cristovao, vidigal.
Exemplo de Execução:

`Python estacoes_cor.py -s sao_cristovao`

Será importado o conjunto de dados da estação de São Cristóvão para a pasta de dados do projeto.

`Python estacoes_cor.py -a 1 -b 2000 -e 2015`

Será importado os conjuntos de dados de todas as estações no período de 2000 até 2015.


O script **_estacoes_inmet.py_** possui quatro argumentos `-s` ou `--sta`, que define qual estação será selecionada, o argumento `-a` ou `--all` que caso seja preenchido com 1 indica que serão importados os dados de todas as estações, e por fim `-b` ou `--begin` e `-e` ou `--end` que pode ser preenchido com o intervalo de anos para a importação dos dados (O intervalo padrão de importação dos dados é de 1997 até 2022). É preciso escolher entre as estações meteorológicas através do seu código: A652 (Forte de Copacabana), A636 (Jacarepagua), A621 (Vila Militar), A602 (Marambaia)
Exemplo de Execução:

`Python estacoes_inmet.py -s A652`

Será importado o conjunto de dados da estação do Forte de Copacabana para a pasta de dados do projeto.

`Python estacoes_inmet.py -a 1 -b 1999 -e 2017`

Será importado os conjuntos de dados de todas as estações no período de 1999 até 2017.


O script **_estacoes_rad.py_** possui apenas dois argumentos `-b` ou `--begin` e `-e` ou `--end` que pode ser preenchido com o intervalo de anos para a importação dos dados (O intervalo padrão de importação dos dados é de 1997 até 2022). Ao rodar será gerado o conjunto de dados da radiossonda do Aeroporto do Galeão.
Exemplo de Execução:

`Python estacoes_rad.py`

Será importado o conjunto de dados da radiossonda do Aeroporto do Galeão para a pasta de dados do projeto.

O script **_index_rad.py_** não possui argumentos, ao rodar será gerado os indices de instabilidade atmosférica para os dados importados no script **_estacoes_rad.py_**.
Exemplo de Execução:

`Python index_rad.py`

Os dados da radiossonda do Aeroporto do Galeão serão utilizados para calcular os indices de instabilidade atmosférica gerando um novo conjunto de dados na pasta de dados do projeto.

Todos os conjuntos de dados gerados pelos códigos estarão alocados na pasta `./dados` do projeto 

#### Pré Processamento
O código de pré processamento é responsável por realizar diversas operações no conjunto de dados original, como a criação de variáveis ou agregação de dados, que podem ser interessantes para o treinamento do modelo e o seu resultado final. Para rodar o script de pré processamento é preciso executar o comando `Python pre_processing.py`. O código pre_processing possui 3 argumentos possíveis, com apénas o primeiro sendo obrigatório.

Os argumentos são:
 - `-f` ou `--file` Argumento obrigatório, representa o nome do arquivo de dados que será usado como base para o modelo. Deve ser igual ao nome de um dos arquivos presente na pasta de *Dados* do projeto.
 - `-d` ou `--data` Define as fontes de dados que serão utilizadas para montar o conjunto de dados.
  Usa o formato de acrônimos definidos no texto
    - E : Apenas estação meteorológica
    - E-N : Estação meteorológica e modelo numérico
    - E-R : Estação meteorológica e radiossonda
    - E-N-R : Estação meteorológica, modelo númerico e radiossonda
-  `-s` ou `--sta` Que define quantas estações próximas serão agregadas ao conjunto de dados
Exemplo de Execução:
  
  `Python pre_processing.py -f 'RIO DE JANEIRO - FORTE DE COPACABANA_1997_2022' -d 'E-N-R' -s 5'`

Será criado um conjunto de dados da estação do Forte de Copacabana, com agregação dos dados das 5 estações meteorológicas mais proximas, utilizando as fontes de dados: modelo númerico e radiossonda
 
#### Geração do modelo
O script de geração de modelo é responsável por realizar o treinamento e exportar os resultados obtidos pelo modelo após o teste. Ele pode ser executado através do comando  `Python cria_modelo.py`, que necessita de dois argumentos `-f` ou `--file` que recebe o nome de um dos conjuntos de dados gerados do pré processamento e  `-r` ou `--reg` que define o arquitetura que será utilizada.  
Exemplo de Execução:

`Python cria_modelo.py -f 'RIO DE JANEIRO - FORTE DE COPACABANA_E-N-R_EI+5NN'`

Será criado um modelo de classificação ordinal baseado no conjunto de dados da estação do Forte de Copacabana já processado

`Python cria_modelo.py -f 'RIO DE JANEIRO - FORTE DE COPACABANA_E-N-R_EI+5NN' -r 1`

Será criado um modelo de regressão baseado no conjunto de dados da estação do Forte de Copacabana já processado

## Exemplo de teste do sistema

Importação : `Python estacoes_inmet.py -s A652`

Pré processamento : `Python pre_processing.py -f 'RIO DE JANEIRO - FORTE DE COPACABANA' -d 'E-N-R' -s 5 `

Geração do modelo : `Python cria_modelo.py -f 'RIO DE JANEIRO - FORTE DE COPACABANA_E-N-R_EI+5NN' -r 1`
