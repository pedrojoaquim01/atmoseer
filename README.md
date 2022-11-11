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

O script `Python estacoes_cor.py` possui um argumento `-s` ou `--sta` que define qual estação será selecionada. É preciso escolher entre as estações meteorológicas através do nome: alto_da_boa_vista, guaratiba, iraja, jardim_botanico, riocentro, santa_cruz, sao_cristovao, vidigal

O script `Python estacoes_inmet.py` possui um argumento `-s` ou `--sta`, que define qual estação será selecionada, e o argumento `-a` ou `--all`, que caso selecionado traz todas as estações. É preciso escolher entre as estações meteorológicas através do seu código: A652 (Forte de Copacabana), A636 (Jacarepagua), A621 (Vila Militar), A602 (Marambaia)

O script `Python estacoes_rad.py` não possui argumentos ao rodar será gerado o conjunto de dados da radiossonda do Aeroporto do Galeão.

Todos os conjuntos de dados gerados pelos códigos estarão alocados na pasta `./dados` do projeto 

#### Pré Processamento
O código de pré processamento é responsável por realizar diversas operações no conjunto de dados original, como a criação de variáveis ou agregação de dados, que podem ser interessantes para o treinamento do modelo e o seu resultado final. Para rodar o script de pré processamento é preciso executar o comando `Python pre_processing.py`. O código pre_processing possui 7 argumentos possíveis, com apénas o primeiro sendo obrigatório.

Os argumentos são:
 - `-f` ou `--file` Argumento obrigatório, representa o nome do arquivo de dados que será usado como base para o modelo. Deve ser igual ao nome de um dos arquivos presente na pasta de *Dados* do projeto.
 - `-c` ou `--cape` Log que determina se será incluso as variáveis CAPE e CIN, no conjunto de dados que será usado para o treinamento.
 - `-t` ou `--time` Log que determina se será incluso as variáveis que representam o tempo como variáveis cíclicas, no conjunto de dados que será usado para o treinamento.
 - `-w` ou `--wind` Log que determina se será incluso as variáveis compostas do vento, no conjunto de dados que será usado para o treinamento.
 - `-i` ou `--min` e `-a` ou `--max`  Delimita os meses que serão apresentados no conjunto de dados, escolhendo o mês de início no _min_ e o mês de fim no _max_. Vale ressaltar que a delimitação só funciona caso os dois períodos estejam preenchidos.
-  `-s` ou `--sta` Que define quantas estações próximas serão agregadas ao conjunto de dados
Exemplo de Execução:
  
  `Python pre_processing.py -f 'RIO DE JANEIRO - FORTE DE COPACABANA_1997_2022' -c 0 -t 1 -w 1 -i 10 -a 5 `

Será criado um conjunto de dados da estação do Forte de Copacabana, com dados apenas de Outubro até Março e serão inclusos as variáveis de Tempo e Vento.
 
#### Geração do modelo
O script de geração de modelo é responsável por realizar o treinamento e exportar os resultados obtidos pelo modelo após o teste. Ele pode ser executado através do comando  `Python cria_modelo.py`, que necessita de um argumento obrigatório  `-f` ou `--file`, o argumento recebe o nome de um dos conjuntos de dados disponíveis na pasta `./dados` do projeto. O
