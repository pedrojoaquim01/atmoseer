# AtmoSeer

## Sobre a ferramenta
Projeto de TCC, que tem o objetivo de realizar previsões de chuva utilizando Redes Neurais Convolucionais, com diferentes fontes de dados.

## Aplicação

### Requerimentos
Para a execução do projeto é preciso que as bibliotecas disponíveis no documento requeriments.txt estejam baixadas.

### Execução
Para rodar o projeto é preciso executar o comando `Python cria_modelo.py`. O código cria_modelo possui 6 argumentos possíveis, com apénas o primeiro sendo obrigatório.

Os argumentos são:
 - `-f` ou `--file` Argumento obrigatório, representa o nome do arquivo de dados que será usado como base para o modelo. Deve ser igual ao nome de um dos arquivos presente na pasta de *Dados* do projeto.
 - `-c` ou `--cape` Log que determina se será incluso as variáveis CAPE e CIN, no conjunto de dados que será usado para o treinamento.
 - `-t` ou `--time` Log que determina se será incluso as variáveis que representam o tempo como variáveis cíclicas, no conjunto de dados que será usado para o treinamento.
 - `-w` ou `--wind` Log que determina se será incluso as variáveis compostas do vento, no conjunto de dados que será usado para o treinamento.
 - `-i` ou `--min` e `-a` ou `--max`  Delimita os meses que serão apresentados no conjunto de dados, escolhendo o mês de início no _min_ e o mês de fim no _max_. Vale ressaltar que a delimitação só funciona caso os dois períodos estejam preenchidos.

*Exemplo de Execução:*
  `Python cria_modelo.py -f 'RIO DE JANEIRO - FORTE DE COPACABANA_1997_2022' -c 0 -t 1 -w 1 -i 10 -a 5`

Será criado um modelo de predição de chuva baseado na estação do Forte de Copacabana, com dados apenas de Outubro até Março e serão inclusos as variáveis de Tempo e Vento.
 
 
