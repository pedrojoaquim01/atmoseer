import logging, argparse, time
import train_model as gm

logging.basicConfig(filename='../../log/log.txt',
                    filemode='a',
                    format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                    datefmt='%H:%M:%S',
                    level=logging.DEBUG)

var_vit = [0,1]

logging.info("Starting evaluation")
for i in var_vit:
    for j in var_vit:
        for k in var_vit:
            for l in var_vit:
                result = 0
                for m in [0,1,2,3,4,5,6,7,8,9]:
                    start = time.time()
                    if l > 0:
                        aux = gm.model('RIO DE JANEIRO - FORTE DE COPACABANA_1997_2022',i,j,k,10,5)
                    else:
                        aux = gm.model('RIO DE JANEIRO - FORTE DE COPACABANA_1997_2022',i,j,k)
                    result = result + aux
                result = result/10
                logging.info('Média modelo = ' + str(result))
                logging.info(f'Training took {time.time() - start} seconds')
logging.info("Finished evaluation")
    