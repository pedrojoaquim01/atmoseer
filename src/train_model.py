# -*- coding: utf-8 -*-
from numpy import array,mean
import torch
import gc
import torch.nn as nn
import pandas as pd
import numpy as np
import sys
import getopt
import  time
from typing import List
from math import cos, asin, sqrt
from Utils.windowing import generate_windowed_split
from Utils.model import NetOrdinalClassification, label2ordinalencoding, NetRegression
from Utils.training import fit, create_train_n_val_loaders, DeviceDataLoader, to_device, gen_learning_curve,seed_everything

cor_est = ['alto_da_boa_vista','guaratiba','iraja','jardim_botanico','riocentro','santa_cruz','sao_cristovao','vidigal']

"""# Main"""

def apply_subsampling(X, y, percentage = 0.1):
  print('*BEGIN*')
  print(X.shape)
  print(y.shape)
  y_eq_zero_idxs = np.where(y==0)[0]
  print('# original samples  eq zero:', y_eq_zero_idxs.shape)
  y_gt_zero_idxs = np.where(y>0)[0]
  print('# original samples gt zero:', y_gt_zero_idxs.shape)
  mask = np.random.choice([True, False], size=y.shape[0], p=[percentage, 1.0-percentage])
  y_train_subsample_idxs = np.where(mask==True)[0]
  print('# subsample:', y_train_subsample_idxs.shape)
  idxs = np.intersect1d(y_eq_zero_idxs, y_train_subsample_idxs)
  print('# subsample that are eq zero:', idxs.shape)
  idxs = np.union1d(idxs, y_gt_zero_idxs)
  print('# subsample final:', idxs.shape)
  X, y = X[idxs], y[idxs]
  print(X.shape)
  print(y.shape)
  print('*END*')
  return X, y

import time
import pandas as pd
import numpy as np
import torch

def train(X_train, y_train, X_val, y_val, ordinal_regression):
  N_EPOCHS = 5000
  PATIENCE = 500
  LEARNING_RATE = .3e-6
  NUM_FEATURES = X_train.shape[2]
  BATCH_SIZE = 512
  weight_decay = 1e-6

  if ordinal_regression:
    NUM_CLASSES = 5
    model = NetOrdinalClassification(in_channels = NUM_FEATURES,
                                    num_classes = NUM_CLASSES)
    y_train, y_val = label2ordinalencoding(y_train, y_val)
  else:
    global y_mean_value
    y_mean_value = np.mean(y_train)
    print(y_mean_value)
    model = NetRegression(in_channels = NUM_FEATURES, y_mean_value = y_mean_value)

  # model.apply(initialize_weights)

  criterion = nn.MSELoss()

  print(model)

  train_loader, val_loader = create_train_n_val_loaders(X_train, y_train, X_val, y_val, batch_size = BATCH_SIZE)

  optimizer = torch.optim.Adam(model.parameters(), 
                               lr=LEARNING_RATE, 
                               weight_decay=weight_decay)

  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  train_loader = DeviceDataLoader(train_loader, device)
  val_loader = DeviceDataLoader(val_loader, device)    
  to_device(model, device)
    
  train_loss, val_loss = fit(model, 
                             n_epochs = N_EPOCHS, 
                             optimizer = optimizer, 
                             train_loader = train_loader, 
                             val_loader = val_loader, 
                             patience = PATIENCE, 
                             criterion = criterion,
                             aux_nome = aux_nome,
                             num_sta = num_sta)

  gen_learning_curve(train_loss, val_loss,aux_nome,num_sta)

  return model

def main(ordinal_regression = True, file = ''):
  seed_everything()

  arquivo = '../data/' + file

  cor_est = ['alto_da_boa_vista','guaratiba','iraja','jardim_botanico','riocentro','santa_cruz','sao_cristovao','vidigal']
  
  for s in cor_est:
    if s in file:
      col_target = 'Chuva'
      break
    else:
      col_target = 'CHUVA'

  X_train, y_train, X_val, y_val, X_test, y_test = generate_windowed_split(arquivo, id_target = col_target, window_size = 6)

  print('***Before subsampling***')
  print('Max precipitation values (train/val/test): %d, %d, %d' % (np.max(y_train), np.max(y_val), np.max(y_test)))
  print('Mean precipitation values (train/val/test): %.4f, %.4f, %.4f' % (np.mean(y_train), np.mean(y_val), np.mean(y_test)))

  # ### Subsampling
  X_train, y_train = apply_subsampling(X_train, y_train)
  X_val, y_val = apply_subsampling(X_val, y_val)
  X_test, y_test = apply_subsampling(X_test, y_test)
  # ### Subsampling

  print('***After subsampling***')
  print('Max precipitation values (train/val/test): %d, %d, %d' % (np.max(y_train), np.max(y_val), np.max(y_test)))
  print('Mean precipitation values (train/val/test): %.4f, %.4f, %.4f' % (np.mean(y_train), np.mean(y_val), np.mean(y_test)))

  model = train(X_train, y_train, X_val, y_val, ordinal_regression)
  
  # load the best model
  model.load_state_dict(torch.load('../model/Modelo_'+ aux_nome +'.pt'))

  model.evaluate(X_test, y_test)


def myfunc(argv):
    arg_file = ""
    arg_model = True
    arg_help = "{0} -f <file> -s <sta>".format(argv[0])
    
    try:
        opts, args = getopt.getopt(argv[1:], "hf:r:", ["help", "file=","reg="])
    except:
        print(arg_help)
        sys.exit(2)
    
    for opt, arg in opts:
        if opt in ("-h", "--help"):
            print(arg_help)  # print the help message
            sys.exit(2)
        elif opt in ("-f", "--file"):
            arg_file = arg
        elif opt in ("-r", "--reg"):
            arg_model = False

        
    global aux_nome
    global num_sta
    aux_nome = ''
    num_sta = ''
    aux_nome = arg_file
    num_sta = str(arg_model)
    start_time = time.time()
    main(ordinal_regression = arg_model, file = arg_file)
    print("--- %s seconds ---" % (time.time() - start_time))



if __name__ == "__main__":
    myfunc(sys.argv)

