import pandas as pd
import numpy as np
import sys, getopt, os, re
from datetime import datetime
import pickle


infile = open('../data/Janelamento_Train_RIO DE JANEIRO - JACAREPAGUA_1997_2022_X','rb')
new_dict_x = pickle.load(infile)

infile = open('../data/Janelamento_Train_RIO DE JANEIRO - VILA MILITAR_1997_2022_X','rb')
x_1 = pickle.load(infile)

new_dict_x = np.concatenate((new_dict_x,x_1), axis = 0)

infile = open('../data/Janelamento_Train_RIO DE JANEIRO-MARAMBAIA_1997_2022_X','rb')
x_2 = pickle.load(infile)

new_dict_x = np.concatenate((new_dict_x,x_2), axis = 0)

outfile = open('../data/Janelamento_Train_X','wb')
pickle.dump(new_dict_x,outfile)


infile = open('../data/Janelamento_Train_RIO DE JANEIRO - JACAREPAGUA_1997_2022_y','rb')
new_dict_y = pickle.load(infile)

infile = open('../data/Janelamento_Train_RIO DE JANEIRO - VILA MILITAR_1997_2022_y','rb')
y_1 = pickle.load(infile)
  
new_dict_y = np.concatenate((new_dict_y,y_1), axis = 0)

infile = open('../data/Janelamento_Train_RIO DE JANEIRO-MARAMBAIA_1997_2022_y','rb')
y_2 = pickle.load(infile)

new_dict_y = np.concatenate((new_dict_y,y_2), axis = 0)

outfile = open('../data/Janelamento_Train_y','wb')
pickle.dump(new_dict_y,outfile)

#------------------- Validação --------------------------


infile = open('../data/Janelamento_Val_RIO DE JANEIRO - JACAREPAGUA_1997_2022_X','rb')
new_dict_x = pickle.load(infile)

infile = open('../data/Janelamento_Val_RIO DE JANEIRO - VILA MILITAR_1997_2022_X','rb')
x_1 = pickle.load(infile)

new_dict_x = np.concatenate((new_dict_x,x_1), axis = 0)

infile = open('../data/Janelamento_Val_RIO DE JANEIRO-MARAMBAIA_1997_2022_X','rb')
x_2 = pickle.load(infile)

new_dict_x = np.concatenate((new_dict_x,x_2), axis = 0)

outfile = open('../data/Janelamento_Val_X','wb')
pickle.dump(new_dict_x,outfile)


infile = open('../data/Janelamento_Val_RIO DE JANEIRO - JACAREPAGUA_1997_2022_y','rb')
new_dict_y = pickle.load(infile)

infile = open('../data/Janelamento_Val_RIO DE JANEIRO - VILA MILITAR_1997_2022_y','rb')
y_1 = pickle.load(infile)
  
new_dict_y = np.concatenate((new_dict_y,y_1), axis = 0)

infile = open('../data/Janelamento_Val_RIO DE JANEIRO-MARAMBAIA_1997_2022_y','rb')
y_2 = pickle.load(infile)

new_dict_y = np.concatenate((new_dict_y,y_2), axis = 0)

outfile = open('../data/Janelamento_Val_y','wb')
pickle.dump(new_dict_y,outfile)