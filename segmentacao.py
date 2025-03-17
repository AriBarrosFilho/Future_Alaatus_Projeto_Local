# Interage com o sistema operacional, como manipular arquivos e diretórios.
import os
import sys

# Criação de strings em base64
import base64

# Geração de números aleatórios
import random

# Funções matemáticas
import math

# Extração de conteúdos em textos
import re

# Contagem de tempo
import time

# Manipulações e tratamentos de arquivos json
import json

# Manipulação de arquivos e pastas no sistema
import shutil

# Realiza computação numérica eficiente com arrays multidimensionais e funções matemáticas avançadas.
import numpy as np

# Tratamento e manipulações de dataframes
import pandas as pd

# Geração de gráficos
import matplotlib.pyplot as plt

# Tratamento de imagens
from PIL import Image, ImageOps

# Criação de redes neurais
import tensorflow

# Apresentação do progresso do código
from tqdm import tqdm

# Divisão entre dados de treino e de validação
from sklearn.model_selection import train_test_split

# Manipulações e tratamentos de imagens
import cv2

import skimage.draw
from skimage.io import imread

# Importa todas as funções auxiliares
from auxiliar import *

# Importa as bibliotecas do MR-CNN
from Mask_RCNN_TF2.mrcnn.config import Config
import Mask_RCNN_TF2.mrcnn.utils as utils
from Mask_RCNN_TF2.mrcnn import visualize
import Mask_RCNN_TF2.mrcnn.model as modellib
from Mask_RCNN_TF2.mrcnn.model import log

# Execute as 5 linhas abaixo para que não tenhamos problemas ao executar com as versões mais recentes do Tensorflow
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

#==========================================================================================================================

# Caminho onde estão os pesos treinados
caminho_pesos_treinados = "pesos_treinados_local\mask_rcnn_mobilephone_0017.h5"

# Tenta carregar o modelo
try:

  # Carrega o modelo com base nos pesos importados
  model_phone, inference_config_phone = carrega_modelo_teste(model_path = caminho_pesos_treinados)

# Caso não seja possível carregar o modelo
except:

  # Alerta ao usuário de que houve um problema
  print("DEU ALGO ERRADO, REVEJA AS CONFIGURAÇÕES DA QUANTIDADE DE CLASSES NA SEÇÃO 3.2.")
  model_phone, inference_config_phone = (0, 0)

#==========================================================================================================================

# Caminho onde estão os pesos treinados
caminho_pesos_treinados = "pesos_treinados_local\mask_rcnn_balloon_0017.h5"

# Tenta carregar o modelo
try:

  # Carrega o modelo com base nos pesos importados
  model_balloon, inference_config_balloon = carrega_modelo_teste(model_path = caminho_pesos_treinados)

# Caso não seja possível carregar o modelo
except:

  # Alerta ao usuário de que houve um problema
  print("DEU ALGO ERRADO, REVEJA AS CONFIGURAÇÕES DA QUANTIDADE DE CLASSES NA SEÇÃO 3.2.")
  model_balloon, inference_config_balloon = (0, 0)

#==========================================================================================================================

# Caminho onde estão os pesos treinados
caminho_pesos_treinados = "NÃO VOU USAR"

# Tenta carregar o modelo
try:

  # Carrega o modelo com base nos pesos importados
  model_balloon_phone, inference_config_balloon_phone = carrega_modelo_teste(model_path = caminho_pesos_treinados)

# Caso não seja possível carregar o modelo
except:
  
  # Alerta ao usuário de que houve um problema
  print("DEU ALGO ERRADO PARA O MODELO DO BALÃO E DO CELULAR, REVEJA AS CONFIGURAÇÕES DA QUANTIDADE DE CLASSES NA SEÇÃO 3.2.")
  model_balloon_phone, inference_config_balloon_phone = (0,0)

#==========================================================================================================================

# Dicionário que conterá o modelo e o nome das classes para a segmentação
modelo_classe = {'balloon': (model_balloon, ['Bexiga']), 
                 'phone': (model_phone, ["Celular"]), 
                 'balloon_phone': (model_balloon_phone, ['Bexiga', 'Celular'])}

id_classe = {'1': 'Bexiga', '2': 'Celular'}

#==========================================================================================================================

# Pasta do Google Colab que receberá o conteúdo do repositório (A pasta será criada caso não exista)
pasta_destino = 'imagens'

# pasta_destino = "imagens"

# "/" para Google Colab
# "\\" para local
divisor_caminho = "\\"

#-------------------------------------------------------------------------------------------------------------------

# Lista que armazenará o caminho completo das imagens válidas
lista_caminho_arquivos = []

# Percorre todas as pastas e subpastas
for caminho, _, arquivos in os.walk(pasta_destino):
  for arquivo in arquivos:

    # Acrescenta na lista o caminho completo de todos os arquivos
    lista_caminho_arquivos.append(os.path.abspath(os.path.join(caminho, arquivo)))

# Filtra os arquivos que possuem a extensão .jpg
lista_caminho_arquivos = list(filter(lambda x: x.endswith('.jpg') and ('balança' not in x), lista_caminho_arquivos))

#-------------------------------------------------------------------------------------------------------------------

# Ordena a lista de acordo com o ID da bexiga
lista_caminho_arquivos = sorted(lista_caminho_arquivos, key = lambda x: int(x.split(divisor_caminho)[-2].split('_')[0]))

# Lista que armazenará o nome da imagens
lista_imagens = list(map(lambda x: os.path.basename(x), lista_caminho_arquivos))

# Lista que armazenará o ID da bexiga nas imagens
lista_id = list(map(lambda x: int(x.split(divisor_caminho)[-2].split('_')[0]), lista_caminho_arquivos))

# Lista que armazenará o peso da bexiga nas imagens
lista_pesos = list(map(lambda x: int(x.split(divisor_caminho)[-2].split('_')[1]), lista_caminho_arquivos))

# Dicionário que conterá os dados das imagens
dicionario_dados = {'caminho': lista_caminho_arquivos, 'imagem': lista_imagens, 'id': lista_id, 'gramas': lista_pesos}

# Cria um dataframe com os dados das imagens
df_imagens = pd.DataFrame(dicionario_dados)

# Percorre todas as imagens
for imagem in lista_caminho_arquivos:

  # Corrige a orientação da imagem e salva ela no lugar da original
  corrigir_e_exibir(diretorio_imagem = imagem, exibir_imagem = False)

#==========================================================================================================================

# Índice da imagem
indice = 4

# Define o objeto que será segmentado
# tipo_modelo: 'balloon', 'phone' ou 'balloon_phone'
tipo = "balloon"

# Seleciona a linha do dataframe referente ao seu índice
df_imagem_especifica = df_imagens.iloc[indice]

# Define a imagem com base em seu caminho
img = imread(df_imagem_especifica['caminho'])

# Define o modelo
model = modelo_classe[tipo][0]

resultados = model.detect([img], verbose = 0)

r = resultados[0]

#==========================================================================================================================

# Caminho da imagem
image_path = df_imagens["caminho"][indice].split("future_alaatus\\")[1]

# Máscara de segmentação
mask = r["masks"]

print(mask)

# Imagem de saída
output_path = "teste2.png"

# Aplica a máscara de segmentação na imagem e salva a nova imagem
apply_mask(image_path, mask, output_path, alpha = 1.0, intensity = 0.0)

#==========================================================================================================================


#==========================================================================================================================


#==========================================================================================================================