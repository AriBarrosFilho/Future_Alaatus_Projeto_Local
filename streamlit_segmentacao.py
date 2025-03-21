# streamlit run streamlit_segmentacao.py


import streamlit as st
import io


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

NOME_IMAGEM = "imagem_streamlit.png"

img_file_buffer = st.camera_input("Tire uma foto de uma bexiga junto com um celular")

tempo_inicial = None

# Caso o usuário tenha clicado no botão para tirar foto
if img_file_buffer is not None:

    # To read image file buffer as bytes:
    bytes_data = img_file_buffer.getvalue()

    # Carrega a imagem a partir dos bytes
    imagem = Image.open(io.BytesIO(bytes_data))  

    # Salva como PNG
    imagem.save(NOME_IMAGEM, format="PNG")

    # 
    st.text("Você acabou de tirar a foto, estamos processando...")
    tempo_inicial = time.time()

#==========================================================================================================================


# # Verifica se as bibliotecas já foram carregadas
# if "pesos_carregados" not in st.session_state:

#     # Caminho onde estão os pesos treinados
#     caminho_pesos_treinados = "pesos_treinados_local\mask_rcnn_mobilephone_0017.h5"

#     # Tenta carregar o modelo
#     try:

#         # Carrega o modelo com base nos pesos importados
#         model_phone, inference_config_phone = carrega_modelo_teste(model_path = caminho_pesos_treinados)

#     # Caso não seja possível carregar o modelo
#     except:

#         # Alerta ao usuário de que houve um problema
#         print("DEU ALGO ERRADO, REVEJA AS CONFIGURAÇÕES DA QUANTIDADE DE CLASSES NA SEÇÃO 3.2.")
#         model_phone, inference_config_phone = (0, 0)

#     #==========================================================================================================================

#     # Caminho onde estão os pesos treinados
#     caminho_pesos_treinados = "pesos_treinados_local\mask_rcnn_balloon_0017.h5"

#     # Tenta carregar o modelo
#     try:

#         # Carrega o modelo com base nos pesos importados
#         model_balloon, inference_config_balloon = carrega_modelo_teste(model_path = caminho_pesos_treinados)

#     # Caso não seja possível carregar o modelo
#     except:

#         # Alerta ao usuário de que houve um problema
#         print("DEU ALGO ERRADO, REVEJA AS CONFIGURAÇÕES DA QUANTIDADE DE CLASSES NA SEÇÃO 3.2.")
#         model_balloon, inference_config_balloon = (0, 0)

#     #==========================================================================================================================

#     # Caminho onde estão os pesos treinados
#     caminho_pesos_treinados = "NÃO VOU USAR"

#     # Tenta carregar o modelo
#     try:

#         # Carrega o modelo com base nos pesos importados
#         model_balloon_phone, inference_config_balloon_phone = carrega_modelo_teste(model_path = caminho_pesos_treinados)

#     # Caso não seja possível carregar o modelo
#     except:
    
#         # Alerta ao usuário de que houve um problema
#         print("DEU ALGO ERRADO PARA O MODELO DO BALÃO E DO CELULAR, REVEJA AS CONFIGURAÇÕES DA QUANTIDADE DE CLASSES NA SEÇÃO 3.2.")
#         model_balloon_phone, inference_config_balloon_phone = (0,0)

#     #==========================================================================================================================

#     # Dicionário que conterá o modelo e o nome das classes para a segmentação
#     modelo_classe = {'balloon': (model_balloon, ['Bexiga']), 
#                     'phone': (model_phone, ["Celular"]), 
#                     'balloon_phone': (model_balloon_phone, ['Bexiga', 'Celular'])}

#     id_classe = {'1': 'Bexiga', '2': 'Celular'}

#     # Marca que as bibliotecas já foram carregadas
#     st.session_state["pesos_carregados"] = True

#     if "modelo_classe" not in st.session_state:
#         st.session_state.modelo_classe = modelo_classe

#     if "id_classe" not in st.session_state:
#         st.session_state.id_classe = id_classe


# modelo_classe = st.session_state.modelo_classe
# id_classe = st.session_state.id_classe

#-----------------------------------------------------------------------------------


# Caminho onde estão os pesos treinados
caminho_pesos_treinados = "pesos_treinados_local\mask_rcnn_mobilephone_0017.h5"

# Carrega o modelo com base nos pesos importados
model_phone, inference_config_phone = carrega_modelo_teste(model_path = caminho_pesos_treinados)

# Caminho onde estão os pesos treinados
caminho_pesos_treinados = "pesos_treinados_local\mask_rcnn_balloon_0017.h5"

# Carrega o modelo com base nos pesos importados
model_balloon, inference_config_balloon = carrega_modelo_teste(model_path = caminho_pesos_treinados)

# Dicionário que conterá o modelo e o nome das classes para a segmentação
modelo_classe = {
                    'balloon': (model_balloon, ['Bexiga']), 
                    'phone': (model_phone, ["Celular"])
                }

id_classe = {'1': 'Bexiga', '2': 'Celular'}

#==========================================================================================================================

if os.path.exists(path = NOME_IMAGEM):

    # Define o objeto que será segmentado
    # tipo_modelo: 'balloon', 'phone' ou 'balloon_phone'
    tipo = "phone"

    # Define a imagem com base em seu caminho
    img = imread(NOME_IMAGEM)

    # Define o modelo
    model = modelo_classe[tipo][0]

    # Obtém os dados de segmentação da imagem
    r = model.detect([img], verbose = 0)[0]

    # Caso nenhum objeto tenha sido detectado
    if r["rois"].size == 0:

        if tempo_inicial != None:
            tempo_total = time.time() - tempo_inicial
            st.text(f"Demorou {tempo_total:.2f} segundos")

        st.text("Não foi possível detectar nenhum objeto, tente novamente.")

    # Caso algum objeto tenha sido detectado
    else:
    
        #------------------------------------------------------------------------------------

        # Máscara de segmentação
        mask = r["masks"]

        # Aplica a máscara de segmentação na imagem e salva a nova imagem
        apply_mask(NOME_IMAGEM, mask, NOME_IMAGEM, alpha = 1.0, intensity = 0.0)

        st.image(NOME_IMAGEM)

        if tempo_inicial != None:
            tempo_total = time.time() - tempo_inicial
            st.text(f"Demorou {tempo_total:.2f} segundos")
            
        st.text("Deu certo")