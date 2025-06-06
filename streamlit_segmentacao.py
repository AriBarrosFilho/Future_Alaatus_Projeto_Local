# streamlit run streamlit_segmentacao.py
import streamlit as st
import io

import joblib

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

import streamlit as st


NOME_IMAGEM = "imagens_geradas/imagem_streamlit.png"

img_file_buffer = st.camera_input("Tire uma foto de uma bexiga junto com um celular")
# img_file_buffer = st.camera_input("")

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

    # Obtém o momento inicial de quando o processamento ocorre
    tempo_inicial = time.time()


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

# id_classe = {'1': 'Bexiga', '2': 'Celular'}

if os.path.exists(path = NOME_IMAGEM):

    for tipo in ["phone", "balloon"]:

        # Define a imagem com base em seu caminho
        img = imread(NOME_IMAGEM)

        # Define o modelo
        model = modelo_classe[tipo][0]

        # Obtém os dados de segmentação da imagem
        if tipo == "phone": r1 = model.detect([img], verbose = 0)[0]
            
        # Obtém os dados de segmentação da imagem
        if tipo == "balloon": r2 = model.detect([img], verbose = 0)[0]

    # Caso nenhum objeto tenha sido detectado
    if (r1["rois"].size == 0) and (r2["rois"].size == 0):

        if tempo_inicial != None:
            tempo_total = time.time() - tempo_inicial
            st.text(f"Demorou {tempo_total:.2f} segundos")

        st.text("Não foi possível detectar nenhum objeto, tente novamente.")

    # Caso algum objeto tenha sido detectado
    else:

        # Salva uma imagem com os objetos destacados
        highlight_objects_streamlit(image_path = NOME_IMAGEM, class1 = r1, 
                                    class2 = r2, output_path = NOME_IMAGEM, 
                                    alpha = 1.0, intensity = 0.0)

        # Apresenta a imagem
        st.image(NOME_IMAGEM)

        if (r1['masks'].shape[2] != 0) and r2['masks'].shape[2] != 0:

            # Calcula a média de pixels do celular
            media_pixels_celular = r1['masks'].sum() / r1['masks'].shape[2]

            # Calcula a densidade de pixels
            densidade_pixels = media_pixels_celular / 102.72

            # Obtém a área do balão em cm2
            area_balao = r2['masks'].sum() / densidade_pixels

            # Define os coeficientes da equação polinomial
            pol = np.poly1d([-2.69384403e-05,  2.44677526e-02,  
                             2.08218398e+00,  2.53396383e+01])

            # Tira o formato de matriz
            X = area_balao.ravel()

            # Obtém o peso previsto
            peso = pol(X)

            # Formata o peso previsto
            peso_final = np.round(a = peso, decimals = 1)[0]

            # Apresenta o peso previsto da bexiga ao usuário
            st.text(f"O balão pesa {peso_final} gramas")

        if tempo_inicial != None:

            # Obtém o tempo total de execução do processamento da imagem
            tempo_total = time.time() - tempo_inicial

            # Apresenta o tempo total de execução do processamento da imagem ao usuário
            st.text(f"Demorou {tempo_total:.1f} segundos")




# # Título da aplicação
# st.title("Previsão do peso de bexigas por imagem")

# # Para dar espaçamento vertical
# st.markdown("""# """)
# st.markdown("""# """)

# # Subtítulo
# st.subheader("Tire uma foto de uma bexiga com um celular do lado.")

# st.markdown(f"## O balão pesa {peso_final} gramas")