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
#import skimage

# Importa as bibliotecas do MR-CNN
from Mask_RCNN_TF2.mrcnn.config import Config
import Mask_RCNN_TF2.mrcnn.utils as utils
from Mask_RCNN_TF2.mrcnn import visualize
import Mask_RCNN_TF2.mrcnn.model as modellib
from Mask_RCNN_TF2.mrcnn.model import log


def mostrar(img: str) -> None:

  """
  Apresenta uma imagem na célula de notebook com base em seu diretório.

  Args:
    img (str): Diretório da imagem.
  """

  fig = plt.gcf()
  fig.set_size_inches(16, 10)
  plt.axis("off")
  plt.imshow(img)
  plt.show()


def github_to_raw(link: str) -> str:

    """
    Converte um link normal do GitHub para o formato Raw Content, permitindo que o conteúdo seja possível de baixar.

    Args:
        link (str): Link normal do GitHub.

    Returns:
        str: Link no formato Raw Content.
    """

    if "github.com" not in link or "/blob/" not in link:
        raise ValueError("O link fornecido não é válido ou não contém '/blob/'.")

    # Substituir o domínio e remover 'blob'
    raw_link = link.replace("https://github.com", "https://raw.githubusercontent.com")
    raw_link = raw_link.replace("/blob/", "/")

    return raw_link


def combine_base64_and_restore(df: pd.DataFrame, output_directory: str) -> None:

    """
    Converte o dataframe que contém as strings em base64 para o arquivo original.

    Args:
        df (pd.DataDrame): Dataframe que contém as strings em base64;
        output_directory (str): Pasta que vai conter o arquivo recriado a partir do dataframe.
    """

    # Cria a pasta de saída caso não exista
    os.makedirs(output_directory, exist_ok = True)

    # Agrupa as linhas do dataframe pelo diretório e reconstrói os arquivos
    for dir_name, group in df.groupby('diretorio'):
        # Junta as partes da base64
        combined_base64 = ''.join(group['base64_string'])

        # Converte de base64 para binário
        file_data = base64.b64decode(combined_base64)

        # Determina o nome do arquivo a ser restaurado
        # Extrai o nome do arquivo a partir do diretório (considera o nome da pasta)
        base_name = os.path.basename(dir_name)

        # Salva o arquivo restaurado no diretório de saída
        with open(file = os.path.join(output_directory, base_name), mode = "wb") as file:
            file.write(file_data)


def read_and_concat_parquets(folder_path: str) -> pd.DataFrame:

    """
    Lê todos os arquivos .parquet de uma pasta e concatena em um único dataframe.

    Args:
        folder_path (str): Caminho para a pasta contendo os arquivos .parquet.

    Returns:
        pd.DataFrame: Dataframe concatenado com os dados de todos os arquivos .parquet.
    """

    # Lista todos os arquivos na pasta especificada
    all_files = os.listdir(folder_path)

    # Filtra os arquivos .parquet
    parquet_files = [f for f in all_files if f.endswith('.parquet')]

    # Ordena os arquivos para garantir a ordem
    parquet_files.sort()

    # Caminho completo para cada arquivo
    parquet_paths = [os.path.join(folder_path, f) for f in parquet_files]

    # Lê e concatena os dataframes
    dataframes = [pd.read_parquet(file) for file in parquet_paths]
    concatenated_df = pd.concat(dataframes, ignore_index=True)

    return concatenated_df


def merge_ordered_dataframes(input_folder: str, output_file: str = 'df_unificado.parquet') -> None:

    """
    Combina e organiza dataframes armazenados em arquivos .parquet em uma pasta,
    salvando o resultado em um único arquivo.

    Parâmetros:
    - input_folder (str): Diretório contendo os arquivos .parquet dos dataframes.
    - output_file (str): Caminho e nome do arquivo .parquet de saída.
    """

    # Lista todos os arquivos na pasta
    files = [f for f in os.listdir(input_folder) if f.endswith('.parquet')]

    if not files:
        raise FileNotFoundError(f"Nenhum arquivo .parquet encontrado na pasta {input_folder}.")

    # Ordena os arquivos com base no número extraído do nome
    files_sorted = sorted(files, key=lambda x: int(x.split('_')[-1].replace('.parquet', '')))

    # Cria uma lista para armazenar os dataframes
    dataframes = []

    # Carrega os dataframes na ordem correta
    for file in files_sorted:
        file_path = os.path.join(input_folder, file)
        df = pd.read_parquet(file_path)
        dataframes.append(df)

    # Concatena todos os dataframes em um único
    combined_df = pd.concat(dataframes, ignore_index=True)

    # Salva o dataframe único no arquivo de saída
    combined_df.to_parquet(output_file)


def corrigir_e_exibir(diretorio_imagem: str, exibir_imagem: bool = True) -> None:

    """
    Recebe o diretório de uma imagem, corrige a sua orientação, salva ela no lugar
    da original, e apresenta a imagem.
    """

    if os.path.exists(diretorio_imagem):  # Verifica se o arquivo existe
        # Corrigir a orientação com base nos metadados
        img = Image.open(diretorio_imagem)
        img_corrigida = ImageOps.exif_transpose(img)

        # Substituir o arquivo original pelo corrigido
        img_corrigida.save(diretorio_imagem)

        # Exibir a imagem caso o usuário tenha definido
        if exibir_imagem:
          plt.imshow(img_corrigida)
          plt.axis('off')
          plt.show()
    else:
        print(f"Arquivo não encontrado: {diretorio_imagem}")

#===========================================================================================================================================

# ESSA FUNÇÃO CONSIDERA CADA OBJETO DENTRO DE UMA IMAGEM
class DatasetPersonalizado(utils.Dataset):

    def load_object(self, dataset_dir, subset, nome_annotation="via_region_data.json"):

        """
        Carrega um subconjunto do dataset Balloon.
        dataset_dir: Diretorio raíz do dataset.
        subset: Subconjunto a ser carregado: train (treinamento) ou val (validação)

        EXEMPLO:

        self.add_class("objetos", 1, "Balloon")
        self.add_class("objetos", 2, "Mobile phone")
        self.add_class("objetos", 3, "Ruler")

        dicionario_classe_id = {"Balloon": 1, "Mobile phone": 2, "Ruler": 3}
        dicionario_id_classe = {valor: chave for chave, valor in dicionario_classe_id.items()}
        """

        #==================================================================================================================================================

        self.add_class("objetos", 1, "Mobile phone")

        dicionario_classe_id = {"Mobile phone": 1}
        dicionario_id_classe = {valor: chave for chave, valor in dicionario_classe_id.items()}

        #==================================================================================================================================================

        # Escolhe se é o dataset de Treinamento ou Validação
        assert subset in ["train", "val"]
        dataset_dir = os.path.join(dataset_dir, subset)

        annotations = json.load(open(os.path.join(dataset_dir, nome_annotation)))
        annotations = list(annotations.values())  # não precisa das dict keys

        # A ferramenta VIA salva as imagens em JSON mesmo que elas não contenham nenhuma anotação. Então, pulamos as imagens não anotadas.
        annotations = [a for a in annotations if a['regions']]

        contagem = 0
        # Adiciona as imagens
        for a in annotations:
            # Pega as coordenadas x e y dos pontos dos poligonos que formam o contorno de cada instância do objeto.
            # Eles são armazenadas em shape_attributes (para visualizar, abra o arquivo json)
            # A condição if é necessária para que o código suporte anotações geradas pelas versões 1.x e 2.x da VIA.
            if type(a['regions']) is dict:
                polygons = [r['shape_attributes'] for r in a['regions'].values()]

            # Caso seja uma lista
            else:
                polygons = [r['shape_attributes'] for r in a['regions']]
                classe = [dicionario_classe_id[r['region_attributes']['label']] for r in a['regions']]

            # A função load_mask() vai precisar do tamanho da imagem para que possa converter os polígonos em mascaras.
            # Infelizmente, o VIA não inclui isso no JSON, então devemos ler a imagem manualmente e gerar essas máscaras.
            image_path = os.path.join(dataset_dir, a['filename'])
            image = skimage.io.imread(image_path)
            height, width = image.shape[:2]
            contagem = contagem+1

            self.add_image("objetos",
                image_id=a['filename'],  # usa o nome do arquivo como id unico da imagem
                path=image_path,
                classe = classe,
                width=width, height=height,
                polygons=polygons)

        print("Imagens "+subset+": " + str(contagem))

    def load_mask(self, image_id):
        """Gera as mascaras das instâncias para a imagem.
       Returna:
        masks: Uma array booleana de formato/shape [height, width, instance count] com 1 mascara por instancia.
        class_ids: uma array de 1D contendo os IDs das mascaras das instancias.
        """
        # Se não for uma imagem de conjunto de dados do balão (balloon dataset), delegue à classe ascendente.
        image_info = self.image_info[image_id]
        #print(image_info)
        if image_info["source"] != "objetos":
            return super(self.__class__, self).load_mask(image_id)

        # Converte os poligonos em uma mascara bitmap com shape  [height, width, instance_count]
        info = self.image_info[image_id]
        mask = np.zeros([info["height"], info["width"], len(info["polygons"])],
                        dtype=np.uint8)

        # Agora será calculado a máscara da instância. para cada pixel da imagem, classificará como pertencente à classe ou não
        for i, p in enumerate(info["polygons"]):
            # Pega os indices dos pixels dentro dos poligonos e define eles como = 1 (cor branca), caso contrário continuará valor 0 (cor preta)
            rr, cc = skimage.draw.polygon(p['all_points_y'], p['all_points_x'], mask.shape) # passamos o .shape também como 3ª parâmetro para evitar possíveis erros

            mask[rr, cc, i] = 1

        # Retorna a mascara e a array dos IDs das classes de cada instancia.
        # Como nesse exemplo temos uma classe apenas, retornamos uma array composta de 1s
        # return mask.astype(bool), np.ones([mask.shape[-1]], dtype=np.int32)
        return mask.astype(bool), (np.array(info['classe'], dtype=np.int32)).ravel()

    def image_reference(self, image_id):
        """Retorna o caminho da imagem."""
        info = self.image_info[image_id]
        if info["source"] == "objetos":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)

class ConfigRede(Config):

  # balloon_phone_ruler
  NAME = 'phone'

  # 2
  IMAGES_PER_GPU = 2

  # 1 + (quantidade de classes), visto que o fundo da imagem também será considerado
  NUM_CLASSES = 1 + 1

  # Quantidade de imagens totais de treinamento
  # quantidade_imagens_train = len(lista_image_path_train)
  quantidade_imagens_train = 800
  STEPS_PER_EPOCH = (quantidade_imagens_train // IMAGES_PER_GPU) # (Quantidade de imagens) / IMAGES_PER_GPU = 5272 / 2 = 1318
  DETECTION_MIN_CONFIDENCE = 0.9
  USE_MINI_MASK = False

  # Recomendado deixar o tamanho 512 para execuções no Colab
  IMAGE_MIN_DIM = 512 # 512
  IMAGE_MAX_DIM = 512 # 512

  # Quantidade de imagens totais de validação
  # quantidade_imagens_val = len(lista_image_path_val)
  quantidade_imagens_val = 200
  VALIDATION_STEPS = (quantidade_imagens_val // IMAGES_PER_GPU) # 5 # (Quantidade de imagens) / IMAGES_PER_GPU = 1810 / 2 = 452

class InferenceConfig(ConfigRede):
  GPU_COUNT = 1
  IMAGES_PER_GPU = 1


def carrega_modelo_teste(model_path: str):

  """
  Retorna um modelo e suas configurações com base nos pesos referenciados.
  """

  inference_config = InferenceConfig()

  inference_config = InferenceConfig()
  model = modellib.MaskRCNN(mode = 'inference', config = inference_config, model_dir = model_path)
  model.load_weights(model_path, by_name = True)
  return model, inference_config


def segmentar_imagem(model, img, tipo_modelo):

  resultados = model.detect([img], verbose = 0)
  r = resultados[0]

  # Define as classes que serão segmentadas
  lista_classes = ['Background']
  lista_classes.extend(tipo_modelo)

  #===============================================================================================================================

  visualize.display_instances(img, r['rois'], r['masks'], r['class_ids'], lista_classes, r['scores'], figsize=(12,10))

  return r


def corrigir_e_exibir(diretorio_imagem: str, exibir_imagem: bool = True) -> None:

    """
    Recebe o diretório de uma imagem, corrige a sua orientação, salva ela no lugar
    da original, e apresenta a imagem.
    """

    # Verifica se o arquivo existe
    if os.path.exists(diretorio_imagem):  
        
        # Corrigir a orientação com base nos metadados
        img = Image.open(diretorio_imagem)
        img_corrigida = ImageOps.exif_transpose(img)

        # Substituir o arquivo original pelo corrigido
        img_corrigida.save(diretorio_imagem)

        # Exibir a imagem caso o usuário tenha definido
        if exibir_imagem:
          plt.imshow(img_corrigida)
          plt.axis('off')
          plt.show()
    else:
        print(f"Arquivo não encontrado: {diretorio_imagem}")



def apply_mask(image_path: str, mask: np.ndarray, output_path: str, alpha: float = 0.9,  intensity: float = 0.05):
    # Verificar se o output_path tem uma extensão válida
    valid_extensions = (".jpg", ".jpeg", ".png", ".bmp", ".tiff")
    if not output_path.lower().endswith(valid_extensions):
        raise ValueError("O caminho de saída deve ter uma extensão de imagem válida (.jpg, .png, .bmp, etc.)")
    
    # Carregar a imagem
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Erro ao carregar a imagem. Verifique o caminho.")
    
    # Garantir que a máscara tenha o mesmo tamanho da imagem
    if mask.shape[:2] != image.shape[:2] or mask.shape[2] != 1:
        raise ValueError("A máscara deve ter as mesmas dimensões espaciais da imagem e formato (x, y, 1).")
    
    # Converter a máscara para 2D
    mask = mask.squeeze(axis=-1)
    
    # Criar um overlay vermelho
    overlay = np.zeros_like(image, dtype=np.uint8)
    overlay[:, :] = (0, 0, 255)  # Vermelho puro em BGR
    
    # Converter a imagem para escala de cinza
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_image = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR)
    
    # Aplicar a intensidade de cores no fundo
    processed_image = cv2.addWeighted(gray_image, 1 - intensity, image, intensity, 0)
    
    # Aplicar a máscara
    masked_image = processed_image.copy()
    
    # Onde a máscara for True, misturamos a cor vermelha com a imagem original
    masked_image[mask] = cv2.addWeighted(image[mask], 1 - alpha, overlay[mask], alpha, 0)
    
    # Salvar a imagem com a máscara aplicada
    success = cv2.imwrite(output_path, masked_image)
    if not success:
        raise RuntimeError("Erro ao salvar a imagem. Verifique o caminho de saída e permissões.")
    
    print(f"Imagem salva em: {output_path}")





def highlight_objects(image_path: str, class1: dict, class2: dict, output_path: str, alpha: float = 1.0, intensity: float = 0.0):
    """
    Processa uma imagem para destacar objetos de duas classes em cores diferentes, desenha caixas delimitadoras
    e adiciona rótulos aleatórios. O restante da imagem é convertido para preto e branco.

    Args:
        image_path (str): Caminho da imagem original.
        class1 (dict): Dicionário contendo "rois" e "masks" para a primeira classe.
        class2 (dict): Dicionário contendo "rois" e "masks" para a segunda classe.
        output_path (str): Caminho onde a imagem processada será salva.
        alpha (float): Intensidade da cor do objeto em destaque (0 a 1).
        intensity (float): Intensidade das cores do fundo (0 preto e branco, 1 original).
    """
    # Cores para as classes (BGR)
    color1 = (0, 255, 0)  # Verde
    color2 = (255, 0, 0)  # Azul
    
    # Carregar imagem
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Erro ao carregar a imagem. Verifique o caminho.")
    
    height, width = image.shape[:2]
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_image = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR)
    processed_image = cv2.addWeighted(gray_image, 1 - intensity, image, intensity, 0)
    
    def apply_class_mask(image, rois, masks, color, label):
        overlay = np.zeros_like(image, dtype=np.uint8)
        overlay[:] = color

        if len(rois) > 0:
        
            for i in range(len(rois)):
                mask = masks[:, :, i]
                image[mask] = cv2.addWeighted(image[mask], 1 - alpha, overlay[mask], alpha, 0)
                
                y1, x1, y2, x2 = rois[i]
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 3)
                # label = "Classe " + str(random.randint(1, 2))
                tamanho_fonte = 0.5
                grossura_fonte = 2
                cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, tamanho_fonte, (0, 0, 255), grossura_fonte, cv2.LINE_AA)
        
    if len(class1["rois"]) > 0:
        apply_class_mask(processed_image, class1["rois"], class1["masks"], color1, "Celular")

    if len(class2["rois"]) > 0:
        apply_class_mask(processed_image, class2["rois"], class2["masks"], color2, "Bexiga")
    
    success = cv2.imwrite(output_path, processed_image)
    if not success:
        raise RuntimeError("Erro ao salvar a imagem. Verifique o caminho de saída e permissões.")
    


def highlight_objects(image_path: str, class1: dict, class2: dict, output_path: str, alpha: float = 1.0, intensity: float = 0.0):
    """
    Processa uma imagem para destacar objetos de duas classes em cores diferentes, desenha caixas delimitadoras
    e adiciona rótulos aleatórios. O restante da imagem é convertido para preto e branco.

    Args:
        image_path (str): Caminho da imagem original.
        class1 (dict): Dicionário contendo "rois" e "masks" para a primeira classe.
        class2 (dict): Dicionário contendo "rois" e "masks" para a segunda classe.
        output_path (str): Caminho onde a imagem processada será salva.
        alpha (float): Intensidade da cor do objeto em destaque (0 a 1).
        intensity (float): Intensidade das cores do fundo (0 preto e branco, 1 original).
    """
    # Cores para as classes (BGR)
    color1 = (0, 255, 0)  # Verde
    color2 = (255, 0, 0)  # Azul
    
    # Carregar imagem
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Erro ao carregar a imagem. Verifique o caminho.")
    
    height, width = image.shape[:2]
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_image = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR)
    processed_image = cv2.addWeighted(gray_image, 1 - intensity, image, intensity, 0)
    
    def apply_class_mask(image, rois, masks, color, label):
        overlay = np.zeros_like(image, dtype=np.uint8)
        overlay[:] = color

        if len(rois) > 0:
        
            for i in range(len(rois)):
                mask = masks[:, :, i]
                image[mask] = cv2.addWeighted(image[mask], 1 - alpha, overlay[mask], alpha, 0)
                
                y1, x1, y2, x2 = rois[i]
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 20)
                # label = "Classe " + str(random.randint(1, 2))
                cv2.putText(image, label, (x1, y1 - 50), cv2.FONT_HERSHEY_SIMPLEX, 10.0, (0, 0, 255), 20, cv2.LINE_AA)
        
    if len(class1["rois"]) > 0:
        apply_class_mask(processed_image, class1["rois"], class1["masks"], color1, "Celular")

    if len(class2["rois"]) > 0:
        apply_class_mask(processed_image, class2["rois"], class2["masks"], color2, "Bexiga")
    
    success = cv2.imwrite(output_path, processed_image)
    if not success:
        raise RuntimeError("Erro ao salvar a imagem. Verifique o caminho de saída e permissões.")
    

def highlight_objects_streamlit(image_path: str, class1: dict, class2: dict, output_path: str, alpha: float = 1.0, intensity: float = 0.0):
    """
    Processa uma imagem para destacar objetos de duas classes em cores diferentes, desenha caixas delimitadoras
    e adiciona rótulos aleatórios. O restante da imagem é convertido para preto e branco.

    Args:
        image_path (str): Caminho da imagem original.
        class1 (dict): Dicionário contendo "rois" e "masks" para a primeira classe.
        class2 (dict): Dicionário contendo "rois" e "masks" para a segunda classe.
        output_path (str): Caminho onde a imagem processada será salva.
        alpha (float): Intensidade da cor do objeto em destaque (0 a 1).
        intensity (float): Intensidade das cores do fundo (0 preto e branco, 1 original).
    """
    # Cores para as classes (BGR)
    color1 = (0, 255, 0)  # Verde
    color2 = (255, 0, 0)  # Azul
    
    # Carregar imagem
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Erro ao carregar a imagem. Verifique o caminho.")
    
    height, width = image.shape[:2]
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_image = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR)
    processed_image = cv2.addWeighted(gray_image, 1 - intensity, image, intensity, 0)
    
    def apply_class_mask(image, rois, masks, color, label):
        overlay = np.zeros_like(image, dtype=np.uint8)
        overlay[:] = color

        if len(rois) > 0:
        
            for i in range(len(rois)):
                mask = masks[:, :, i]
                image[mask] = cv2.addWeighted(image[mask], 1 - alpha, overlay[mask], alpha, 0)
                
                y1, x1, y2, x2 = rois[i]
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
                # label = "Classe " + str(random.randint(1, 2))
                # cv2.putText(image, label, (x1, y1 - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 5, cv2.LINE_AA)
        
    if len(class1["rois"]) > 0:
        apply_class_mask(processed_image, class1["rois"], class1["masks"], color1, "Celular")

    if len(class2["rois"]) > 0:
        apply_class_mask(processed_image, class2["rois"], class2["masks"], color2, "Bexiga")
    
    success = cv2.imwrite(output_path, processed_image)
    if not success:
        raise RuntimeError("Erro ao salvar a imagem. Verifique o caminho de saída e permissões.")