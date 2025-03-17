import streamlit as st
from PIL import Image
import io

img_file_buffer = st.camera_input("Take a picture")

def rotacionar_e_salvar(dados_bytes, caminho_arquivo):
    imagem = Image.open(io.BytesIO(dados_bytes))  # Carrega a imagem a partir dos bytes
    imagem_rotacionada = imagem.rotate(-90, expand=True)  # Rotaciona 90 graus no sentido horário
    imagem_rotacionada.save(caminho_arquivo, format="PNG")  # Salva como PNG

if img_file_buffer is not None:
    # To read image file buffer as bytes:
    bytes_data = img_file_buffer.getvalue()
    # Check the type of bytes_data:
    # Should output: <class 'bytes'>
    rotacionar_e_salvar(bytes_data, "saida.png")

# Caminho da imagem
caminho_imagem = "saida.png"

# Exibir a imagem no Streamlit
# st.image(caminho_imagem): Carrega e exibe a imagem.
# caption="Minha Imagem": Adiciona uma legenda opcional.
# use_column_width=True: Ajusta a imagem à largura da página.
st.image(caminho_imagem)
# st.image(caminho_imagem, caption="Minha Imagem", use_container_width =True)