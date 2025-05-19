> Objetivo do projeto

Este projeto busca trazer todos os notebooks e arquivos relavantes utilizados no Google Colab para ambiente local, e com isso realizar a integração dos códigos de segmentação e/ou detecção com o ``Streamlit`` ou ``Kivy``

> Registro dos principais acontecimentos

|Autor|Data|Descrição|Commit|
|-|-|-|-|
|Ariosvaldo Filho|16/03/2025|Realização do primeiro commit, onde o arquivo `9. Testes com os pesos de segmentação - Teste.ipynb` do Google Colab foi importado para que eu pudesse começar a testá-lo em ambiente local. As bibliotecas e suas versões foram o primeiro teste e que funcionaram. <br><br> As bibliotecas instaladas permitiram que todas as células de importação de bibliotecas do notebook `9 - Testes com os pesos de segmentação - Teste.ipynb` não apresentassem nenhum erro. <br><br>A pasta `Mask_RCNN_TF2` é original e ainda não passou pela célula que realiza a sua instalação.|a61eb87|
|Ariosvaldo Filho|19/05/2025|Organização dos arquivos, dos notebooks.|xxxxxx|

> Descrição dos arquivos e pastas

|Arquivo ou pasta|Descrição|
|-|-|
|.venv|É o ambiente virtual que contém as bibliotecas com as versões utilizadas para rodar todos os arquivos do projeto.|
|analise_dados|Pasta que contém notebooks e arquivos que serão utilizados para a análise do modelo de previsão do peso através da área em $cm^2$ da bexiga.|
|imagens|Pasta que contém uma série de pastas no com o seguinte os nomes no seguinte formato id_peso, cada pasta vai conter uma série de fotos das bexigas com os celulares.|
|Mask_RCNN_TF2|Arquitetura de rede neural que realizará a segmentação dos objetos que foram treinados.|
|pesos_treinados_local|Contém os modelos que foram treinados, o modelo que segmenta a bexiga e o celular|
|.gitignore|Um arquivo padrão em projetos que faz com que o Git ignore as pastas e arquivos que forem listados lá dentro. <br><br> As pastas que foram ignoradas ocupam muito espaço de armazenamento.|
|.python-version|Arquivo que determina qual a versão do Python será utilizada no projeto.|
|9 - Colab.ipynb|Notebook extraído diretamente do Google Colab, onde utilizei o mesmo para comparação e adaptação de um novo noteook local. <br><br> Esse notebook|
|requirements.txt|Um arquivo padrão em projetos que busca apresentar as bibliotecas e suas respectivas versões. É fundamental para evitar conflitos de versões, você pode usar o comando `pip install -r requirements.txt` para a instalação das bibliotecas. <br><br>É aconselhado realizar a instalação das bibliotecas de forma manual e na ordem abaixo, visto que é a forma garantida de que não haverá problemas, já que a instalação através do ``requirements.txt`` já apresentou problemas.|
|auxiliar.py|Conjunto de funções próprias para auxiliar na execução de algumas células|




<br><br><br><br>

---

### Baixe e instale o pyenv
[Como instalar o pyenv para Windows, Linux e Mac](https://www.youtube.com/watch?v=9LYqtLuD7z4)

### Baixe e instale a versão 3.10.11 do python
Com o terminal aberto, digite: `pyenv install 3.10.11`

### Crie o ambiente virtual:

`python -m venv .venv`


### Entre no ambiente virtual:

`.\.venv\Scripts\activate`


### Instale as seguintes bibliotecas manualmente:

* pip install streamlit==1.11.0

* pip uninstall altair pandas pillow numpy -y

* pip install numpy==1.23

* pip install pandas==1.4

* pip install Pillow==9.4.0

* pip install altair==4.1.0

* pip install tensorflow==2.9.2

* pip install tensorflow-gpu==2.9.2

* pip install matplotlib==3.6

* pip install tqdm==4.64.1

* pip install scipy==1.10.0

* pip install scikit-learn==1.1.0

* pip install scikit-image==0.20.0

* pip install opencv-python==4.7.0.72 `pip install opencv-python==4.7.0.68, funcionou mas apareceu um aviso para atualizar`

* pip install Cython==3.0.7

* pip install imgaug==0.4.0

* ~~pip install h5py==3.8.0~~ `Já existe, não precisa instalar`

* ~~pip install ipython==4.7.0.68~~ `Não instalar, porque depois ela impossibilita que qualquer célula de qualquer notebook rode`


### Insira o comando abaixo para rodar a aplicação streamlit: <br>
`streamlit run streamlit_segmentacao.py`