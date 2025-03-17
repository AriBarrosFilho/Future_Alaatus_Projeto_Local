> Objetivo do projeto

Este projeto busca trazer todos os notebooks e arquivos relavantes utilizados no Google Colab para ambiente local, e com isso realizar a integração dos códigos de segmentação e/ou detecção com o ``Streamlit`` ou ``Kivy``

> Registro dos principais acontecimentos

|Autor|Data|Descrição|Commit|
|-|-|-|-|
|Ariosvaldo Filho|xx/xx/xxxx|Realização do primeiro commit, onde o arquivo `9. Testes com os pesos de segmentação - Teste.ipynb` do Google Colab foi importado para que eu pudesse começar a testá-lo em ambiente local. As bibliotecas e suas versões foram o primeiro teste e que funcionaram. <br><br> As bibliotecas instaladas permitiram que todas as células de importação de bibliotecas do notebook `9 - Testes com os pesos de segmentação - Teste.ipynb` não apresentassem nenhum erro. <br><br>A pasta `Mask_RCNN_TF2` é original e ainda não passou pela célula que realiza a sua instalação.|xxxx|

> Descrição dos arquivos e pastas

|Arquivo ou pasta|Descrição|
|-|-|
|requirements.txt|Um arquivo padrão em projetos que busca apresentar as bibliotecas e suas respectivas versões. É fundamental para evitar conflitos de versões, você pode usar o comando `pip install -r requirements.txt` para a instalação das bibliotecas. <br><br>É aconselhado realizar a instalação das bibliotecas de forma manual e na ordem abaixo, visto que é a forma garantida de que não haverá problemas, já que a instalação através do ``requirements.txt`` já apresentou problemas.|
|.gitignore|Um arquivo padrão em projetos que faz com que o Git ignore as pastas e arquivos que forem listados lá dentro. <br><br> As pastas que foram ignoradas ocupam muito espaço de memória.|
|utils.py|Conjunto de funções próprias para auxiliar na execução de algumas células|



<br><br><br><br>

---

Crie o ambiente virtual:

`python -m venv .venv`

Entre no ambiente virtual:

`.\.venv\Scripts\activate`

Instale as seguintes bibliotecas manualmente:

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