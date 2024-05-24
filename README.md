# pre-processing-and-classification
Repositório referente a um modelo responsável por usar atributos biomecânicos para classificar os pacientes de acordo com os rótulos presentes em cada dataset.

### Contextualização dos dados:
Cada paciente é representado no conjunto de dados por seis atributos biomecânicos derivados da
forma e orientação da pelve e da coluna lombar (cada um é uma coluna):
- incidência pélvica
- inclinação pélvica
- ângulo de lordose lombar
- inclinação sacral
- rádio pélvico
- grau de espondilolistese

**Os dados foram organizados em dois arquivos de classificação diferentes, mas correlacionados:**
- column_2C_weka.csv (arquivo com dois rótulos de classe): as categorias Hérnia de
Disco e Espondilolistese foram fundidas em uma única categoria rotulada como
'anormal'. Assim, neste arquivo contém dados para classificar os pacientes como
pertencentes a uma de duas categorias: Normal (100 pacientes) ou Anormal (210
pacientes).
- column_3C_weka.csv (arquivo com três rótulos de classe): esse arquivo consiste em
classificar os pacientes como pertencentes a uma das três categorias: Normal (100
pacientes), Hérnia de Disco (60 pacientes) ou Espondilolistese (150 pacientes).

### Desenvolvimento:
Nesse código, foi aplicado o pré-processamento dos dados, verificando a existencia de valores faltantes, normalizando os dados e aplicando tecnicas de redução de dimensionalidade (como o PCA).
Após o pré-processamento do dataset separamos as instâncias aleatoriamente entre dados de treino e dados de teste.
Utilizando o conjunto de dados de treino, foi realizado a aprendizado para tês tipos de classificação:
- Método de Árvore de Decisão
- Método Bayesiano (Naive Bayes)
- Método de Vetores SVM

Após o treinamento dos dados utilizando cada um dos métodos foi realizada a classifcação pelo conjunto de testes e verificada a acurácia para cada um dos três modelos.

