import pandas as pd
import numpy
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

def decision_tree_method(X_train, y_train, X_test, y_test):
    # Inicializar o classificador da árvore de decisão
    clf = DecisionTreeClassifier()
    
    # Treinar o classificador com os dados de treinamento
    clf.fit(X_train, y_train)
    
    # fazer previsões nos dados de teste
    y_pred = clf.predict(X_test)
    
    # Caso queira visualizar a predição dos rótulos nos dados de teste remova o comentario no print
    #print(y_pred) 
    
    # Calcular a precisão do modelo
    accuracy = accuracy_score(y_test, y_pred)
    
    return accuracy

def naive_bayes_method(X_train, y_train, X_test, y_test):
    # Inicializar o classificador Naive Bayes
    clf = GaussianNB()

    # Treinar o classificador com os dados de treinamento
    clf.fit(X_train, y_train)

    # Fazer previsões nos dados de teste
    y_pred = clf.predict(X_test)
    
    # Caso queira visualizar a predição dos rótulos nos dados de teste remova o comentario no print
    #print(y_pred) 

    # Calcular a precisão do modelo
    accuracy = accuracy_score(y_test, y_pred)

    return accuracy

def SVM_vector_method(X_train, y_train, X_test, y_test):
    
     # Inicializar o classificador SVM
    clf = SVC(kernel='rbf', C=1.0, gamma='scale')

    # Treinar o classificador SVM
    clf.fit(X_train, y_train)

    # Prever os rótulos para os dados de teste
    y_pred = clf.predict(X_test)
    
    # Caso queira visualizar a predição dos rótulos nos dados de teste remova o comentario no print
    #print(y_pred)     

    # Calcular a precisão do modelo
    accuracy = accuracy_score(y_test, y_pred)
    
    return accuracy

def main():
    
    # Ler o arquivo .csv desejado e remover linhas nulas/vazias do dataset
    dados = pd.read_csv(fr"column_3C_weka.csv")
    dados = dados.dropna()

    # Separar os recursos (features) e os rótulos (labels)
    X = dados.drop('class', axis=1)
    y = dados['class']

    # Dividir os dados em conjuntos de treinamento e teste
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)  # 80% treinamento, 20% teste

    # Método da Árvore de Decisão
    decision_tree = decision_tree_method(X_train, y_train, X_test, y_test)
    print("\nPrecisão utilizando o método Árvore de Decisão:", decision_tree)
    
    # Método de Naive Bayes
    naive_bayes = naive_bayes_method(X_train, y_train, X_test, y_test)
    print("\nPrecisão utilizando o método de Naive Bayes:", naive_bayes)
    
    # Método de Vetores SVM
    SVM_vector = SVM_vector_method(X_train, y_train, X_test, y_test)
    print("\nPrecisão Utilizando o método de Vetores SVM:", SVM_vector)
    

if __name__ == "__main__":
    main()