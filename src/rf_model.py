#%%
from pathlib import Path
import mlflow
import json
import requests
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc # metricas curva_roc e gini
import matplotlib.pyplot as plt # visualização gráfica
import numpy as np
import pandas as pd

current_dir = Path(__file__).resolve().parent

data_path = current_dir.parent / 'data' / 'processed' / 'df_final_dummies.csv'

#%%
df_final_dummies = pd.read_csv(data_path)

df_final_dummies.columns

#%%
## Transforma em array por otmizaçao
caracteristicas = df_final_dummies.drop(columns=['Unnamed: 0', 'km','municipio','data','acidentes','mes','uf'])

previsor = df_final_dummies.drop(columns=['Unnamed: 0', 'km', 'municipio', 'data', 'sentido_crescente', 'uf',
       'velocidade_Comercial', 'velocidade_Moto', 'velocidade_Passeio',
       'velocidade_Ônibus', 'volume_Comercial', 'volume_Moto',
       'volume_Passeio', 'volume_Ônibus', 'chuva',
       'mes', 'iluminacao', 'dia_da_semana_Final_de_Semana',
       'tipo_faixa_Terceira_faixa', 'tipo_perfil_de_terreno_Perfil_Ondulado',
       'tipo_perfil_de_terreno_Perfil_Plano'])


#%%
## Separa base teste e base treino
x_treino,x_teste,y_treino,y_teste = train_test_split(
    caracteristicas,
    previsor,
    test_size=0.3,
    random_state=10
)

#%%
## Construcao do modelo manual - testado max_depth [4,8,10,12,11]
algoritmo_rf1 = RandomForestClassifier(n_estimators=500, criterion='gini', random_state=10, max_depth=4)

algoritmo_rf1.fit(x_treino, y_treino)

#%%
## Construcaoo da curva ROC
fpr, tpr, thresholds =roc_curve(y_teste,
                                algoritmo_rf1.predict_proba(x_teste)[:,1])
roc_auc = auc(fpr, tpr)

## Calculo do coeficiente de GINI
gini = (roc_auc - 0.5)/(0.5)

print(round(roc_auc,4), round(gini,4))


#%%
#============ Inicializa Mlflow ============#

## Aponta a instancia do mlflow 
mlflow.set_tracking_uri("http://localhost:5000")

#%%
## Cria o eperimento
mlflow.set_experiment(experiment_name='Random Forest - acidentes rodoviários')


#%%
#============ Definição dos imputs ============#

## Modelo proposto: Modelo 1

## Coleta dos metadados do dataset para o padrao do mlflow
dataset = mlflow.data.from_pandas(df_final_dummies)

## Formula utilizada no modelo
#caracteristicas = caracteristicas
#previsor = previsor
sed = 10
criterio = 'gini'
arvores = 500
profundidade = 4


#%%
with mlflow.start_run(run_name='Modelo 1'):

    #============ Inputs ============#

    ## Coleta dos metadados do dataset. O contexto pode ser utilizado
    ## para adicionar mais uma informacao de contexto do dataset, por
    ## exemplo: se ele é um dataset de treino ou teste
    mlflow.log_input(dataset, context='training')

    ## Parametros de entrada
    mlflow.log_param('Caracteristicas', caracteristicas)
    mlflow.log_param('Previsor', previsor)
    mlflow.log_param('Seed', sed)
    mlflow.log_param('Criterio', criterio)
    mlflow.log_param('Arvores', arvores)
    mlflow.log_param('Profundidade', profundidade)
    
    #============ Modelagem ============#

    ## Treinamento do modelo
    modelo_1 = RandomForestClassifier(n_estimators=arvores, criterion=criterio, random_state=sed, max_depth=profundidade).fit(x_treino,y_treino)

    #============ Outputs ============#

    ## Registrando as mettricas do modelo
    fpr, tpr, thresholds =roc_curve(y_teste,
                                    modelo_1.predict_proba(x_teste)[:,1])
    
    roc_auc = auc(fpr, tpr)
    gini = (roc_auc - 0.5)/(0.5)

    mlflow.log_metric('Curva ROC', roc_auc)
    mlflow.log_metric('Gini', gini)

    ## Coletando artefatos do modelo

    ## Grafico que traz a curva ROC e GINI
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, marker='o', color='darkorchid', markersize=10, linewidth=3)
    ax.plot(fpr, fpr, color='gray', linestyle='dashed')
    ax.set_title('Área abaixo da curva: %g' % round(roc_auc, 4) +
            ' | Coeficiente de GINI: %g' % round(gini, 4))
    ax.set_xlabel('1 - Especificidade')
    ax.set_ylabel('Sensitividade')
    ax.set_xticks(np.arange(0, 1.1, 0.2))
    ax.set_yticks(np.arange(0, 1.1, 0.2))

    ## Registra o grafico como artefato do experimento
    mlflow.log_figure(
        fig, 'curva_roc.png'
    )

    ## Carregamento do modelo no experimento
    mlflow.sklearn.log_model(modelo_1, 'modelo-1')


#%%
#============ Definição dos imputs ============#

## Modelo proposto: Modelo 2

## Coleta dos metadados do dataset para o padrao do mlflow
dataset = mlflow.data.from_pandas(df_final_dummies)

## Formula utilizada no modelo
#caracteristicas = caracteristicas
#previsor = previsor
sed = 10
criterio = 'gini'
arvores = 500
profundidade = 6

#%%
with mlflow.start_run(run_name='Modelo 2'):

    #============ Inputs ============#

    ## Coleta dos metadados do dataset. O contexto pode ser utilizado
    ## para adicionar mais uma informacao de contexto do dataset, por
    ## exemplo: se ele é um dataset de treino ou teste
    mlflow.log_input(dataset, context='training')

    ## Parametros de entrada
    mlflow.log_param('Caracteristicas', caracteristicas)
    mlflow.log_param('Previsor', previsor)
    mlflow.log_param('Seed', sed)
    mlflow.log_param('Criterio', criterio)
    mlflow.log_param('Arvores', arvores)
    mlflow.log_param('Profundidade', profundidade)
    
    #============ Modelagem ============#

    ## Treinamento do modelo
    modelo_2 = RandomForestClassifier(n_estimators=arvores, criterion=criterio, random_state=sed, max_depth=profundidade).fit(x_treino,y_treino)

    #============ Outputs ============#

    ## Registrando as mettricas do modelo
    fpr, tpr, thresholds =roc_curve(y_teste,
                                    modelo_2.predict_proba(x_teste)[:,1])
    
    roc_auc = auc(fpr, tpr)
    gini = (roc_auc - 0.5)/(0.5)

    mlflow.log_metric('Curva ROC', roc_auc)
    mlflow.log_metric('Gini', gini)

    ## Coletando artefatos do modelo

    ## Grafico que traz a curva ROC e GINI
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, marker='o', color='darkorchid', markersize=10, linewidth=3)
    ax.plot(fpr, fpr, color='gray', linestyle='dashed')
    ax.set_title('Área abaixo da curva: %g' % round(roc_auc, 4) +
            ' | Coeficiente de GINI: %g' % round(gini, 4))
    ax.set_xlabel('1 - Especificidade')
    ax.set_ylabel('Sensitividade')
    ax.set_xticks(np.arange(0, 1.1, 0.2))
    ax.set_yticks(np.arange(0, 1.1, 0.2))

    ## Registra o grafico como artefato do experimento
    mlflow.log_figure(
        fig, 'curva_roc.png'
    )

    ## Carregamento do modelo no experimento
    mlflow.sklearn.log_model(modelo_2, 'modelo-2')


#%%
#============ Definição dos imputs ============#

## Modelo proposto: Modelo 3

## Coleta dos metadados do dataset para o padrao do mlflow
dataset = mlflow.data.from_pandas(df_final_dummies)

## Formula utilizada no modelo
#caracteristicas = caracteristicas
#previsor = previsor
sed = 10
criterio = 'gini'
arvores = 500
profundidade = 8

#%%
with mlflow.start_run(run_name='Modelo 3'):

    #============ Inputs ============#

    ## Coleta dos metadados do dataset. O contexto pode ser utilizado
    ## para adicionar mais uma informacao de contexto do dataset, por
    ## exemplo: se ele é um dataset de treino ou teste
    mlflow.log_input(dataset, context='training')

    ## Parametros de entrada
    mlflow.log_param('Caracteristicas', caracteristicas)
    mlflow.log_param('Previsor', previsor)
    mlflow.log_param('Seed', sed)
    mlflow.log_param('Criterio', criterio)
    mlflow.log_param('Arvores', arvores)
    mlflow.log_param('Profundidade', profundidade)
    
    #============ Modelagem ============#

    ## Treinamento do modelo
    modelo_3 = RandomForestClassifier(n_estimators=arvores, criterion=criterio, random_state=sed, max_depth=profundidade).fit(x_treino,y_treino)

    #============ Outputs ============#

    ## Registrando as mettricas do modelo
    fpr, tpr, thresholds =roc_curve(y_teste,
                                    modelo_3.predict_proba(x_teste)[:,1])
    
    roc_auc = auc(fpr, tpr)
    gini = (roc_auc - 0.5)/(0.5)

    mlflow.log_metric('Curva ROC', roc_auc)
    mlflow.log_metric('Gini', gini)

    ## Coletando artefatos do modelo

    ## Grafico que traz a curva ROC e GINI
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, marker='o', color='darkorchid', markersize=10, linewidth=3)
    ax.plot(fpr, fpr, color='gray', linestyle='dashed')
    ax.set_title('Área abaixo da curva: %g' % round(roc_auc, 4) +
            ' | Coeficiente de GINI: %g' % round(gini, 4))
    ax.set_xlabel('1 - Especificidade')
    ax.set_ylabel('Sensitividade')
    ax.set_xticks(np.arange(0, 1.1, 0.2))
    ax.set_yticks(np.arange(0, 1.1, 0.2))

    ## Registra o grafico como artefato do experimento
    mlflow.log_figure(
        fig, 'curva_roc.png'
    )

    ## Carregamento do modelo no experimento
    mlflow.sklearn.log_model(modelo_3, 'modelo-3')


#%%
#============ Definição dos imputs ============#

## Modelo proposto: Modelo 4

## Coleta dos metadados do dataset para o padrao do mlflow
dataset = mlflow.data.from_pandas(df_final_dummies)

## Formula utilizada no modelo
#caracteristicas = caracteristicas
#previsor = previsor
sed = 10
criterio = 'gini'
arvores = 500
profundidade = 10

#%%
with mlflow.start_run(run_name='Modelo 4'):

    #============ Inputs ============#

    ## Coleta dos metadados do dataset. O contexto pode ser utilizado
    ## para adicionar mais uma informacao de contexto do dataset, por
    ## exemplo: se ele é um dataset de treino ou teste
    mlflow.log_input(dataset, context='training')

    ## Parametros de entrada
    mlflow.log_param('Caracteristicas', caracteristicas)
    mlflow.log_param('Previsor', previsor)
    mlflow.log_param('Seed', sed)
    mlflow.log_param('Criterio', criterio)
    mlflow.log_param('Arvores', arvores)
    mlflow.log_param('Profundidade', profundidade)
    
    #============ Modelagem ============#

    ## Treinamento do modelo
    modelo_4 = RandomForestClassifier(n_estimators=arvores, criterion=criterio, random_state=sed, max_depth=profundidade).fit(x_treino,y_treino)

    #============ Outputs ============#

    ## Registrando as mettricas do modelo
    fpr, tpr, thresholds =roc_curve(y_teste,
                                    modelo_4.predict_proba(x_teste)[:,1])
    
    roc_auc = auc(fpr, tpr)
    gini = (roc_auc - 0.5)/(0.5)

    mlflow.log_metric('Curva ROC', roc_auc)
    mlflow.log_metric('Gini', gini)

    ## Coletando artefatos do modelo

    ## Grafico que traz a curva ROC e GINI
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, marker='o', color='darkorchid', markersize=10, linewidth=3)
    ax.plot(fpr, fpr, color='gray', linestyle='dashed')
    ax.set_title('Área abaixo da curva: %g' % round(roc_auc, 4) +
            ' | Coeficiente de GINI: %g' % round(gini, 4))
    ax.set_xlabel('1 - Especificidade')
    ax.set_ylabel('Sensitividade')
    ax.set_xticks(np.arange(0, 1.1, 0.2))
    ax.set_yticks(np.arange(0, 1.1, 0.2))

    ## Registra o grafico como artefato do experimento
    mlflow.log_figure(
        fig, 'curva_roc.png'
    )

    ## Carregamento do modelo no experimento
    mlflow.sklearn.log_model(modelo_4, 'modelo-4')


#%%
#============ Definição dos imputs ============#

## Modelo proposto: Modelo 5

## Coleta dos metadados do dataset para o padrao do mlflow
dataset = mlflow.data.from_pandas(df_final_dummies)

## Formula utilizada no modelo
#caracteristicas = caracteristicas
#previsor = previsor
sed = 10
criterio = 'gini'
arvores = 500
profundidade = 12

#%%
with mlflow.start_run(run_name='Modelo 5'):

    #============ Inputs ============#

    ## Coleta dos metadados do dataset. O contexto pode ser utilizado
    ## para adicionar mais uma informacao de contexto do dataset, por
    ## exemplo: se ele é um dataset de treino ou teste
    mlflow.log_input(dataset, context='training')

    ## Parametros de entrada
    mlflow.log_param('Caracteristicas', caracteristicas)
    mlflow.log_param('Previsor', previsor)
    mlflow.log_param('Seed', sed)
    mlflow.log_param('Criterio', criterio)
    mlflow.log_param('Arvores', arvores)
    mlflow.log_param('Profundidade', profundidade)
    
    #============ Modelagem ============#

    ## Treinamento do modelo
    modelo_5 = RandomForestClassifier(n_estimators=arvores, criterion=criterio, random_state=sed, max_depth=profundidade).fit(x_treino,y_treino)

    #============ Outputs ============#

    ## Registrando as mettricas do modelo
    fpr, tpr, thresholds =roc_curve(y_teste,
                                    modelo_5.predict_proba(x_teste)[:,1])
    
    roc_auc = auc(fpr, tpr)
    gini = (roc_auc - 0.5)/(0.5)

    mlflow.log_metric('Curva ROC', roc_auc)
    mlflow.log_metric('Gini', gini)

    ## Coletando artefatos do modelo

    ## Grafico que traz a curva ROC e GINI
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, marker='o', color='darkorchid', markersize=10, linewidth=3)
    ax.plot(fpr, fpr, color='gray', linestyle='dashed')
    ax.set_title('Área abaixo da curva: %g' % round(roc_auc, 4) +
            ' | Coeficiente de GINI: %g' % round(gini, 4))
    ax.set_xlabel('1 - Especificidade')
    ax.set_ylabel('Sensitividade')
    ax.set_xticks(np.arange(0, 1.1, 0.2))
    ax.set_yticks(np.arange(0, 1.1, 0.2))

    ## Registra o grafico como artefato do experimento
    mlflow.log_figure(
        fig, 'curva_roc.png'
    )

    ## Carregamento do modelo no experimento
    mlflow.sklearn.log_model(modelo_5, 'modelo-5')


#%%
#============ Definição dos imputs ============#

## Modelo proposto: Modelo 6

## Coleta dos metadados do dataset para o padrao do mlflow
dataset = mlflow.data.from_pandas(df_final_dummies)

## Formula utilizada no modelo
#caracteristicas = caracteristicas
#previsor = previsor
sed = 10
criterio = 'gini'
arvores = 500
profundidade = 11

#%%
with mlflow.start_run(run_name='Modelo 6'):

    #============ Inputs ============#

    ## Coleta dos metadados do dataset. O contexto pode ser utilizado
    ## para adicionar mais uma informacao de contexto do dataset, por
    ## exemplo: se ele é um dataset de treino ou teste
    mlflow.log_input(dataset, context='training')

    ## Parametros de entrada
    mlflow.log_param('Caracteristicas', caracteristicas)
    mlflow.log_param('Previsor', previsor)
    mlflow.log_param('Seed', sed)
    mlflow.log_param('Criterio', criterio)
    mlflow.log_param('Arvores', arvores)
    mlflow.log_param('Profundidade', profundidade)
    
    #============ Modelagem ============#

    ## Treinamento do modelo
    modelo_6 = RandomForestClassifier(n_estimators=arvores, criterion=criterio, random_state=sed, max_depth=profundidade).fit(x_treino,y_treino)

    #============ Outputs ============#

    ## Registrando as mettricas do modelo
    fpr, tpr, thresholds =roc_curve(y_teste,
                                    modelo_6.predict_proba(x_teste)[:,1])
    
    roc_auc = auc(fpr, tpr)
    gini = (roc_auc - 0.5)/(0.5)

    mlflow.log_metric('Curva ROC', roc_auc)
    mlflow.log_metric('Gini', gini)

    ## Coletando artefatos do modelo

    ## Grafico que traz a curva ROC e GINI
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, marker='o', color='darkorchid', markersize=10, linewidth=3)
    ax.plot(fpr, fpr, color='gray', linestyle='dashed')
    ax.set_title('Área abaixo da curva: %g' % round(roc_auc, 4) +
            ' | Coeficiente de GINI: %g' % round(gini, 4))
    ax.set_xlabel('1 - Especificidade')
    ax.set_ylabel('Sensitividade')
    ax.set_xticks(np.arange(0, 1.1, 0.2))
    ax.set_yticks(np.arange(0, 1.1, 0.2))

    ## Registra o grafico como artefato do experimento
    mlflow.log_figure(
        fig, 'curva_roc.png'
    )

    ## Carregamento do modelo no experimento
    mlflow.sklearn.log_model(modelo_6, 'modelo-6')


#%%
#============ Definição dos imputs ============#

## Modelo proposto: Modelo Grid Search

## Coleta dos metadados do dataset para o padrao do mlflow
dataset = mlflow.data.from_pandas(df_final_dummies)

## Formula utilizada no modelo
#caracteristicas = caracteristicas
#previsor = previsor
sed = 10
criterio = 'gini'
param_grid = {
    'n_estimators': [100, 200, 500, 1000],  # Número de árvores
    'max_depth': [4, 6, 8, 10, 11],  # Profundidade máxima
    'min_samples_split': [2, 5, 10, 15, None],  # Número mínimo de amostras para dividir um nó
    'min_samples_leaf': [1, 2, 4, 10, None],  # Número mínimo de amostras em uma folha
    'bootstrap': [True, False],  # Amostragem com ou sem reposição
    'max_features': ['auto', 'sqrt', 'log2']  # Número de variáveis a considerar para divisão
}
kfold = 3
scor = 'roc_auc'

#%%
with mlflow.start_run(run_name='Modelo Grid Search'):

    #============ Inputs ============#

    ## Coleta dos metadados do dataset. O contexto pode ser utilizado
    ## para adicionar mais uma informacao de contexto do dataset, por
    ## exemplo: se ele é um dataset de treino ou teste
    mlflow.log_input(dataset, context='training')

    ## Parametros de entrada
    mlflow.log_param('Caracteristicas', caracteristicas)
    mlflow.log_param('Previsor', previsor)
    mlflow.log_param('Seed', sed)
    mlflow.log_param('Criterio', criterio)
    mlflow.log_param('Grade de parametros', param_grid)
    mlflow.log_param('Kfold', kfold)
    mlflow.log_param('Scoring', scor)
    
    #============ Modelagem ============#

    ## Treinamento do modelo
    modelo_puro = RandomForestClassifier(random_state=sed)
    modelo_gs = GridSearchCV(estimator=modelo_puro, param_grid=param_grid, scoring=scor, cv=3, n_jobs=-1, verbose=2).fit(x_treino, y_treino)

    #============ Outputs ============#

    ## Registrando as mettricas do modelo
    fpr, tpr, thresholds =roc_curve(y_teste,
                                    modelo_gs.predict_proba(x_teste)[:,1])
    
    roc_auc = auc(fpr, tpr)
    gini = (roc_auc - 0.5)/(0.5)

    mlflow.log_metric('Curva ROC', roc_auc)
    mlflow.log_metric('Gini', gini)

    ## Coletando artefatos do modelo

    ## Grafico que traz a curva ROC e GINI
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, marker='o', color='darkorchid', markersize=10, linewidth=3)
    ax.plot(fpr, fpr, color='gray', linestyle='dashed')
    ax.set_title('Área abaixo da curva: %g' % round(roc_auc, 4) +
            ' | Coeficiente de GINI: %g' % round(gini, 4))
    ax.set_xlabel('1 - Especificidade')
    ax.set_ylabel('Sensitividade')
    ax.set_xticks(np.arange(0, 1.1, 0.2))
    ax.set_yticks(np.arange(0, 1.1, 0.2))

    ## Registra o grafico como artefato do experimento
    mlflow.log_figure(
        fig, 'curva_roc.png'
    )

    ## Carregamento do modelo no experimento
    mlflow.sklearn.log_model(modelo_gs, 'modelo-gs')


#%%
#============ Definição dos imputs ============#

## Modelo proposto: Modelo Random Search

## Coleta dos metadados do dataset para o padrao do mlflow
dataset = mlflow.data.from_pandas(df_final_dummies)

## Formula utilizada no modelo
#caracteristicas = caracteristicas
#previsor = previsor
sed = 10
criterio = 'gini'
param_grid = {
    'n_estimators': [100, 200, 500, 1000],  # Número de árvores
    'max_depth': [4, 6, 8, 10, 11],  # Profundidade máxima
    'min_samples_split': [2, 5, 10, 15, None],  # Número mínimo de amostras para dividir um nó
    'min_samples_leaf': [1, 2, 4, 10, None],  # Número mínimo de amostras em uma folha
    'bootstrap': [True, False],  # Amostragem com ou sem reposição
    'max_features': ['auto', 'sqrt', 'log2']  # Número de variáveis a considerar para divisão
}
kfold = 3
scor = 'roc_auc'

#%%
with mlflow.start_run(run_name='Modelo Random Search'):

    #============ Inputs ============#

    ## Coleta dos metadados do dataset. O contexto pode ser utilizado
    ## para adicionar mais uma informacao de contexto do dataset, por
    ## exemplo: se ele é um dataset de treino ou teste
    mlflow.log_input(dataset, context='training')

    ## Parametros de entrada
    mlflow.log_param('Caracteristicas', caracteristicas)
    mlflow.log_param('Previsor', previsor)
    mlflow.log_param('Seed', sed)
    mlflow.log_param('Criterio', criterio)
    mlflow.log_param('Grade de parametros', param_grid)
    mlflow.log_param('Kfold', kfold)
    mlflow.log_param('Scoring', scor)
    
    #============ Modelagem ============#

    ## Treinamento do modelo
    modelo_puro = RandomForestClassifier(random_state=sed)
    modelo_rs = RandomizedSearchCV(estimator=modelo_puro, param_distributions=param_grid, scoring=scor, n_iter=50, cv=kfold, verbose=2, random_state=sed, n_jobs=-1).fit(x_treino, y_treino)

    #============ Outputs ============#

    ## Registrando as mettricas do modelo
    fpr, tpr, thresholds =roc_curve(y_teste,
                                    modelo_rs.predict_proba(x_teste)[:,1])
    
    roc_auc = auc(fpr, tpr)
    gini = (roc_auc - 0.5)/(0.5)

    mlflow.log_metric('Curva ROC', roc_auc)
    mlflow.log_metric('Gini', gini)

    ## Coletando artefatos do modelo

    ## Grafico que traz a curva ROC e GINI
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, marker='o', color='darkorchid', markersize=10, linewidth=3)
    ax.plot(fpr, fpr, color='gray', linestyle='dashed')
    ax.set_title('Área abaixo da curva: %g' % round(roc_auc, 4) +
            ' | Coeficiente de GINI: %g' % round(gini, 4))
    ax.set_xlabel('1 - Especificidade')
    ax.set_ylabel('Sensitividade')
    ax.set_xticks(np.arange(0, 1.1, 0.2))
    ax.set_yticks(np.arange(0, 1.1, 0.2))

    ## Registra o grafico como artefato do experimento
    mlflow.log_figure(
        fig, 'curva_roc.png'
    )

    ## Carregamento do modelo no experimento
    mlflow.sklearn.log_model(modelo_rs, 'modelo-rs')

#%%

