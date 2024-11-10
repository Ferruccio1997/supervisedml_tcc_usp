#%%
from pathlib import Path
import mlflow
import json
import mlflow.client
import requests
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from xgboost import XGBClassifier
from sklearn.metrics import roc_curve, auc # metricas curva_roc e gini
import matplotlib.pyplot as plt # visualização gráfica
import numpy as np
import pandas as pd
import utils.functions as functions

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
## Construcao do modelo manual - testado max_depth [4,6] e learning_rate [0.05, 0.01, 0.005]
algoritmo_xgb1 = XGBClassifier(max_depth=4, learning_rate=0.05, n_estimators=500, 
                             objective='binary:logistic', random_state=10)

algoritmo_xgb1.fit(x_treino, y_treino)

#%%
## Construcaoo da curva ROC
fpr, tpr, thresholds =roc_curve(y_teste,
                                algoritmo_xgb1.predict_proba(x_teste)[:,1])
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
mlflow.set_experiment(experiment_name='XGBoost - acidentes rodoviários')

#%%
#============ Definição dos imputs ============#

## Modelo proposto: Modelo 1

## Coleta dos metadados do dataset para o padrao do mlflow
dataset = mlflow.data.from_pandas(df_final_dummies)

## Formula utilizada no modelo
#caracteristicas = caracteristicas
#previsor = previsor
sed = 10
objetivo = 'binary:logistic'
arvores = 500
profundidade = 4
tx_apren = 0.05


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
    mlflow.log_param('Objetivo', objetivo)
    mlflow.log_param('Arvores', arvores)
    mlflow.log_param('Profundidade', profundidade)
    mlflow.log_param('Taxa de Aprendizado', tx_apren)
    
    #============ Modelagem ============#

    ## Treinamento do modelo
    modelo_1 = XGBClassifier(max_depth=profundidade, learning_rate=tx_apren, n_estimators=arvores, 
                             objective=objetivo, random_state=sed).fit(x_treino,y_treino)

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
    mlflow.xgboost.log_model(modelo_1, 'modelo-1')

#%%
#============ Definição dos imputs ============#

## Modelo proposto: Modelo 2

## Coleta dos metadados do dataset para o padrao do mlflow
dataset = mlflow.data.from_pandas(df_final_dummies)

## Formula utilizada no modelo
#caracteristicas = caracteristicas
#previsor = previsor
sed = 10
objetivo = 'binary:logistic'
arvores = 500
profundidade = 6
tx_apren = 0.05


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
    mlflow.log_param('Objetivo', objetivo)
    mlflow.log_param('Arvores', arvores)
    mlflow.log_param('Profundidade', profundidade)
    mlflow.log_param('Taxa de Aprendizado', tx_apren)
    
    #============ Modelagem ============#

    ## Treinamento do modelo
    modelo_2 = XGBClassifier(max_depth=profundidade, learning_rate=tx_apren, n_estimators=arvores, 
                             objective=objetivo, random_state=sed).fit(x_treino,y_treino)

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
    mlflow.xgboost.log_model(modelo_2, 'modelo-2')

#%%
#============ Definição dos imputs ============#

## Modelo proposto: Modelo 3

## Coleta dos metadados do dataset para o padrao do mlflow
dataset = mlflow.data.from_pandas(df_final_dummies)

## Formula utilizada no modelo
#caracteristicas = caracteristicas
#previsor = previsor
sed = 10
objetivo = 'binary:logistic'
arvores = 500
profundidade = 5
tx_apren = 0.05


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
    mlflow.log_param('Objetivo', objetivo)
    mlflow.log_param('Arvores', arvores)
    mlflow.log_param('Profundidade', profundidade)
    mlflow.log_param('Taxa de Aprendizado', tx_apren)
    
    #============ Modelagem ============#

    ## Treinamento do modelo
    modelo_3 = XGBClassifier(max_depth=profundidade, learning_rate=tx_apren, n_estimators=arvores, 
                             objective=objetivo, random_state=sed).fit(x_treino,y_treino)

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
    mlflow.xgboost.log_model(modelo_3, 'modelo-3')

#%%
#============ Definição dos imputs ============#

## Modelo proposto: Modelo 4

## Coleta dos metadados do dataset para o padrao do mlflow
dataset = mlflow.data.from_pandas(df_final_dummies)

## Formula utilizada no modelo
#caracteristicas = caracteristicas
#previsor = previsor
sed = 10
objetivo = 'binary:logistic'
arvores = 500
profundidade = 3
tx_apren = 0.05


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
    mlflow.log_param('Objetivo', objetivo)
    mlflow.log_param('Arvores', arvores)
    mlflow.log_param('Profundidade', profundidade)
    mlflow.log_param('Taxa de Aprendizado', tx_apren)
    
    #============ Modelagem ============#

    ## Treinamento do modelo
    modelo_4 = XGBClassifier(max_depth=profundidade, learning_rate=tx_apren, n_estimators=arvores, 
                             objective=objetivo, random_state=sed).fit(x_treino,y_treino)

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
    mlflow.xgboost.log_model(modelo_4, 'modelo-4')

#%%
#============ Definição dos imputs ============#

## Modelo proposto: Modelo 5

## Coleta dos metadados do dataset para o padrao do mlflow
dataset = mlflow.data.from_pandas(df_final_dummies)

## Formula utilizada no modelo
#caracteristicas = caracteristicas
#previsor = previsor
sed = 10
objetivo = 'binary:logistic'
arvores = 500
profundidade = 2
tx_apren = 0.05


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
    mlflow.log_param('Objetivo', objetivo)
    mlflow.log_param('Arvores', arvores)
    mlflow.log_param('Profundidade', profundidade)
    mlflow.log_param('Taxa de Aprendizado', tx_apren)
    
    #============ Modelagem ============#

    ## Treinamento do modelo
    modelo_5 = XGBClassifier(max_depth=profundidade, learning_rate=tx_apren, n_estimators=arvores, 
                             objective=objetivo, random_state=sed).fit(x_treino,y_treino)

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
    mlflow.xgboost.log_model(modelo_5, 'modelo-5')

#%%
#============ Definição dos imputs ============#

## Modelo proposto: Modelo 6

## Coleta dos metadados do dataset para o padrao do mlflow
dataset = mlflow.data.from_pandas(df_final_dummies)

## Formula utilizada no modelo
#caracteristicas = caracteristicas
#previsor = previsor
sed = 10
objetivo = 'binary:logistic'
arvores = 500
profundidade = 1
tx_apren = 0.05


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
    mlflow.log_param('Objetivo', objetivo)
    mlflow.log_param('Arvores', arvores)
    mlflow.log_param('Profundidade', profundidade)
    mlflow.log_param('Taxa de Aprendizado', tx_apren)
    
    #============ Modelagem ============#

    ## Treinamento do modelo
    modelo_6 = XGBClassifier(max_depth=profundidade, learning_rate=tx_apren, n_estimators=arvores, 
                             objective=objetivo, random_state=sed).fit(x_treino,y_treino)

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
    mlflow.xgboost.log_model(modelo_6, 'modelo-6')

#%%
#============ Definição dos imputs ============#

## Modelo proposto: Modelo 7

## Coleta dos metadados do dataset para o padrao do mlflow
dataset = mlflow.data.from_pandas(df_final_dummies)

## Formula utilizada no modelo
#caracteristicas = caracteristicas
#previsor = previsor
sed = 10
objetivo = 'binary:logistic'
arvores = 500
profundidade = 2
tx_apren = 0.01


#%%
with mlflow.start_run(run_name='Modelo 7'):

    #============ Inputs ============#

    ## Coleta dos metadados do dataset. O contexto pode ser utilizado
    ## para adicionar mais uma informacao de contexto do dataset, por
    ## exemplo: se ele é um dataset de treino ou teste
    mlflow.log_input(dataset, context='training')

    ## Parametros de entrada
    mlflow.log_param('Caracteristicas', caracteristicas)
    mlflow.log_param('Previsor', previsor)
    mlflow.log_param('Seed', sed)
    mlflow.log_param('Objetivo', objetivo)
    mlflow.log_param('Arvores', arvores)
    mlflow.log_param('Profundidade', profundidade)
    mlflow.log_param('Taxa de Aprendizado', tx_apren)
    
    #============ Modelagem ============#

    ## Treinamento do modelo
    modelo_7 = XGBClassifier(max_depth=profundidade, learning_rate=tx_apren, n_estimators=arvores, 
                             objective=objetivo, random_state=sed).fit(x_treino,y_treino)

    #============ Outputs ============#

    ## Registrando as mettricas do modelo
    fpr, tpr, thresholds =roc_curve(y_teste,
                                    modelo_7.predict_proba(x_teste)[:,1])
    
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
    mlflow.xgboost.log_model(modelo_7, 'modelo-7')

#%%
#============ Definição dos imputs ============#

## Modelo proposto: Modelo 8

## Coleta dos metadados do dataset para o padrao do mlflow
dataset = mlflow.data.from_pandas(df_final_dummies)

## Formula utilizada no modelo
#caracteristicas = caracteristicas
#previsor = previsor
sed = 10
objetivo = 'binary:logistic'
arvores = 500
profundidade = 3
tx_apren = 0.01


#%%
with mlflow.start_run(run_name='Modelo 8'):

    #============ Inputs ============#

    ## Coleta dos metadados do dataset. O contexto pode ser utilizado
    ## para adicionar mais uma informacao de contexto do dataset, por
    ## exemplo: se ele é um dataset de treino ou teste
    mlflow.log_input(dataset, context='training')

    ## Parametros de entrada
    mlflow.log_param('Caracteristicas', caracteristicas)
    mlflow.log_param('Previsor', previsor)
    mlflow.log_param('Seed', sed)
    mlflow.log_param('Objetivo', objetivo)
    mlflow.log_param('Arvores', arvores)
    mlflow.log_param('Profundidade', profundidade)
    mlflow.log_param('Taxa de Aprendizado', tx_apren)
    
    #============ Modelagem ============#

    ## Treinamento do modelo
    modelo_8 = XGBClassifier(max_depth=profundidade, learning_rate=tx_apren, n_estimators=arvores, 
                             objective=objetivo, random_state=sed).fit(x_treino,y_treino)

    #============ Outputs ============#

    ## Registrando as mettricas do modelo
    fpr, tpr, thresholds =roc_curve(y_teste,
                                    modelo_8.predict_proba(x_teste)[:,1])
    
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
    mlflow.xgboost.log_model(modelo_8, 'modelo-8')

#%%
#============ Definição dos imputs ============#

## Modelo proposto: Modelo 9

## Coleta dos metadados do dataset para o padrao do mlflow
dataset = mlflow.data.from_pandas(df_final_dummies)

## Formula utilizada no modelo
#caracteristicas = caracteristicas
#previsor = previsor
sed = 10
objetivo = 'binary:logistic'
arvores = 500
profundidade = 4
tx_apren = 0.01


#%%
with mlflow.start_run(run_name='Modelo 9'):

    #============ Inputs ============#

    ## Coleta dos metadados do dataset. O contexto pode ser utilizado
    ## para adicionar mais uma informacao de contexto do dataset, por
    ## exemplo: se ele é um dataset de treino ou teste
    mlflow.log_input(dataset, context='training')

    ## Parametros de entrada
    mlflow.log_param('Caracteristicas', caracteristicas)
    mlflow.log_param('Previsor', previsor)
    mlflow.log_param('Seed', sed)
    mlflow.log_param('Objetivo', objetivo)
    mlflow.log_param('Arvores', arvores)
    mlflow.log_param('Profundidade', profundidade)
    mlflow.log_param('Taxa de Aprendizado', tx_apren)
    
    #============ Modelagem ============#

    ## Treinamento do modelo
    modelo_9 = XGBClassifier(max_depth=profundidade, learning_rate=tx_apren, n_estimators=arvores, 
                             objective=objetivo, random_state=sed).fit(x_treino,y_treino)

    #============ Outputs ============#

    ## Registrando as mettricas do modelo
    fpr, tpr, thresholds =roc_curve(y_teste,
                                    modelo_9.predict_proba(x_teste)[:,1])
    
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
    mlflow.xgboost.log_model(modelo_9, 'modelo-9')

#%%
#============ Definição dos imputs ============#

## Modelo proposto: Modelo 10

## Coleta dos metadados do dataset para o padrao do mlflow
dataset = mlflow.data.from_pandas(df_final_dummies)

## Formula utilizada no modelo
#caracteristicas = caracteristicas
#previsor = previsor
sed = 10
objetivo = 'binary:logistic'
arvores = 500
profundidade = 5
tx_apren = 0.01


#%%
with mlflow.start_run(run_name='Modelo 10'):

    #============ Inputs ============#

    ## Coleta dos metadados do dataset. O contexto pode ser utilizado
    ## para adicionar mais uma informacao de contexto do dataset, por
    ## exemplo: se ele é um dataset de treino ou teste
    mlflow.log_input(dataset, context='training')

    ## Parametros de entrada
    mlflow.log_param('Caracteristicas', caracteristicas)
    mlflow.log_param('Previsor', previsor)
    mlflow.log_param('Seed', sed)
    mlflow.log_param('Objetivo', objetivo)
    mlflow.log_param('Arvores', arvores)
    mlflow.log_param('Profundidade', profundidade)
    mlflow.log_param('Taxa de Aprendizado', tx_apren)
    
    #============ Modelagem ============#

    ## Treinamento do modelo
    modelo_10 = XGBClassifier(max_depth=profundidade, learning_rate=tx_apren, n_estimators=arvores, 
                             objective=objetivo, random_state=sed).fit(x_treino,y_treino)

    #============ Outputs ============#

    ## Registrando as mettricas do modelo
    fpr, tpr, thresholds =roc_curve(y_teste,
                                    modelo_10.predict_proba(x_teste)[:,1])
    
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
    mlflow.xgboost.log_model(modelo_10, 'modelo-10')

#%%
#============ Definição dos imputs ============#

## Modelo proposto: Modelo 11

## Coleta dos metadados do dataset para o padrao do mlflow
dataset = mlflow.data.from_pandas(df_final_dummies)

## Formula utilizada no modelo
#caracteristicas = caracteristicas
#previsor = previsor
sed = 10
objetivo = 'binary:logistic'
arvores = 500
profundidade = 5
tx_apren = 0.005


#%%
with mlflow.start_run(run_name='Modelo 11'):

    #============ Inputs ============#

    ## Coleta dos metadados do dataset. O contexto pode ser utilizado
    ## para adicionar mais uma informacao de contexto do dataset, por
    ## exemplo: se ele é um dataset de treino ou teste
    mlflow.log_input(dataset, context='training')

    ## Parametros de entrada
    mlflow.log_param('Caracteristicas', caracteristicas)
    mlflow.log_param('Previsor', previsor)
    mlflow.log_param('Seed', sed)
    mlflow.log_param('Objetivo', objetivo)
    mlflow.log_param('Arvores', arvores)
    mlflow.log_param('Profundidade', profundidade)
    mlflow.log_param('Taxa de Aprendizado', tx_apren)
    
    #============ Modelagem ============#

    ## Treinamento do modelo
    modelo_11 = XGBClassifier(max_depth=profundidade, learning_rate=tx_apren, n_estimators=arvores, 
                             objective=objetivo, random_state=sed).fit(x_treino,y_treino)

    #============ Outputs ============#

    ## Registrando as mettricas do modelo
    fpr, tpr, thresholds =roc_curve(y_teste,
                                    modelo_11.predict_proba(x_teste)[:,1])
    
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
    mlflow.xgboost.log_model(modelo_11, 'modelo-11')

#%%
#============ Definição dos imputs ============#

## Modelo proposto: Modelo Grid Search

## Coleta dos metadados do dataset para o padrao do mlflow
dataset = mlflow.data.from_pandas(df_final_dummies)

## Formula utilizada no modelo
#caracteristicas = caracteristicas
#previsor = previsor
sed = 10
objetivo = 'binary:logistic'
param_grid = {
    'n_estimators': [100, 200, 500],
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [4, 6, 8],
    'subsample': [0.8, 1, None],
    'colsample_bytree': [0.8, 1, None],
    'gamma': [0, 0.1, 0.3]
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
    mlflow.log_param('Objetivo', objetivo)
    mlflow.log_param('K-fold', kfold)
    mlflow.log_param('Scoring', scor)
    mlflow.log_param('Grade de parametros', param_grid)
    
    #============ Modelagem ============#

    ## Treinamento do modelo
    modelo_puro = XGBClassifier(random_state=sed, objective=objetivo,)
    modelo_gs = GridSearchCV(estimator=modelo_puro, param_grid=param_grid, cv=kfold, n_jobs=-1, verbose=2, scoring=scor).fit(x_treino,y_treino)

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
objetivo = 'binary:logistic'
param_grid = {
    'n_estimators': [100, 200, 500],
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [4, 6, 8],
    'subsample': [0.8, 1, None],
    'colsample_bytree': [0.8, 1, None],
    'gamma': [0, 0.1, 0.3]
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
    mlflow.log_param('Objetivo', objetivo)
    mlflow.log_param('K-fold', kfold)
    mlflow.log_param('Scoring', scor)
    mlflow.log_param('Grade de parametros', param_grid)
    
    #============ Modelagem ============#

    ## Treinamento do modelo
    modelo_puro = XGBClassifier(random_state=sed, objective=objetivo,)
    modelo_rs = RandomizedSearchCV(estimator=modelo_puro, param_distributions=param_grid, n_iter=50, cv=kfold, verbose=2, random_state=sed, n_jobs=-1, scoring=scor).fit(x_treino,y_treino)

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
with mlflow.start_run(run_name='Modelo 9 Graficos Finais'):

    dados_plotagem = functions.espec_sens(observado = y_teste,
                                predicts = modelo_9.predict_proba(x_teste)[:,1])
    dados_plotagem

    ## Grafico cutoffs, sensitividade e especificidade
    fig1, ax = plt.subplots()
    with plt.style.context('seaborn-v0_8-whitegrid'):
        ax.plot(dados_plotagem.cutoffs,dados_plotagem.sensitividade, marker='o',
            color='indigo', markersize=8)
        ax.plot(dados_plotagem.cutoffs,dados_plotagem.especificidade, marker='o',
            color='limegreen', markersize=8)
    ax.set_xlabel('Cuttoff')
    ax.set_ylabel('Sensitividade / Especificidade')
    ax.set_xticks(np.arange(0, 1.1, 0.2))
    ax.set_yticks(np.arange(0, 1.1, 0.2))
    ax.legend(['Sensitividade', 'Especificidade'])
    mlflow.log_figure(
        fig1, 'sens_espe.png'
    )


    ## Matriz de confusao - previsoes1
    fig2 = functions.matriz_confusao(observado=y_teste,
                    predicts=modelo_9.predict_proba(x_teste)[:,1],
                    cutoff=0.45)
    mlflow.log_figure(
        fig2, 'matrix_conf.png'
    )

# %%

## Servir o modelo em uma api
# Colocar no bash
# Serve pra indicar qual porta esta o mlflow
# export MLFLOW_TRACKING_URI=http://localhost:5000

# Serve o modelo
# mlflow models serve -m "models:/acidentes/1" -p 5200 --env-manager=local

#%%
## Usando API pra fazer o predict
df_novos_dados = df_final_dummies.drop(columns={'Unnamed: 0', 'km','municipio','data','acidentes','mes','uf'}).iloc[0].T
df_novos_dados = df_novos_dados.to_frame().T 

#%%
dados_transformados = json.dumps(
    {'dataframe_records': df_novos_dados.to_dict(orient='records')}
)

#%%
response = requests.post(
    url='http://127.0.0.1:5200/invocations',
    data=dados_transformados,
    headers={'Content-Type': 'application/json'}
)

#%%
response.text

# %%
