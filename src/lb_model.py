#%%
from pathlib import Path
import mlflow
import json
import requests
import numpy as np # operações matemáticas
import pandas as pd
import matplotlib.pyplot as plt # visualização gráfica
import statsmodels.api as sm # estimação de modelos
from statstests.process import stepwise # procedimento Stepwise
from sklearn.metrics import roc_curve, auc # metricas curva_roc e gini

current_dir = Path(__file__).resolve().parent

data_path = current_dir.parent / 'data' / 'processed' / 'df_final_dummies.csv'

#%%
df_final_dummies = pd.read_csv(data_path)

df_final_dummies.columns

#%%
## Construcao da eq. do modelo
lista_colunas = list(df_final_dummies.drop(columns=['Unnamed: 0', 'km','municipio','data','acidentes','mes','uf']).columns)

formula_dummies_modelo = ' + '.join(lista_colunas)
formula_dummies_modelo = "acidentes ~ " + formula_dummies_modelo
print("Fórmula utilizada: ",formula_dummies_modelo)

## Estimacao do modelo
modelo_local = sm.Logit.from_formula(formula_dummies_modelo,
                                               df_final_dummies).fit()


#%%
## Parâmetros do modelo
modelo_local.summary()

fpr, tpr, thresholds =roc_curve(df_final_dummies['acidentes'],
                                    modelo_local.predict())
    
roc_auc = auc(fpr, tpr)
gini = (roc_auc - 0.5)/(0.5)

print(roc_auc)


#%%
#============ Inicializa Mlflow ============#

## Aponta a instancia do mlflow 
mlflow.set_tracking_uri("http://localhost:5000")

#%%
## Cria o eperimento
mlflow.set_experiment(experiment_name='Regressão Logística Binária - acidentes rodoviários')


#%%
#============ Definição dos imputs ============#

## Modelo proposto: Modelo sem  Stepwise

## Coleta dos metadados do dataset para o padrao do mlflow
dataset = mlflow.data.from_pandas(df_final_dummies)

## Formula utilizada no modelo
formula = formula_dummies_modelo

#%%
with mlflow.start_run(run_name='Modelo sem Stepwise'):

    #============ Inputs ============#

    ## Coleta dos metadados do dataset. O contexto pode ser utilizado
    ## para adicionar mais uma informacao de contexto do dataset, por
    ## exemplo: se ele é um dataset de treino ou teste
    mlflow.log_input(dataset, context='training')

    ## Parametros de entrada
    mlflow.log_param('Formula', formula)

    #============ Modelagem ============#

    ## Treinamento do modelo
    modelo_sem_sw = sm.Logit.from_formula(formula=formula, data=df_final_dummies).fit()

    #============ Outputs ============#

    ## Registrando as mettricas do modelo
    fpr, tpr, thresholds =roc_curve(df_final_dummies['acidentes'],
                                    modelo_sem_sw.predict())
    
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
        fig, 'lb_curva_roc.png'
    )

    ## Obtem sumario do modelo
    modelo_sem_sw_summary_texto = modelo_sem_sw.summary().as_text()

    ## Registra o sumario
    mlflow.log_text(
        modelo_sem_sw_summary_texto, 'summary.txt'
    )

    ## Carregamento do modelo no experimento
    mlflow.statsmodels.log_model(modelo_sem_sw, 'modelo-sem-sw')


#%%
with mlflow.start_run(run_name='Modelo com Stepwise'):

    #============ Inputs ============#

    ## Coleta dos metadados do dataset. O contexto pode ser utilizado
    ## para adicionar mais uma informacao de contexto do dataset, por
    ## exemplo: se ele é um dataset de treino ou teste
    mlflow.log_input(dataset, context='training')

    ## Parametros de entrada
    mlflow.log_param('Formula', formula)

    #============ Modelagem ============#

    ## Treinamento do modelo
    modelo_com_sw = stepwise(modelo_sem_sw,pvalue_limit=0.05)

    #============ Outputs ============#

    ## Registrando as mettricas do modelo
    fpr, tpr, thresholds =roc_curve(df_final_dummies['acidentes'],
                                    modelo_com_sw.predict())
    
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
        fig, 'lb_curva_roc.png'
    )

    ## Obtem sumario do modelo
    modelo_com_sw_summary_texto = modelo_com_sw.summary().as_text()

    ## Registra o sumario
    mlflow.log_text(
        modelo_com_sw_summary_texto, 'summary.txt'
    )

    ## Carregamento do modelo no experimento
    mlflow.statsmodels.log_model(modelo_com_sw, 'modelo-com-sw')

# %%
