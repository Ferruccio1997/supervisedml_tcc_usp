#%%
import pandas as pd # manipulação de dados em formato de dataframe
import numpy as np # operações matemáticas
import seaborn as sns # visualização gráfica
import matplotlib.pyplot as plt # visualização gráfica
from sklearn.metrics import confusion_matrix, accuracy_score, ConfusionMatrixDisplay, recall_score # acuracia e matriz e confusao
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from xgboost import XGBClassifier
from sklearn.metrics import roc_curve, auc # metricas curva_roc e gini

from df_final import df_final_dummies # base de dados

#%%
def matriz_confusao(predicts, observado, cutoff):
    
    values = predicts
    
    predicao_binaria = []
        
    for item in values:
        if item < cutoff:
            predicao_binaria.append(0)
        else:
            predicao_binaria.append(1)
           
    cm = confusion_matrix(predicao_binaria, observado)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.xlabel('True')
    plt.ylabel('Classified')
    plt.gca().invert_xaxis()
    plt.gca().invert_yaxis()
    plt.savefig('xgb_matriz_conf.png', dpi=600)
    plt.show()
        
    sensitividade = recall_score(observado, predicao_binaria, pos_label=1)
    especificidade = recall_score(observado, predicao_binaria, pos_label=0)
    acuracia = accuracy_score(observado, predicao_binaria)

    #Visualizando os principais indicadores desta matriz de confusão
    indicadores = pd.DataFrame({'Sensitividade':[sensitividade],
                                'Especificidade':[especificidade],
                                'Acurácia':[acuracia]})
    return indicadores

def espec_sens(observado,predicts):
    
    # adicionar objeto com os valores dos predicts
    values = predicts
    
    # range dos cutoffs a serem analisados em steps de 0.01
    cutoffs = np.arange(0,1.01,0.01)
    
    # Listas que receberão os resultados de especificidade e sensitividade
    lista_sensitividade = []
    lista_especificidade = []
    
    for cutoff in cutoffs:
        
        predicao_binaria = []
        
        # Definindo resultado binário de acordo com o predict
        for item in values:
            if item >= cutoff:
                predicao_binaria.append(1)
            else:
                predicao_binaria.append(0)
                
        # Cálculo da sensitividade e especificidade no cutoff
        sensitividade = recall_score(observado, predicao_binaria, pos_label=1)
        especificidadee = recall_score(observado, predicao_binaria, pos_label=0)
        
        # Adicionar valores nas listas
        lista_sensitividade.append(sensitividade)
        lista_especificidade.append(especificidadee)
        
    # Criar dataframe com os resultados nos seus respectivos cutoffs
    resultado = pd.DataFrame({'cutoffs':cutoffs,'sensitividade':lista_sensitividade,'especificidade':lista_especificidade})
    return resultado

#%%
df_final_dummies.columns

#%%
## Transforma em array por otmizaçao
caracteristicas = df_final_dummies.drop(columns=['km','municipio','data','acidentes','mes','uf'])

previsor = df_final_dummies.drop(columns=['km', 'municipio', 'data', 'sentido_crescente', 'uf',
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
## Construcao do modelo manual - testado max_depth [3,4,8,10,12,11] e learning_rate [0.05, 0.01, 0.005]
algoritmo_xgb1 = XGBClassifier(max_depth=4, learning_rate=0.05, n_estimators=500, 
                             objective='binary:logistic', random_state=10)


algoritmo_xgb1.fit(x_treino, y_treino)

algoritmo_xgb2 = XGBClassifier(max_depth=4, learning_rate=0.01, n_estimators=500, 
                             objective='binary:logistic', random_state=10)


algoritmo_xgb2.fit(x_treino, y_treino)

#%%
## GridSearch
xgb_gs = XGBClassifier(random_state=10, objective='binary:logistic',)

# Definindo os parâmetros para a busca
param_grid = {
    'n_estimators': [100, 200, 500],
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [4, 6, 8],
    'subsample': [0.8, 1, None],
    'colsample_bytree': [0.8, 1, None],
    'gamma': [0, 0.1, 0.3]
}

# Realizando a busca com GridSearchCV
grid_search = GridSearchCV(estimator=xgb_gs, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2, scoring='roc_auc')
grid_search.fit(x_treino, y_treino)

#%% a
## RandomSearch
xgb_rs = XGBClassifier(random_state=10, objective='binary:logistic',)

# Definindo o espaço de busca para o ajuste de hiperparâmetros
param_dist = {
    'n_estimators': [100, 200, 500],
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [4, 6, 8],
    'subsample': [0.8, 1, None],
    'colsample_bytree': [0.8, 1, None],
    'gamma': [0, 0.1, 0.3]
}

# Realizando a busca com RandomizedSearchCV
random_search = RandomizedSearchCV(estimator=xgb_rs, param_distributions=param_dist, n_iter=50, cv=3, verbose=2, random_state=10, n_jobs=-1, scoring='roc_auc')
random_search.fit(x_treino, y_treino)

#%%
## Previsoes
dict_previsoes = {
    'previsoes0': algoritmo_xgb1.predict_proba(x_teste)[:,1],
    'previsoes1': algoritmo_xgb2.predict_proba(x_teste)[:,1],
    'previsoes2': grid_search.predict_proba(x_teste)[:,1],
    'previsoes3': random_search.predict_proba(x_teste)[:,1]
}

#%%
for i in range(4):
    ## Construcaoo da curva ROC
    fpr, tpr, thresholds =roc_curve(y_teste,
                                    dict_previsoes[f'previsoes{i}'])
    roc_auc = auc(fpr, tpr)

    ## Calculo do coeficiente de GINI
    gini = (roc_auc - 0.5)/(0.5)

    print(round(roc_auc,4), round(gini,4))


#%%
## Matriz de confusao - previsoes1
matriz_confusao(observado=y_teste,
                predicts=dict_previsoes['previsoes1'],
                cutoff=0.45)

# %%
dados_plotagem = espec_sens(observado = y_teste,
                            predicts = dict_previsoes['previsoes1'])
dados_plotagem

## Grafico cutoffs, sensitividade e especificidade
plt.figure(figsize=(15,10))
with plt.style.context('seaborn-v0_8-whitegrid'):
    plt.plot(dados_plotagem.cutoffs,dados_plotagem.sensitividade, marker='o',
         color='indigo', markersize=8)
    plt.plot(dados_plotagem.cutoffs,dados_plotagem.especificidade, marker='o',
         color='limegreen', markersize=8)
plt.xlabel('Cuttoff', fontsize=20)
plt.ylabel('Sensitividade / Especificidade', fontsize=20)
plt.xticks(np.arange(0, 1.1, 0.2), fontsize=14)
plt.yticks(np.arange(0, 1.1, 0.2), fontsize=14)
plt.legend(['Sensitividade', 'Especificidade'], fontsize=20)
plt.savefig('xgb_senst_espec.png', dpi=600)
plt.show()

# #%%
# ## Construcaoo da curva ROC
# fpr, tpr, thresholds =roc_curve(y_teste,
#                                 previsoes)
# roc_auc = auc(fpr, tpr)

# ## Calculo do coeficiente de GINI
# gini = (roc_auc - 0.5)/(0.5)

# plt.figure(figsize=(15,10))
# plt.plot(fpr, tpr, marker='o', color='darkorchid', markersize=10, linewidth=3)
# plt.plot(fpr, fpr, color='gray', linestyle='dashed')
# plt.title('Área abaixo da curva: %g' % round(roc_auc, 4) +
#           ' | Coeficiente de GINI: %g' % round(gini, 4), fontsize=22)
# plt.xlabel('1 - Especificidade', fontsize=20)
# plt.ylabel('Sensitividade', fontsize=20)
# plt.xticks(np.arange(0, 1.1, 0.2), fontsize=14)
# plt.yticks(np.arange(0, 1.1, 0.2), fontsize=14)
# plt.savefig('xgb_curva_roc.png', dpi=600)
# plt.show()

# # %%



# %%
