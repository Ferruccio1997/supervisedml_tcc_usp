#%%
import pandas as pd # manipulação de dados em formato de dataframe
import numpy as np # operações matemáticas
import seaborn as sns # visualização gráfica
import matplotlib.pyplot as plt # visualização gráfica
import matplotlib.image as mpimg
from sklearn.metrics import confusion_matrix, accuracy_score, ConfusionMatrixDisplay, recall_score # acuracia e matriz e confusao
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
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
    plt.savefig('rf_matriz_conf.png', dpi=600)
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
## Construcao do modelo manual - testado max_depth [4,8,10,12,11]
algoritmo_rf1 = RandomForestClassifier(n_estimators=500, criterion='gini', random_state=10, max_depth=4)

algoritmo_rf1.fit(x_treino, y_treino)

algoritmo_rf2 = RandomForestClassifier(n_estimators=500, criterion='gini', random_state=10, max_depth=11)

algoritmo_rf2.fit(x_treino, y_treino)

#%%
## GridSearch
rf_gs = RandomForestClassifier(random_state=10, criterion='gini')

# Definindo os parâmetros para a busca
param_grid = {
    'n_estimators': [100, 200, 500, 1000],  # Número de árvores
    'max_depth': [4, 6, 8, 10, 11],  # Profundidade máxima
    'min_samples_split': [2, 5, 10, 15, None],  # Número mínimo de amostras para dividir um nó
    'min_samples_leaf': [1, 2, 4, 10, None],  # Número mínimo de amostras em uma folha
    'bootstrap': [True, False],  # Amostragem com ou sem reposição
    'max_features': ['auto', 'sqrt', 'log2']  # Número de variáveis a considerar para divisão
}

# Realizando a busca com GridSearchCV
grid_search = GridSearchCV(estimator=rf_gs, param_grid=param_grid, scoring='roc_auc', cv=3, n_jobs=-1, verbose=2)
grid_search.fit(x_treino, y_treino)

#%%
## RandomSearch
rf_rs = RandomForestClassifier(random_state=10, criterion='gini')

# Definindo o espaço de busca para o ajuste de hiperparâmetros
param_dist = {
    'n_estimators': [100, 200, 500, 1000],  # Número de árvores
    'max_depth': [4, 6, 8, 10, 11],  # Profundidade máxima
    'min_samples_split': [2, 5, 10, 15, None],  # Número mínimo de amostras para dividir um nó
    'min_samples_leaf': [1, 2, 4, 10, None],  # Número mínimo de amostras em uma folha
    'bootstrap': [True, False],  # Amostragem com ou sem reposição
    'max_features': ['auto', 'sqrt', 'log2']  # Número de variáveis a considerar para divisão
}

# Realizando a busca com RandomizedSearchCV
random_search = RandomizedSearchCV(estimator=rf_rs, param_distributions=param_dist, scoring='roc_auc', n_iter=50, cv=3, verbose=2, random_state=10, n_jobs=-1)
random_search.fit(x_treino, y_treino)

#%%
## Previsoes
dict_previsoes = {
    'previsoes0': algoritmo_rf1.predict_proba(x_teste)[:,1],
    'previsoes1': algoritmo_rf2.predict_proba(x_teste)[:,1],
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
# ## Matriz de confusao
# matriz_confusao(observado=y_teste,
#                 predicts=previsoes,
#                 cutoff=0.40)

# # %%
# dados_plotagem = espec_sens(observado = y_teste,
#                             predicts = previsoes)
# dados_plotagem

# ## Grafico cutoffs, sensitividade e especificidade
# plt.figure(figsize=(15,10))
# with plt.style.context('seaborn-v0_8-whitegrid'):
#     plt.plot(dados_plotagem.cutoffs,dados_plotagem.sensitividade, marker='o',
#          color='indigo', markersize=8)
#     plt.plot(dados_plotagem.cutoffs,dados_plotagem.especificidade, marker='o',
#          color='limegreen', markersize=8)
# plt.xlabel('Cuttoff', fontsize=20)
# plt.ylabel('Sensitividade / Especificidade', fontsize=20)
# plt.xticks(np.arange(0, 1.1, 0.2), fontsize=14)
# plt.yticks(np.arange(0, 1.1, 0.2), fontsize=14)
# plt.legend(['Sensitividade', 'Especificidade'], fontsize=20)
# plt.savefig('rf_senst_espec.png', dpi=600)
# plt.show()

# # %%


