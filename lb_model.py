#%%
import pandas as pd # manipulação de dados em formato de dataframe
import numpy as np # operações matemáticas
import seaborn as sns # visualização gráfica
import matplotlib.pyplot as plt # visualização gráfica
import statsmodels.api as sm # estimação de modelos
from statstests.process import stepwise # procedimento Stepwise
from sklearn.metrics import roc_curve, auc # metricas curva_roc e gini
from sklearn.metrics import confusion_matrix, accuracy_score,\
    ConfusionMatrixDisplay, recall_score # acuracia e matriz e confusao

from df_final import df_final_dummies # base de dados

#%%
def matriz_confusao(predicts, observado, cutoff):
    
    values = predicts.values
    
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
    plt.savefig('lb_matriz_conf.png', dpi=600)
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
    values = predicts.values
    
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
## Construcao da eq. do modelo
lista_colunas = list(df_final_dummies.drop(columns=['km','municipio','data','acidentes','mes','uf']).columns)

formula_dummies_modelo = ' + '.join(lista_colunas)
formula_dummies_modelo = "acidentes ~ " + formula_dummies_modelo
print("Fórmula utilizada: ",formula_dummies_modelo)

## Estimacao do modelo
modelo_final = sm.Logit.from_formula(formula_dummies_modelo,
                                               df_final_dummies).fit()

#%%
## Parâmetros do modelo
modelo_final.summary()

#%%
## Procedimento Stepwise - dado que tem variávei pouco significativas
step_modelo_final = stepwise(modelo_final, pvalue_limit=0.05)

#%%
## Add as variáveis preditas
df_final_dummies['phat'] = step_modelo_final.predict()

# Matriz de confusão para cutoff = 0.46
matriz_confusao(observado=df_final_dummies['acidentes'],
                predicts=df_final_dummies['phat'],
                cutoff=0.42)

#%%
## Dados sobre cutoffs, sensitividade e especificidade
dados_plotagem = espec_sens(observado = df_final_dummies['acidentes'],
                            predicts = df_final_dummies['phat'])
dados_plotagem

#%%
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
plt.savefig('lb_senst_espec.png', dpi=600)
plt.show()

#%%
## Construcaoo da curva ROC
fpr, tpr, thresholds =roc_curve(df_final_dummies['acidentes'],
                                df_final_dummies['phat'])
roc_auc = auc(fpr, tpr)

## Calculo do coeficiente de GINI
gini = (roc_auc - 0.5)/(0.5)

plt.figure(figsize=(15,10))
plt.plot(fpr, tpr, marker='o', color='darkorchid', markersize=10, linewidth=3)
plt.plot(fpr, fpr, color='gray', linestyle='dashed')
plt.title('Área abaixo da curva: %g' % round(roc_auc, 4) +
          ' | Coeficiente de GINI: %g' % round(gini, 4), fontsize=22)
plt.xlabel('1 - Especificidade', fontsize=20)
plt.ylabel('Sensitividade', fontsize=20)
plt.xticks(np.arange(0, 1.1, 0.2), fontsize=14)
plt.yticks(np.arange(0, 1.1, 0.2), fontsize=14)
plt.savefig('lb_curva_roc.png', dpi=600)
plt.show()

# %%
