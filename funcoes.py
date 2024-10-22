
import math
import holidays
import parametros
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


## Calcula a mediana das velocidades    
def calcular_mediana(faixa):
    faixa = faixa.replace(' KM/h', '')
    faixa = faixa.replace(' K', '')
    faixa = faixa.replace('M', '')
    if '<=' in faixa:
        return int(faixa.split('<=')[-1])
    elif '>' in faixa:
        return int(faixa.split('>')[-1])
    else:
        min_vel, max_vel = map(int, faixa.split('-'))
        return (min_vel + max_vel)/2


## Determina se é feriado
def feriado_fds(date):
    # Verificar se é final de semana
    if date.weekday() >= 5:
        return 'Final de Semana' 
    return 'Dia de semana'


## Descreve variaveis categoricas
def describe_categorical(df):
    categorical_summary = {}
    for column in df.select_dtypes(include=['object', 'category', 'bool']).columns:
        categorical_summary[column] = {
            'count': df[column].count(),
            'unique': df[column].nunique(),
            'top': df[column].mode()[0],
            'freq': df[column].value_counts().iloc[0]
        }
    return pd.DataFrame(categorical_summary)


## Remove a linha que apresenta dia a mais que o mes 
def alterar_ou_remover(row):
    try:
        # Tenta substituir o dia na data
        nova_data = row['data'].replace(day=int(row['Dia']))
        return nova_data
    except ValueError:
        # Se for uma data inválida, retorna None para que possamos removê-la
        return None


## Calcula distancia entre dois pontos
def calcula_distancia(lat1deg, lat2deg, long1deg, long2deg):
    lat1 = math.radians(lat1deg)
    lat2 = math.radians(lat2deg)
    long1 = math.radians(long1deg)
    long2 = math.radians(long2deg)

    a = math.sin((lat2 - lat1)/2)**2 + math.cos(lat1)*math.cos(lat2)*(math.sin((long2-long1)/2)**2)
    dist = 2*6371*math.atan2(math.sqrt(a), math.sqrt(1-a))
    return dist

