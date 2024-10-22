
import numpy as np
import pandas as pd
from unidecode import unidecode
from datetime import datetime
from sklearn.preprocessing import StandardScaler

import parametros
import funcoes
from df_acidentes_tratado import df_acidentes
from df_chuva_tratado import df_unido_cidades
from df_velocidade_tratado import df_velocidade
from dfs_estrada_tratado import df_faixa, df_perfil_terreno, df_iluminacao


df_final = df_velocidade
df_final = df_final.reset_index(level=None, drop=False, inplace=False, col_level=0, col_fill='')


## Acidentes
df_final_cross = df_final.merge(df_acidentes, left_on=['data','municipio', 'uf', 'sentido_crescente', 'sentido_decrescente'], 
                                right_on=['data_inversa', 'municipio', 'uf', 'sentido_crescente', 'sentido_decrescente'], how='left')

df_final_cross['dist'] = df_final_cross.apply(lambda row: abs(row['km_x'] - row['km_y']), axis=1, result_type='expand')

df_final_filtered = df_final_cross.drop(columns=['km_x','municipio','data','latitude_x','longitude_x',
                                                  'velocidade_Comercial','velocidade_Moto','velocidade_Passeio',
                                                  'velocidade_Ônibus','volume_Comercial','volume_Moto',
                                                  'volume_Passeio','volume_Ônibus',
                                                  'data_inversa','km_y','uf','latitude_y','longitude_y',
                                                  'acidentes', 'uf'])

df_final_filtered = df_final_filtered.groupby(['index'], observed=True).agg({'dist': 'min'}).reset_index()

df_final_filtered = df_final_cross.merge(df_final_filtered[['index', 'dist']], on=['index', 'dist'], how='inner')
df_final_filtered['dist'] = df_final_filtered['dist'].fillna(0)

df_final_filtered = df_final_filtered[df_final_filtered['dist'] <= 10]

df_final_filtered = df_final_filtered.drop(columns=['index','data_inversa','km_y','latitude_y','longitude_y','dist'])

df_final_filtered['acidentes'] = np.where((df_final_filtered['acidentes'] >= 1), 1, 0)

df_sem_acid = df_final_filtered[df_final_filtered['acidentes'] == 0]
df_com_acid = df_final_filtered[df_final_filtered['acidentes'] == 1]

df_sem_acid = df_sem_acid.sample(n=675, random_state=42)

df_final = pd.concat([df_sem_acid, df_com_acid], axis=0)

df_final = df_final.reset_index(level=None, drop=False, inplace=False, col_level=0, col_fill='')


## Chuva
df_final_cross = df_final.merge(df_unido_cidades, left_on=['data','municipio'], 
                                right_on=['data', 'municipio'], how='left')

df_final_cross['dist'] = df_final_cross.apply(lambda row: funcoes.calcula_distancia(row['latitude_x'], row['latitude'], row['longitude_x'], row['longitude']), axis=1, result_type='expand')

df_final_filtered = df_final_cross.drop(columns=['km_x','municipio','data','sentido_crescente','sentido_decrescente',
                                                  'latitude_x','longitude_x','uf','velocidade_Comercial','velocidade_Moto',
                                                  'velocidade_Passeio','velocidade_Ônibus','volume_Comercial',
                                                  'volume_Moto','volume_Passeio','volume_Ônibus',
                                                  'acidentes','latitude','longitude','chuva'])

df_final_filtered = df_final_filtered.groupby(['index'], observed=True).agg({'dist': 'min'}).reset_index()

df_final_filtered = df_final_cross.merge(df_final_filtered[['index', 'dist']], on=['index', 'dist'], how='inner')

df_final_filtered = df_final_filtered.drop(columns=['latitude','longitude','dist'])

df_final_filtered['chuva'] = np.where((df_final_filtered['chuva'] == 0), 0, 1)

df_final = df_final_filtered.drop(columns=['latitude_x','longitude_x','sentido_decrescente'])

df_final = df_final.rename(columns={'km_x': 'km'})

df_final['dia_da_semana'] = df_final_cross.apply(lambda row: funcoes.feriado_fds(row['data']), axis=1, result_type='expand')

df_final['mes'] = df_final['data'].dt.month


## Estrada
dfs = {
    'df_faixa': df_faixa,
    'df_perfil_terreno': df_perfil_terreno,
    'df_iluminacao': df_iluminacao
}

for df_name, df in dfs.items():
    df_final_cross = df_final.merge(df, left_on=['sentido_crescente', 'uf'], 
                                    right_on=['sentido', 'uf'], how='left')

    df_final_filtered = df_final_cross.query('km_m_inicial <= km <= km_m_final')

    if df_name == 'df_iluminacao':
        df_final_filtered['iluminacao'] = 1
        df_final_filtered = df_final.merge(df_final_filtered[['index','iluminacao']],left_on=['index'],right_on=['index'],how='left')

    else:
        df_final_filtered = df_final_filtered.drop(columns=['km_m_inicial','km_m_final', 'sentido'])

    df_final = df_final_filtered

df_final = df_final.fillna(0)

df_final = df_final.drop(columns = ['index'])

df_final['iluminacao'] = df_final['iluminacao'].astype(int)

df_final = df_final.replace(" ", "_", regex=True)

df_final_dummies = pd.get_dummies(df_final, columns=['dia_da_semana', 'tipo_faixa', 'tipo_perfil_de_terreno'],dtype=int,drop_first=True)

## Normalização
scaler = StandardScaler()
df_final_dummies[['velocidade_Comercial', 'velocidade_Moto', 'velocidade_Passeio',
       'velocidade_Ônibus', 'volume_Comercial', 'volume_Moto',
       'volume_Passeio', 'volume_Ônibus']] = scaler.fit_transform(df_final_dummies[['velocidade_Comercial', 'velocidade_Moto', 'velocidade_Passeio',
       'velocidade_Ônibus', 'volume_Comercial', 'volume_Moto',
       'volume_Passeio', 'volume_Ônibus']])

variaveis_a_manter = ['df_final', 'df_final_dummies']

for var in list(globals().keys()):
    if var not in variaveis_a_manter and not var.startswith('__'):
        del globals()[var]

