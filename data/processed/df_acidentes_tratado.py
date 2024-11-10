
from pathlib import Path
import pandas as pd
import numpy as np
from unidecode import unidecode
import parametros

#Link = https://dados.antt.gov.br/dataset/acidentes-rodovias
#AUTOPISTA FERNÃO DIAS

current_dir = Path(__file__).resolve().parent

data_path = current_dir.parent / 'raw' / 'dados_acidente.csv'


df_acidentes = pd.read_csv(data_path, 
                           sep = ';', encoding = 'cp1252')

df_acidentes = df_acidentes.drop(columns=['id', 'dia_semana', 'horario', 'causa_acidente', 'classificacao_acidente',
                                          'fase_dia', 'tipo_pista', 'tracado_via', 'uso_solo', 'pessoas', 'mortos',
                                          'feridos_leves', 'feridos_graves', 'ilesos', 'ignorados', 'feridos',
                                          'veiculos', 'regional', 'delegacia', 'uop', 'condicao_metereologica','tipo_acidente'])



df_acidentes['data_inversa'] = pd.to_datetime(df_acidentes['data_inversa'])
df_acidentes['uf'] = df_acidentes['uf'].astype('category')
df_acidentes['municipio'] = df_acidentes['municipio'].astype('category')

df_acidentes['km'] = df_acidentes['km'].str.replace(',', '.')
df_acidentes['km'] = df_acidentes['km'].astype(float).round()
df_acidentes = df_acidentes.dropna()
df_acidentes['km'] = df_acidentes['km'].astype(int)

df_acidentes['latitude'] = df_acidentes['latitude'].str.replace(',', '.').astype(float)
df_acidentes['longitude'] = df_acidentes['longitude'].str.replace(',', '.').astype(float)

df_acidentes['municipio'] = df_acidentes['municipio'].str.lower().apply(unidecode).astype('category')
df_acidentes = df_acidentes[df_acidentes['municipio'].isin(parametros.cidades)]

df_acidentes = df_acidentes.loc[(df_acidentes['data_inversa'] >= parametros.data_inicio) 
                                          & (df_acidentes['data_inversa'] <= parametros.data_fim)]
df_acidentes = df_acidentes[df_acidentes['br'] == 381]
df_acidentes = df_acidentes[(df_acidentes['uf'] == 'MG') | (df_acidentes['uf'] == 'SP')]
df_acidentes = df_acidentes[((df_acidentes['municipio'] == 'betim') & (df_acidentes['km']>=477))|
                             (df_acidentes['municipio'] != 'betim')]

df_acidentes['sentido_crescente'] = np.where((df_acidentes['sentido_via'] == 'Crescente'), 1, 0)
df_acidentes['sentido_decrescente'] = np.where((df_acidentes['sentido_via'] == 'Decrescente'), 1, 0)
df_acidentes['sentido_crescente'] = df_acidentes['sentido_crescente'].astype(bool)
df_acidentes['sentido_decrescente'] = df_acidentes['sentido_decrescente'].astype(bool)
df_acidentes = df_acidentes.drop(columns=['br', 'sentido_via'])

df_acidentes = df_acidentes.groupby(['data_inversa', 'km', 'municipio', 
                                     'sentido_crescente', 'sentido_decrescente',
                                     'uf', 'latitude', 'longitude'],observed=True).size().reset_index(name = 'acidentes')

df_acidentes = df_acidentes.reset_index(level=None, drop=True, inplace=False, col_level=0, col_fill='')

variaveis_a_manter = ['df_acidentes']

for var in list(globals().keys()):
    if var not in variaveis_a_manter and not var.startswith('__'):
        del globals()[var]

