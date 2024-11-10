from pathlib import Path
import pandas as pd
import numpy as np
from unidecode import unidecode
import parametros

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


# Link = https://dados.gov.br/dados/conjuntos-dados/volume-radar
current_dir = Path(__file__).resolve().parent
data_path = current_dir.parent / 'raw' / 'dados_velocidade.csv'

df_velocidade = pd.read_csv(data_path, sep = ';', encoding = 'cp1252')

df_velocidade = df_velocidade[df_velocidade['rodovia'] == 'BR-381']
df_velocidade = df_velocidade[(df_velocidade['uf'] == 'MG') | (df_velocidade['uf'] == 'SP')]
df_velocidade = df_velocidade[df_velocidade['tipo_de_pista'] == 'Principal']
df_velocidade = df_velocidade[df_velocidade['tipo_de_veiculo'] != 'Não classificado']

df_velocidade['municipio'] = df_velocidade['municipio'].str.lower().apply(unidecode)
df_velocidade = df_velocidade[df_velocidade['municipio'].isin(parametros.cidades)]

df_velocidade['sentido_crescente'] = np.where((df_velocidade['sentido_da_passagem'] == 'Crescente'), 1, 0)
df_velocidade['sentido_decrescente'] = np.where((df_velocidade['sentido_da_passagem'] == 'Decrescente'), 1, 0)

df_velocidade = df_velocidade.drop(columns=['faixa_da_passagem', 'concessionaria', 'identificador', 'rodovia', 'tipo_de_pista', 'sentido_da_passagem'])

df_velocidade['data_da_passagem'] = pd.to_datetime(df_velocidade['data_da_passagem'], format = '%d/%m/%Y')
df_velocidade = df_velocidade.loc[(df_velocidade['data_da_passagem'] >= parametros.data_inicio) & (df_velocidade['data_da_passagem'] <= parametros.data_fim)]

df_velocidade['velocidade'] = df_velocidade['velocidade'].apply(calcular_mediana)

df_velocidade['velocidadeXvolume'] = df_velocidade['velocidade'] * df_velocidade['volume_total']

df_velocidade = df_velocidade.groupby(['municipio', 'data_da_passagem', 'tipo_de_veiculo',  'sentido_crescente', 'sentido_decrescente', 'km_m', 'latitude', 'longitude', 'uf']).agg({'velocidadeXvolume': 'sum', 'volume_total': 'sum'})
df_velocidade = df_velocidade.reset_index()
df_velocidade.columns.name = None
df_velocidade.columns = ['_'.join(col).strip() if isinstance(col, tuple) else col for col in df_velocidade.columns.values]

df_velocidade = df_velocidade.rename(columns={'volume_total': 'volume'})

df_velocidade = df_velocidade.pivot_table(index = ['km_m', 'municipio', 'data_da_passagem', 'sentido_crescente', 'sentido_decrescente', 'latitude', 'longitude', 'uf'], columns = 'tipo_de_veiculo', values = {'velocidadeXvolume': 'sum', 'volume': 'sum'},aggfunc={'velocidadeXvolume': 'sum', 'volume': 'sum'}, fill_value='')
df_velocidade = df_velocidade.reset_index()
df_velocidade.columns.name = None
df_velocidade.columns = ['_'.join(col).strip() if isinstance(col, tuple) else col for col in df_velocidade.columns.values]

df_velocidade['velocidadeXvolume_Comercial'] = df_velocidade['velocidadeXvolume_Comercial'].replace('','0').astype(float)
df_velocidade['velocidadeXvolume_Moto'] = df_velocidade['velocidadeXvolume_Moto'].replace('','0').astype(float)
df_velocidade['velocidadeXvolume_Passeio'] = df_velocidade['velocidadeXvolume_Passeio'].replace('','0').astype(float)
df_velocidade['velocidadeXvolume_Ônibus'] = df_velocidade['velocidadeXvolume_Ônibus'].replace('','0').astype(float)

df_velocidade['volume_Comercial'] = df_velocidade['volume_Comercial'].replace('','0').astype(int)
df_velocidade['volume_Moto'] = df_velocidade['volume_Moto'].replace('','0').astype(int)
df_velocidade['volume_Passeio'] = df_velocidade['volume_Passeio'].replace('','0').astype(int)
df_velocidade['volume_Ônibus'] = df_velocidade['volume_Ônibus'].replace('','0').astype(int)

for veiculo in ['Comercial', 'Moto', 'Passeio', 'Ônibus']:
    df_velocidade[f'velocidadeXvolume_{veiculo}'] = df_velocidade[f'velocidadeXvolume_{veiculo}'] / df_velocidade[f'volume_{veiculo}']
    df_velocidade = df_velocidade.rename(columns={f'velocidadeXvolume_{veiculo}': f'velocidade_{veiculo}'})



df_velocidade = df_velocidade.rename(columns={'km_m_': 'km', 'municipio_': 'municipio', 'data_da_passagem_': 'data',
                                              'sentido_crescente_': 'sentido_crescente', 'sentido_decrescente_': 'sentido_decrescente',
                                              'latitude_': 'latitude', 'longitude_': 'longitude', 'uf_': 'uf'})


