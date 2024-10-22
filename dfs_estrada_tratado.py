
import numpy as np
import pandas as pd

links = {
    'faixa': 'https://dados.antt.gov.br/dataset/bcb0c299-6910-4027-879c-5f1ea3c88a34/resource/d466ebc8-d708-4e84-8779-5a527dffee54/download/dados_tipo_.json',
    'iluminacao': 'https://dados.antt.gov.br/dataset/fd66f90f-cc64-418c-aa71-6e3563e79d95/resource/a225868c-4171-4bd7-8f9a-f5726016469d/download/dados_da_iluminacao.json',
    'perfil_terreno': 'https://dados.antt.gov.br/dataset/2c146f6e-24a1-4f5c-93b7-a5d9e496b7d6/resource/338cbaa5-aabc-4ea7-951c-a1a4b8eae1dc/download/dados_perfil_do_terreno.json'
}

lista = []
for nome in links:
    x = f'df_{nome}'
    lista.append(x)
    df = pd.read_json(links[nome])
    df = pd.json_normalize(df[f'{df.columns[0]}'])

    df = df[(df['rodovia_uf'] == 'BR-381/MG') | (df['rodovia_uf'] == 'BR-381/SP')]

    df['sentido'] = np.where((df['sentido'] == 'Crescente'), 1, 0)
    df[['rodovia', 'uf']] = df['rodovia_uf'].str.split('/', expand=True)
    df[['km_m_inicial', 'km_m_final']] = df[['km_m_inicial', 'km_m_final']].apply(lambda x: x.str.replace(',', '.').astype(float))

    df = df.drop(columns=['concessionaria','ano_do_pnv_snv','rodovia_uf','tipo_pista','latitude_inicial',
                                    'longitude_inicial','latitude_final','longitude_final','rodovia'])

    df = df.reset_index(level=None, drop=True, inplace=False, col_level=0, col_fill='')

    globals()[f'df_{nome}'] = df

variaveis_a_manter = lista
for var in list(globals().keys()):
    if var not in variaveis_a_manter and not var.startswith('__'):
        del globals()[var]


