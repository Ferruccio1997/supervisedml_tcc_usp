
import pandas as pd
from unidecode import unidecode
import parametros
import numpy as np
import funcoes
import pyodbc

# Link = https://www.snirh.gov.br/hidroweb/serieshistoricas
# Link = https://mapainterativo.cemaden.gov.br/#

for cidade in [c for c in parametros.cidades if c != 'santo antonio do amparo']:
    if cidade == 'betim': 
        for i in range(12):
            if i == 0:
                folder = f'C:\\Users\\Ferruccio\\OneDrive\\Área de Trabalho\\Conhecimento\\FACULDADE\\MBA USP\\TCC\\Dados\\dados_{cidade}'
                file = f'data.csv'
                raw_path = fr'{folder}\{file}'
                estacoes_file = f'estacoes.txt'
                estacoes_path = fr'{folder}\{estacoes_file}'
                with open(estacoes_path, 'r') as file:
                    conteudo = file.read()
                    estacoes = conteudo.split(',')
                df = pd.read_csv(raw_path, sep = ';', encoding = 'cp1252')
                df_unido = df

            else:
                folder = 'C:\\Users\\Ferruccio\\OneDrive\\Área de Trabalho\\Conhecimento\\FACULDADE\\MBA USP\\TCC\\Dados\\dados_betim'
                file = f'data ({i}).csv'
                raw_path = fr'{folder}\{file}'
                df = pd.read_csv(raw_path, sep = ';', encoding = 'cp1252')
                df_unido = pd.concat([df_unido, df], ignore_index=True)

            df_unido['nomeEstacao'] = df_unido['nomeEstacao'].str.lower().apply(unidecode)
            df_unido = df_unido[df_unido['nomeEstacao'].isin(estacoes)]
        df_unido_cidades = df_unido

    else:
        for i in range(12):
            if i == 0:
                folder = f'C:\\Users\\Ferruccio\\OneDrive\\Área de Trabalho\\Conhecimento\\FACULDADE\\MBA USP\\TCC\\Dados\\dados_{cidade}'
                file = f'data.csv'
                raw_path = fr'{folder}\{file}'
                estacoes_file = f'estacoes.txt'
                estacoes_path = fr'{folder}\{estacoes_file}'
                with open(estacoes_path, 'r') as file:
                    conteudo = file.read()
                    estacoes = conteudo.split(',')
                df = pd.read_csv(raw_path, sep = ';', encoding = 'cp1252')
                df_unido = df

            else:
                folder = f'C:\\Users\\Ferruccio\\OneDrive\\Área de Trabalho\\Conhecimento\\FACULDADE\\MBA USP\\TCC\\Dados\\dados_{cidade}'
                file = f'data ({i}).csv'
                raw_path = fr'{folder}\{file}'
                df = pd.read_csv(raw_path, sep = ';', encoding = 'cp1252')
                df_unido = pd.concat([df_unido, df], ignore_index=True)

        df_unido_cidades = pd.concat([df_unido_cidades, df_unido], ignore_index=True)

del df_unido

df_unido_cidades = df_unido_cidades.drop(columns=['codEstacao', 'uf', 'nomeEstacao'])


df_unido_cidades = df_unido_cidades.rename(columns={'ï»¿municipio': 'municipio', 
                                                    'datahora': 'data', 'valorMedida': 'chuva'})

df_unido_cidades['data'] = pd.to_datetime(df_unido_cidades['data'])
df_unido_cidades['data'] = df_unido_cidades['data'].dt.date
df_unido_cidades['data'] = pd.to_datetime(df_unido_cidades['data'])

df_unido_cidades['municipio'] = df_unido_cidades['municipio'].astype('category')

df_unido_cidades['chuva'] = df_unido_cidades['chuva'].str.replace(',', '.')
df_unido_cidades['chuva'] = df_unido_cidades['chuva'].astype(float)

df_unido_cidades['longitude'] = df_unido_cidades['longitude'].str.replace(',', '.')
df_unido_cidades['longitude'] = df_unido_cidades['longitude'].astype(float)

df_unido_cidades['latitude'] = df_unido_cidades['latitude'].str.replace(',', '.')
df_unido_cidades['latitude'] = df_unido_cidades['latitude'].astype(float)

df_unido_cidades = df_unido_cidades.groupby(['municipio', 'latitude', 'longitude', 'data'], observed=True).agg({'chuva': 'sum'}).reset_index()
df_unido_cidades.columns.name = None
df_unido_cidades.columns = ['_'.join(col).strip() if isinstance(col, tuple) else col for col in df_unido_cidades.columns.values]

df_unido_cidades['municipio'] = np.where((df_unido_cidades['municipio'] == 'SÃƒO PAULO'), 'sao paulo', df_unido_cidades['municipio'])
df_unido_cidades['municipio'] = np.where((df_unido_cidades['municipio'] == 'MAIRIPORÃƒ'), 'mairipora', df_unido_cidades['municipio'])
df_unido_cidades['municipio'] = df_unido_cidades['municipio'].astype('category')


db_path = r'C:\Users\Ferruccio\OneDrive\Área de Trabalho\Conhecimento\FACULDADE\MBA USP\TCC\Dados\banco_santoantonio.mdb'

conn_str = (
    r'DRIVER={Microsoft Access Driver (*.mdb, *.accdb)};'
    r'DBQ=' + db_path + ';'
)

conn = pyodbc.connect(conn_str)

query = '''
SELECT 
    'santo antonio do amparo' AS municipio,
    EST.Latitude AS latitude,
    EST.Longitude AS longitude,
    CHU.Data AS data,
    CHU.Chuva01,CHU.Chuva02,CHU.Chuva03,CHU.Chuva04,CHU.Chuva05,CHU.Chuva06,CHU.Chuva07,CHU.Chuva08,CHU.Chuva09,
    CHU.Chuva10,CHU.Chuva11,CHU.Chuva12,CHU.Chuva13,CHU.Chuva14,CHU.Chuva15,CHU.Chuva16,CHU.Chuva17,CHU.Chuva18,
    CHU.Chuva19,CHU.Chuva20,CHU.Chuva21,CHU.Chuva22,CHU.Chuva23,CHU.Chuva24,CHU.Chuva25,CHU.Chuva26,CHU.Chuva27,
    CHU.Chuva28,CHU.Chuva29,CHU.Chuva30,CHU.Chuva31
FROM Chuvas AS CHU 
LEFT JOIN Estacao AS EST ON (CHU.EstacaoCodigo = EST.Codigo)
WHERE
    YEAR(CHU.Data) = 2023
'''

df_santoantonio = pd.read_sql(query, conn)

conn.close()

for i in range(31):
    if i <= 8:
        colum_rename = 'Chuva0'+str(i+1)
        dia = '0'+str(i+1)

    else:
        colum_rename = 'Chuva'+str(i+1)
        dia = str(i+1)

    df_santoantonio = df_santoantonio.rename(columns={colum_rename: dia})

df_santoantonio['municipio'] = df_santoantonio['municipio'].astype('category')

df_santoantonio = pd.melt(df_santoantonio, id_vars=['municipio', 'latitude', 
                                                    'longitude', 'data'], value_vars=['01', '02', '03', '04', '05', '06', '07',
                                                                                       '08', '09', '10', '11', '12', '13', '14',
                                                                                        '15', '16', '17', '18', '19', '20', '21',
                                                                                        '22','23', '24', '25', '26', '27', '28',
                                                                                        '29', '30', '31'], var_name='Dia', value_name='chuva')

df_santoantonio['Dia'] = df_santoantonio['Dia'].astype('category')

df_santoantonio['chuva'] = df_santoantonio['chuva'].astype(float)
df_santoantonio['chuva'] = df_santoantonio['chuva'].fillna(0)
df_santoantonio['data'] = pd.to_datetime(df_santoantonio['data'], format = '%d/%m/%Y')

df_santoantonio['Data_modificada'] = df_santoantonio.apply(lambda row: funcoes.alterar_ou_remover(row), axis=1)
df_santoantonio = df_santoantonio.dropna(subset=['Data_modificada'])
df_santoantonio['data'] = df_santoantonio['Data_modificada']
df_santoantonio = df_santoantonio.drop(columns=['Data_modificada'])
df_santoantonio = df_santoantonio.drop(columns=['Dia'])


df_unido_cidades = pd.concat([df_unido_cidades, df_santoantonio], ignore_index=True)

df_unido_cidades['municipio'] = df_unido_cidades['municipio'].str.lower().apply(unidecode).astype('category')

df_unido_cidades = df_unido_cidades.reset_index(level=None, drop=True, inplace=False, col_level=0, col_fill='')

variaveis_a_manter = ['df_unido_cidades']

for var in list(globals().keys()):
    if var not in variaveis_a_manter and not var.startswith('__'):
        del globals()[var]

