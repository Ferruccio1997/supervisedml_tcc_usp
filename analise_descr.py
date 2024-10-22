#%%
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from df_final import df_final

df_agrup_dict={}


for coluna in df_final.columns:
    df_agrup_dict[f'df_agrup_{coluna}'] = df_final.groupby(coluna).agg(acidentes_sum=('acidentes', 'sum')).reset_index()
    if coluna == 'mes':
         continue
    else:
        df_agrup_dict[f'df_agrup_{coluna}'] = df_agrup_dict[f'df_agrup_{coluna}'].sort_values(by='acidentes_sum',ascending=False)
#%%
df_final.columns
plt.rc('font', family='Arial', size=11)
#%%
## Grafico de barra
for coluna in df_final.drop(columns={'data','sentido_crescente','uf','velocidade_Comercial','velocidade_Moto',
                                     'velocidade_Passeio','velocidade_Ônibus','volume_Comercial','volume_Moto',
                                     'volume_Passeio','volume_Ônibus','acidentes','chuva','dia_da_semana',
                                     'mes','tipo_faixa','iluminacao'}).columns:
    with sns.axes_style("whitegrid"):
        plt.figure(figsize=(15,10))
        ax = sns.barplot(data=df_agrup_dict[f'df_agrup_{coluna}'],x=coluna, y='acidentes_sum',
                    color=sns.color_palette('Blues_d')[0])#, edgecolor='white')
        plt.xlabel(coluna)
        plt.ylabel('Quantidade de acidentes')
        plt.xticks()
        plt.yticks()

        for p in ax.patches:
                ax.annotate(format(p.get_height(), '.0f'),  # Formato do número
                        (p.get_x() + p.get_width() / 2., p.get_height()),  # Posição
                        ha = 'center', va = 'center', 
                        xytext = (0, 9),  # Distância do número para a barra
                        textcoords = 'offset points', fontsize=14)

        plt.savefig(f'ad_barra_{coluna}.png', dpi=600)
        plt.show()

#%%
## Grafico de Rosca 
for coluna in df_final.drop(columns={'km','municipio','data','velocidade_Comercial','velocidade_Moto',
                                     'velocidade_Passeio','velocidade_Ônibus','volume_Comercial',
                                     'volume_Moto','volume_Passeio','volume_Ônibus',
                                     'acidentes','mes','tipo_perfil_de_terreno'}).columns:
    with sns.axes_style("whitegrid"):
        def autopct_format(values):
                def inner_autopct(pct):
                        total = sum(values)
                        val = int(round(pct*total/100.0))  # Valor absoluto
                        return f'{pct:.1f}%\n({val})'
                return inner_autopct
        plt.figure(figsize=(15,10))
        plt.pie(df_agrup_dict[f'df_agrup_{coluna}']['acidentes_sum'],autopct=autopct_format(df_agrup_dict[f'df_agrup_{coluna}']['acidentes_sum']),
                startangle=90,colors=['#8ABBDB','#527083'])
        plt.legend(df_agrup_dict[f'df_agrup_{coluna}'][coluna], loc="best", title="Categorias",
                   prop={'size': 20})
        #plt.xlabel(coluna, fontsize=20)
        #plt.ylabel('Quantidade de acidentes', fontsize=20)
        #plt.title(f'Acidentes por {coluna}', fontsize=20, fontweight='bold')
        plt.xticks()
        plt.yticks()
        plt.savefig(f'ad_rosca_{coluna}.png', dpi=600)
        plt.show()  

#%%
## Timeline
with sns.axes_style("whitegrid"):
        plt.figure(figsize=(15,10))
        ax = sns.lineplot(data=df_agrup_dict['df_agrup_mes'], x='mes', y='acidentes_sum',
                    palette='Blues_d')
        plt.fill_between(df_agrup_dict['df_agrup_mes']['mes'], df_agrup_dict['df_agrup_mes']['acidentes_sum'], 
                     color=sns.color_palette('Blues_d',n_colors=1), alpha=0.3)
        plt.xlabel('Mes')
        plt.ylabel('Quantidade de acidentes')
        #plt.title('Acidentes por mes', fontsize=20, fontweight='bold')
        plt.ylim(0)
        plt.xticks(ticks=df_agrup_dict['df_agrup_mes']['mes'])
        plt.yticks()

        plt.savefig('ad_linha_mes.png', dpi=600)
        plt.show() 

#%%
## Graficos de BoxPlot Volume x Acidente
# Definindo as colunas que serão plotadas
colunas = df_final.drop(columns={'km','municipio','data','sentido_crescente','uf','acidentes','chuva',
                                 'dia_da_semana','mes','tipo_faixa','tipo_perfil_de_terreno','iluminacao',
                                 'velocidade_Comercial','velocidade_Moto','velocidade_Passeio','velocidade_Ônibus'}).columns

# Definindo o número de linhas e colunas para os subplots
num_cols = 2  # Número de colunas por linha de gráficos
num_rows = int(np.ceil(len(colunas) / num_cols))  # Número de linhas necessárias

# Criando a figura com subplots
fig, axes = plt.subplots(num_rows, num_cols, figsize=(18, num_rows * 6))
axes = axes.flatten()  # Flatten para iterar facilmente nos eixos

# Iterando sobre cada coluna e seu respectivo eixo
for i, coluna in enumerate(colunas):
    sns.boxplot(x='acidentes', y=coluna, data=df_final, hue='acidentes', palette='Blues_d', ax=axes[i])
    axes[i].set_xlabel('')
    axes[i].set_ylabel(coluna)
    axes[i].tick_params(axis='both')
    axes[i].set_title(f'DISTRIBUIÇÃO DO {coluna.upper()} POR ACIDENTE', loc='left')

    if i == 0:
         handles, labels = axes[i].get_legend_handles_labels()
    axes[i].legend_.remove()
# Removendo quaisquer eixos não utilizados (caso o número de colunas seja menor que o número de subplots)
for i in range(len(colunas), len(axes)):
    fig.delaxes(axes[i])

#plt.suptitle("Distribuição dos volumes por Acidente (0 = Não, 1 = Sim)", fontsize=22)
fig.legend(handles, labels, loc='upper right', title='ACIDENTE')
plt.tight_layout()
plt.savefig('ad_box_subplots_volumexacidente.png', dpi=600)
plt.show()

#%%
## Graficos de BoxPlot Velocidade x Acidente
# Definindo as colunas que serão plotadas
colunas = df_final.drop(columns={'km','municipio','data','sentido_crescente','uf','acidentes','chuva',
                                 'dia_da_semana','mes','tipo_faixa','tipo_perfil_de_terreno','iluminacao',
                                 'volume_Comercial','volume_Moto','volume_Passeio','volume_Ônibus'}).columns

# Definindo o número de linhas e colunas para os subplots
num_cols = 2  # Número de colunas por linha de gráficos
num_rows = int(np.ceil(len(colunas) / num_cols))  # Número de linhas necessárias

# Criando a figura com subplots
fig, axes = plt.subplots(num_rows, num_cols, figsize=(18, num_rows * 6))
axes = axes.flatten()  # Flatten para iterar facilmente nos eixos

# Iterando sobre cada coluna e seu respectivo eixo
for i, coluna in enumerate(colunas):
    sns.boxplot(x='acidentes', y=coluna, data=df_final, hue='acidentes', palette='Blues_d', ax=axes[i])
    axes[i].set_xlabel('')
    axes[i].set_ylabel(coluna)
    axes[i].tick_params(axis='both')
    axes[i].set_title(f'DISTRIBUIÇÃO DO {coluna.upper()} POR ACIDENTE', loc='left')

    if i == 0:
         handles, labels = axes[i].get_legend_handles_labels()
    axes[i].legend_.remove()
# Removendo quaisquer eixos não utilizados (caso o número de colunas seja menor que o número de subplots)
for i in range(len(colunas), len(axes)):
    fig.delaxes(axes[i])

#plt.suptitle("Distribuição dos volumes por Acidente (0 = Não, 1 = Sim)", fontsize=22)
fig.legend(handles, labels, loc='upper right', title='ACIDENTE')
plt.tight_layout()
plt.savefig('ad_box_subplots_velocidadexacidente.png', dpi=600)
plt.show()

#%%
## Graficos de BoxPlot Volume x Chuva
# Definindo as colunas que serão plotadas
colunas = df_final.drop(columns={'km','municipio','data','sentido_crescente','uf','acidentes','acidentes',
                                 'dia_da_semana','mes','tipo_faixa','tipo_perfil_de_terreno','iluminacao',
                                 'velocidade_Comercial','velocidade_Moto','velocidade_Passeio','velocidade_Ônibus', 'chuva'}).columns

# Definindo o número de linhas e colunas para os subplots
num_cols = 2  # Número de colunas por linha de gráficos
num_rows = int(np.ceil(len(colunas) / num_cols))  # Número de linhas necessárias

# Criando a figura com subplots
fig, axes = plt.subplots(num_rows, num_cols, figsize=(18, num_rows * 6))
axes = axes.flatten()  # Flatten para iterar facilmente nos eixos

# Iterando sobre cada coluna e seu respectivo eixo
for i, coluna in enumerate(colunas):
    sns.boxplot(x='chuva', y=coluna, data=df_final, hue='chuva', palette='Blues_d', ax=axes[i])
    axes[i].set_xlabel('')
    axes[i].set_ylabel(coluna)
    axes[i].tick_params(axis='both')
    axes[i].set_title(f'DISTRIBUIÇÃO DO {coluna.upper()} POR CHUVA', loc='left')

    if i == 0:
         handles, labels = axes[i].get_legend_handles_labels()
    axes[i].legend_.remove()
# Removendo quaisquer eixos não utilizados (caso o número de colunas seja menor que o número de subplots)
for i in range(len(colunas), len(axes)):
    fig.delaxes(axes[i])

#plt.suptitle("Distribuição dos volumes por Chuva (0 = Não, 1 = Sim)", fontsize=22)
fig.legend(handles, labels, loc='upper right', title='CHUVA')
plt.tight_layout()
plt.savefig('ad_box_subplots_volumexchuva.png', dpi=600)
plt.show()

#%%
## Detalha as features
df_final.info()

#%%
## Estatisticas univariadas quantitativas
df_final[['velocidade_Comercial','velocidade_Moto','velocidade_Passeio','velocidade_Ônibus']].describe()

#%%       
df_final[[ 'volume_Comercial','volume_Moto','volume_Passeio','volume_Ônibus']].describe()

#%%
## Estatisticas univariadas qualitativas
df_final[['sentido_crescente','chuva','dia_da_semana','tipo_faixa']].astype(str).describe()

#%%
df_final[['tipo_perfil_de_terreno','iluminacao']].astype(str).describe()

# %%
df_final['dia_da_semana'].value_counts()





# # %%
# ## Volume de veiculos por acidentes
# image_paths = ['ad_box_volume_Comercial.png', 'ad_box_volume_Moto.png', 'ad_box_volume_Ônibus.png', 'ad_box_volume_Passeio.png']
# images = [mpimg.imread(img) for img in image_paths]

# # Criar uma figura com 2 linhas e 3 colunas
# fig, axes = plt.subplots(2, 2, figsize=(20, 15))

# # Plotar cada imagem em um subplot
# for i, ax in enumerate(axes.flat):
#     ax.imshow(images[i])
#     ax.axis('off')  # Desabilitar os eixos para que as imagens fiquem limpas

# # Ajustar o layout
# plt.savefig(f'ad_grade_acidentes_volume.png', dpi=600)
# plt.tight_layout()
# plt.show()

