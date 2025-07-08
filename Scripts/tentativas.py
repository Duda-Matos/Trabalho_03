# -*- coding: utf-8 -*-
"""
Created on Mon Jul  7 09:11:23 2025

@author: dudad
"""

# %% 1. [ETAPA A] Imports e Configuração de Caminhos
import os
import xarray as xr
import geopandas as gpd
import numpy as np
import rioxarray
import matplotlib.pyplot as plt

#print(">>> INICIANDO SCRIPT DE ANÁLISE DE SAZONALIDADE <<<")

# ==============================================================================
NIVEL_PRESSAO = 700 
ESTADO_ALVO = "Santa Catarina"
# ==============================================================================
#Estrutura de pastas
PASTA_RAIZ = os.path.join("..") # Isso se resolve para C:/Users/dudad/Documents/GitHub/ENS5132/Trabalho_03/
PASTA_DADOS = os.path.join(PASTA_RAIZ, "inputs", "merra")
CAMINHO_SHAPEFILE = r"C:\Users\dudad\Documents\GitHub\ENS5132\Trabalho_03\inputs\VARIOS.BR_UF_2024"
PASTA_SAIDA = os.path.join(PASTA_RAIZ, "outputs")
os.makedirs(PASTA_SAIDA, exist_ok=True)
shape = gpd.read_file(r"C:\Users\dudad\Documents\GitHub\ENS5132\Trabalho_03\inputs\VARIOS.BR_UF_2024")
PASTA_SAIDA = os.path.join(PASTA_RAIZ, "outputs")
os.makedirs(PASTA_SAIDA, exist_ok=True)
estado_selecionado = shape.iloc[[5]]
# Adicione esta linha para ver o diretório de trabalho atual, útil para depuração de caminhos
print(f"Diretório de trabalho atual: {os.getcwd()}")
#%%
print(shape.columns.tolist())
print(shape.head())
print(f"\nCRS inicial do shapefile: {shape.crs}")
print("--- FIM DAS LINHAS DE DEPURACÃO ---")
print(f"\nSelecionando o estado: {ESTADO_ALVO}...")
NOME_COLUNA_ESTADO = 'NM_UF'
if shape.crs is not None and shape.crs != "EPSG:4326":
    shape = shape.to_crs("EPSG:4326")
    print("DEBUG: SRC do shapefile APÓS to_crs: EPSG:4326")
elif shape.crs is None:
    print("ERRO: Não foi possível definir um SRC inicial para o shapefile. Recorte espacial pode ser impreciso.")
    raise SystemExit("Erro: Falha na definição do SRC do shapefile. Saindo.") # Interrompe

print(f"SRC FINAL do shapefile para uso: {shape.crs}")
  
if NOME_COLUNA_ESTADO not in shape.columns:
    print(f"ERRO CRÍTICO: Coluna '{NOME_COLUNA_ESTADO}' não encontrada no shapefile.")
    print(f"Colunas disponíveis: {shape.columns.tolist()}")
    raise SystemExit("Erro: Nome da coluna de estado incorreto. Saindo.")

estado_selecionado = shape[shape[NOME_COLUNA_ESTADO] == ESTADO_ALVO]

if estado_selecionado.empty:
    print(f"ERRO: Estado '{ESTADO_ALVO}' não encontrado na coluna '{NOME_COLUNA_ESTADO}'.")
    print("Verifique se o nome do estado está escrito exatamente como no arquivo.")
    print("Valores únicos na coluna de estado (amostra):", shape[NOME_COLUNA_ESTADO].unique()[:10]) # Mostra alguns nomes para depuração
    raise SystemExit("Erro: Estado não encontrado. Saindo.")
else:
    print(f"Estado '{ESTADO_ALVO}' selecionado com sucesso.")

# --- FIM DA NOVA SEÇÃO DE SELEÇÃO ---

# %% 2. [ETAPA A] Carregamento e Preparação dos Dados
print(f"\nCarregando e preparando dados para o nível de {NIVEL_PRESSAO} hPa...")
Lista_dados= ["C:/Users/dudad/Documents/GitHub/ENS5132/trabalho_03/inputs/merra/MERRA2_400.tavgU_3d_tdt_Np.202412.nc4",
"C:/Users/dudad/Documents/GitHub/ENS5132/trabalho_03/inputs/merra/MERRA2_400.tavgU_3d_tdt_Np.202401.nc4",
"C:/Users/dudad/Documents/GitHub/ENS5132/trabalho_03/inputs/merra/MERRA2_400.tavgU_3d_tdt_Np.202402.nc4",
"C:/Users/dudad/Documents/GitHub/ENS5132/trabalho_03/inputs/merra/MERRA2_400.tavgU_3d_tdt_Np.202403.nc4",
"C:/Users/dudad/Documents/GitHub/ENS5132/trabalho_03/inputs/merra/MERRA2_400.tavgU_3d_tdt_Np.202404.nc4",
"C:/Users/dudad/Documents/GitHub/ENS5132/trabalho_03/inputs/merra/MERRA2_400.tavgU_3d_tdt_Np.202405.nc4",
"C:/Users/dudad/Documents/GitHub/ENS5132/trabalho_03/inputs/merra/MERRA2_400.tavgU_3d_tdt_Np.202406.nc4",
"C:/Users/dudad/Documents/GitHub/ENS5132/trabalho_03/inputs/merra/MERRA2_400.tavgU_3d_tdt_Np.202407.nc4",
"C:/Users/dudad/Documents/GitHub/ENS5132/trabalho_03/inputs/merra/MERRA2_400.tavgU_3d_tdt_Np.202408.nc4",
"C:/Users/dudad/Documents/GitHub/ENS5132/trabalho_03/inputs/merra/MERRA2_400.tavgU_3d_tdt_Np.202409.nc4",
"C:/Users/dudad/Documents/GitHub/ENS5132/trabalho_03/inputs/merra/MERRA2_400.tavgU_3d_tdt_Np.202410.nc4",
"C:/Users/dudad/Documents/GitHub/ENS5132/trabalho_03/inputs/merra/MERRA2_400.tavgU_3d_tdt_Np.202411.nc4"]

print("Verificando a existência dos arquivos na lista...")
arquivos_existentes = []
for f in Lista_dados:
    if os.path.exists(f):
        arquivos_existentes.append(f)
        print(f"  - Encontrado: {f}")
    else:
        print(f"  - **AVISO: Não encontrado:** {f}")

if not arquivos_existentes:
    print("ERRO: Nenhum dos arquivos na lista foi encontrado. Verifique os caminhos.")
    ds = None # Define ds como None para a verificação posterior
    raise SystemExit("Erro: Nenhum dado de MERRA2 encontrado para análise. Saindo.") # Usar raise SystemExit para parar o script
else:
    try:
        print(f"Tentando abrir {len(arquivos_existentes)} arquivos com xarray.open_mfdataset...")
        ds = xr.open_mfdataset(arquivos_existentes, combine='by_coords', parallel=True)
        print("Datasets combinados com sucesso!")
    except Exception as e:
        print(f"ERRO ao carregar e combinar os datasets: {e}")
        print("Verifique a integridade dos arquivos e a compatibilidade para combinação.")
        ds = None # Garante que ds é None em caso de falha
        raise SystemExit(f"Erro ao carregar dados MERRA2: {e}. Saindo.") # Usar raise SystemExit para parar o script

# --- Seção do Shapefile ---
# Adicione esta linha para tentar restaurar ou criar o arquivo .shx (manter por precaução)
os.environ['SHAPE_RESTORE_SHX'] = 'YES'

# Imprime o caminho completo que o script tentará abrir o shapefile
caminho_absoluto_shapefile = os.path.abspath(CAMINHO_SHAPEFILE)
print(f"\nDEBUG: Tentando carregar shapefile do caminho: {caminho_absoluto_shapefile}")
#%%
# Verifica se o arquivo .shp existe no caminho absoluto
if not os.path.exists(caminho_absoluto_shapefile):
    print(f"ERRO CRÍTICO: O arquivo shapefile PRINCIPAL (.shp) NÃO foi encontrado em: {caminho_absoluto_shapefile}")
    print("Por favor, verifique se o arquivo 'BR_UF_2024.shp' realmente existe neste local.")
    print("Certifique-se de que o nome está correto (incluindo letras maiúsculas/minúsculas).")
    raise SystemExit("Erro: Arquivo SHP principal não encontrado. Saindo.") # Interrompe o script

try:
    # 1. Carregar o shapefile SEM tentar transformar o CRS inicialmente
    shape = gpd.read_file(CAMINHO_SHAPEFILE)
    print("Shapefile carregado com sucesso (SRC ainda não processado).")

    # 2. Verificar o SRC do shapefile recém-carregado
    print(f"DEBUG: SRC do shapefile APÓS leitura inicial: {shape.crs}")

    # 3. SE o SRC for 'None' (naive), defina-o explicitamente para SIRGAS 2000 (EPSG:4674).
    if shape.crs is None:
        print("AVISO: O SRC do shapefile é 'None' (geometrias 'naives').")
        print("Definindo o SRC manualmente para SIRGAS 2000 (EPSG:4674) conforme sua confirmação.")
        shape = shape.set_crs("EPSG:4674", allow_override=True) # <-- CORREÇÃO AQUI: USANDO EPSG:4674
        print(f"DEBUG: SRC do shapefile APÓS set_crs para EPSG:4674: {shape.crs}")
    # Caso contrário, se o SRC não for None, mas for diferente de EPSG:4326
    elif shape.crs != "EPSG:4326":
        print(f"Transformando SRC do shapefile de {shape.crs} para EPSG:4326...")
    else: # Já é EPSG:4326
        print(f"SRC do shapefile já é {shape.crs}. Nenhuma transformação é necessária.")

    # 4. Agora, transformar para EPSG:4326, se necessário
    if shape.crs is not None and shape.crs != "EPSG:4326":
        shape = shape.to_crs("EPSG:4326")
        print("DEBUG: SRC do shapefile APÓS to_crs: EPSG:4326")
    elif shape.crs is None:
        # Isso só deve acontecer se set_crs falhou (o que é raro se o valor for válido)
        print("ERRO: Não foi possível definir um SRC inicial para o shapefile. Recorte espacial pode ser impreciso.")
        raise SystemExit("Erro: Falha na definição do SRC do shapefile. Saindo.") # Interrompe

    print(f"SRC FINAL do shapefile para uso: {shape.crs}")

except Exception as e:
    print(f"ERRO FATAL ao carregar ou processar o shapefile: {e}")
    print("Possíveis causas: arquivo .shp não encontrado, .prj ausente/corrompido, ou SRC inicial incorreto.")
    raise SystemExit(f"Erro: Falha no processamento do shapefile: {e}. Saindo.") # Interrompe

# Opcional: Remover a variável de ambiente se não for mais necessária
del os.environ['SHAPE_RESTORE_SHX']
#%%

# Certifique-se de que 'ds' foi carregado com sucesso antes de continuar as operações rioxarray
if ds is None:
    print("Erro: O dataset MERRA2 não foi carregado corretamente. Não é possível prosseguir.")
    raise SystemExit("Erro: Dataset MERRA2 ausente. Saindo.")

# Continuar com as operações rioxarray
# É importante que 'ds' também tenha um CRS definido para o clipping funcionar corretamente
print("\nDefinindo CRS e dimensões espaciais para o dataset MERRA2...")
ds.rio.write_crs("EPSG:4326", inplace=True) # Confirma o CRS do dataset MERRA2
ds.rio.set_spatial_dims(x_dim="lon", y_dim="lat", inplace=True)
print("CRS e dimensões espaciais do MERRA2 configurados.")


#--- ATENÇÃO: Mude a variável de clipping para 'estado_selecionado' ---
print(f"Realizando recorte do dataset MERRA2 pelo estado '{ESTADO_ALVO}'...")
ds_recortado = ds.rio.clip(estado_selecionado.geometry, drop=True, all_touched=True)
print(f"Recorte do dataset MERRA2 para '{ESTADO_ALVO}' concluído.")
ds_mensal = ds_recortado.resample(time='ME').mean()
print("Etapa de carregamento e preparação concluída.")

# --- INÍCIO DO BLOCO DE INTERPOLAÇÃO (Movido para o local correto no fluxo de dados) ---
# O objetivo é criar uma grade espacial mais fina para uma visualização mais suave.
# Define a nova resolução desejada em graus. Valores menores resultam em mapas mais suaves.
nova_resolucao_lat = 0.5 # Ex: 0.02 graus (muito mais fino que os 0.5 graus originais)
nova_resolucao_lon = 0.5 # Ex: 0.02 graus (muito mais fino que os 0.625 graus originais)

# Define os novos pontos de latitude e longitude para a grade mais fina.
# Eles devem cobrir a extensão da área recortada (ds_recortado).
min_lat_interp, max_lat_interp = ds_recortado['lat'].min().item(), ds_recortado['lat'].max().item()
min_lon_interp, max_lon_interp = ds_recortado['lon'].min().item(), ds_recortado['lon'].max().item()

# np.arange cria uma sequência de valores. np.floor/ceil arredondam para garantir cobertura total.
new_lat = np.arange(np.floor(min_lat_interp * 100) / 100, np.ceil(max_lat_interp * 100) / 100 + nova_resolucao_lat/2, nova_resolucao_lat)
new_lon = np.arange(np.floor(min_lon_interp * 100) / 100, np.ceil(max_lon_interp * 100) / 100 + nova_resolucao_lon/2, nova_resolucao_lon)

print(f"Interpolando dataset para resolução {nova_resolucao_lat}°x{nova_resolucao_lon}° usando interpolação linear...")
# Realiza a interpolação. 'linear' para 2D é o equivalente a interpolação bilinear.
ds_interpolado = ds_recortado.interp(coords={'lat': new_lat, 'lon': new_lon}, method='linear')
print("Interpolação concluída.")

# --- Verificações de Depuração da Interpolação ---
print("\n--- Verificando dimensões e coordenadas após interpolação ---")
print("Dimensões de ds_recortado (original):", ds_recortado.dims)
print(f"  Latitudes ds_recortado (amostra, {len(ds_recortado['lat'])} pontos): {ds_recortado['lat'].values[::2]}")
print(f"  Longitudes ds_recortado (amostra, {len(ds_recortado['lon'])} pontos): {ds_recortado['lon'].values[::2]}")

print("\nDimensões de ds_interpolado (interpolado):", ds_interpolado.dims)
print(f"  Latitudes ds_interpolado (amostra, {len(ds_interpolado['lat'])} pontos): {ds_interpolado['lat'].values[::20]}")
print(f"  Longitudes ds_interpolado (amostra, {len(ds_interpolado['lon'])} pontos): {ds_interpolado['lon'].values[::20]}")

# *** CORREÇÃO CRUCIAL: ds_mensal AGORA É CRIADO A PARTIR DE ds_interpolado ***
# Reamostra os dados interpolados para médias mensais.
ds_mensal = ds_interpolado.resample(time='ME').mean()
print("DEBUG: ds_mensal criado a partir de ds_interpolado (dados mensais interpolados).")

print("\nDimensões de ds_mensal (final para análise):", ds_mensal.dims)
print(f"  Latitudes ds_mensal (amostra, {len(ds_mensal['lat'])} pontos): {ds_mensal['lat'].values[::20]}")
print(f"  Longitudes ds_mensal (amostra, {len(ds_mensal['lon'])} pontos): {ds_mensal['lon'].values[::20]}")
print("--- FIM DAS VERIFICAÇÕES DE INTERPOLAÇÃO ---")

print("Etapa de carregamento e preparação concluída.")

# --- FIM DO BLOCO DE INTERPOLAÇÃO ---

# %% 3. [ETAPA B] Cálculo da Sazonalidade
print("\n>>> Iniciando cálculo da sazonalidade...")

def calcula_sazonalidade_markham_robusto(tendencia_mensal):
    denominador = np.sum(np.abs(tendencia_mensal))
    if np.isclose(denominador, 0): return np.nan
    media_anual = np.mean(tendencia_mensal)
    desvio_absoluto_total = np.sum(np.abs(tendencia_mensal - media_anual))
    return desvio_absoluto_total / denominador

# AQUI a variável NIVEL_PRESSAO é usada para selecionar a altitude correta
variavel_analise = ds_mensal['DTDTTOT'].sel(lev=NIVEL_PRESSAO).squeeze(drop=True)
variavel_analise.load()

output_array = np.full(variavel_analise.shape[1:], np.nan, dtype=np.float32)
for i in range(len(variavel_analise['lat'])):
    for j in range(len(variavel_analise['lon'])):
        time_series = variavel_analise[:, i, j].values
        if not np.all(np.isnan(time_series)):
            output_array[i, j] = calcula_sazonalidade_markham_robusto(time_series)

sazonalidade_map = xr.DataArray(
    data=output_array,
    dims=['lat', 'lon'],
    coords={'lat': variavel_analise['lat'], 'lon': variavel_analise['lon']}
)
print("Cálculo da sazonalidade concluído.")

# %% 4. [ETAPA C] Visualização e Interpretação Estado Selecionado


print("\n>>> Gerando o mapa final...")
fig, ax = plt.subplots(figsize=(10, 8))
sazonalidade_map.plot(cmap='viridis', ax=ax, cbar_kwargs={'label': 'Índice de Sazonalidade (Adimensional)'})

# --- ATENÇÃO: Plote APENAS a geometria do estado selecionado ---
estado_selecionado.plot(ax=ax, facecolor='none', edgecolor='black', linewidth=0.5)

# Opcional: Se quiser mostrar os outros estados como uma referência mais sutil (descomente se desejar)
# shape.plot(ax=ax, facecolor='none', edgecolor='gray', linewidth=0.3, alpha=0.5)

ax.set_title(f"Sazonalidade da Tendência de Temperatura ({NIVEL_PRESSAO} hPa) - {ESTADO_ALVO} 2024", fontsize=14, pad=20)
ax.set_xlabel('Longitude', fontsize=12)
ax.set_ylabel('Latitude', fontsize=12)
ax.grid(True, linestyle='--', alpha=0.3)
plt.tight_layout()

# O nome do arquivo salvo agora inclui o nome do estado e o nível de pressão
caminho_figura = os.path.join(PASTA_SAIDA, f"mapa_sazonalidade_{ESTADO_ALVO.replace(' ', '_')}_{NIVEL_PRESSAO}hpa.png")
plt.savefig(caminho_figura, dpi=300)
print(f"Mapa final salvo em: {caminho_figura}")

plt.show()

print(f"\n>>> ANÁLISE PARA {ESTADO_ALVO} ({NIVEL_PRESSAO} hPa) CONCLUÍDA! <<<")
