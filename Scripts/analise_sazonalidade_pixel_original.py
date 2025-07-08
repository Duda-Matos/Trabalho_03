# -*- coding: utf-8 -*-
"""
Script final para o Trabalho de ENS5132.
Permite a fácil alteração do nível de pressão para análise comparativa.
"""
# %% 1. [ETAPA A] Imports e Configuração de Caminhos
import os
import xarray as xr
import geopandas as gpd
import numpy as np
import rioxarray
import matplotlib.pyplot as plt
#%%
print(">>> INICIANDO SCRIPT DE ANÁLISE DE SAZONALIDADE <<<")

# ==============================================================================
NIVEL_PRESSAO = 700

# ==============================================================================

# Estrutura de pastas
PASTA_RAIZ = os.path.join("..")
PASTA_DADOS = os.path.join(PASTA_RAIZ, "inputs", "merra")
CAMINHO_SHAPEFILE = r"C:\Users\dudad\Documents\GitHub\ENS5132\Trabalho_03\inputs\VARIOS.BR_UF_2024"
PASTA_SAIDA = os.path.join(PASTA_RAIZ, "outputs")
os.makedirs(PASTA_SAIDA, exist_ok=True)
#shape = gpd.read_filer( r"C:\Users\dudad\Documents\GitHub\ENS5132\Trabalho_03\inputs\VARIOS.BR_UF_2024")
# %% 2. 

print(f"\nCarregando e preparando dados para o nível de {NIVEL_PRESSAO} hPa...")
padrao_arquivos = os.path.join(PASTA_DADOS, "MERRA2_400.tavgU_3d_tdt_Np.*.nc4")
ds = xr.open_mfdataset(padrao_arquivos, combine='by_coords', parallel=True).sortby('time')
shape = gpd.read_file(CAMINHO_SHAPEFILE).to_crs("EPSG:4326")
ds.rio.write_crs("EPSG:4326", inplace=True)
ds.rio.set_spatial_dims(x_dim="lon", y_dim="lat", inplace=True)
ds_recortado = ds.rio.clip(shape.geometry, drop=True, all_touched=True)
ds_mensal = ds_recortado.resample(time='ME').mean()
print("Etapa de carregamento e preparação concluída.")

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
    coords={'lat': variavel_analise['lat'], 'lon': variavel_analise['lon']})

print("Cálculo da sazonalidade concluído.")

# %% 4. [ETAPA C] Visualização e Interpretação


print("\n>>> Gerando o mapa final...")
fig, ax = plt.subplots(figsize=(10, 8))
sazonalidade_map.plot(cmap='viridis', ax=ax, cbar_kwargs={'label': 'Índice de Sazonalidade (Adimensional)'})
shape.plot(ax=ax, facecolor='none', edgecolor='black', linewidth=0.5)
ax.set_title(f"Mapa de Sazonalidade da Tendência de Temperatura ({NIVEL_PRESSAO} hPa) - Brasil 2024", fontsize=14, pad=20)
ax.set_xlabel('Longitude', fontsize=12)
ax.set_ylabel('Latitude', fontsize=12)
ax.grid(True, linestyle='--', alpha=0.3)
plt.tight_layout()

# O nome do arquivo salvo agora muda automaticamente com o nível de pressão
caminho_figura = os.path.join(PASTA_SAIDA, f"mapa_sazonalidade_{NIVEL_PRESSAO}hpa.png")
plt.savefig(caminho_figura, dpi=300)
print(f"Mapa final salvo em: {caminho_figura}")

plt.show()

print(f"\n>>> ANÁLISE PARA {NIVEL_PRESSAO} hPa CONCLUÍDA! <<<")