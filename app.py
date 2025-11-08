import streamlit as st
import pandas as pd
import numpy as np
import pyomo.environ as pyo
import gdown 
import warnings
import logging
import os
from dataclasses import dataclass
from typing import Dict, Optional, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from pyomo.opt import SolverFactory, TerminationCondition, SolverStatus
from pyomo.environ import ConcreteModel, Set, Var, NonNegativeReals, Objective, minimize, Constraint, ConstraintList
from matplotlib.ticker import ScalarFormatter, FuncFormatter

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

warnings.filterwarnings('ignore')
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams['figure.dpi'] = 100 
plot_col, space_col = st.columns([3, 1])

# --- FUNCIÓN PARA INICIALIZAR ESTRUCTURAS DE COLUMNAS ---
def get_empty_df_structure(df_key: str) -> pd.DataFrame:
    if df_key == 'plantas':
        return pd.DataFrame(columns=['Planta', 'Producto', 'Capacidad_Produccion', 'Costo_Produccion'])
    elif df_key == 'centros':
        return pd.DataFrame(columns=['Centro', 'Producto', 'Capacidad_Almacenamiento'])
    elif df_key == 'clientes':
        return pd.DataFrame(columns=['Cliente', 'Producto', 'Demanda'])
    elif df_key == 'productos':
        return pd.DataFrame(columns=['Producto'])
    elif df_key == 'costos':
        return pd.DataFrame(columns=['Planta', 'Centro', 'Cliente', 'Producto', 'Costo_Plant_Centro', 'Costo_Centro_Cliente'])
    return pd.DataFrame()
# ---------------------------------------------------------------------

@dataclass
class NetworkData:
    plantas: pd.DataFrame
    centros: pd.DataFrame
    clientes: pd.DataFrame
    productos: pd.DataFrame
    costos: pd.DataFrame

    def get_hash(self):
        hash_val = 0
        for df in [self.plantas, self.centros, self.clientes, self.productos, self.costos]:
            if not df.empty:
                 hash_val += df.shape[0] * df.shape[1]
                 for col in df.columns:
                     if df[col].dtype == object:
                         hash_val += hash(tuple(df[col].astype(str)))
                     else:
                         hash_val += df[col].sum()
        return hash_val

    def update_data(self, key: str, df: pd.DataFrame):
        if hasattr(self, key):
            setattr(self, key, df)
        else:
            logger.warning(f"Error: El atributo '{key}' no existe en NetworkData.")

#Descarga datos desde Google Drive.
class DataDownloader:
    FILE_IDS = {
        'plantas': '1Mq1C4Q5BXX-RUP6RyLlRDXMsLIxJPkMX', 'centros': '1OxMfsW98iIfm8hqiul23Pmo8Iec1roX2',
        'clientes': '1v7UchVRnrKYPgYsir_0aCqk4W1Nj8o0O', 'costos': '1Y4M_U7i_7k0-MVjU8itzGK4kYxzXdNFT',
        'productos': '1B1UGqYzLTE3uh_1-MdvuT22eAGA6BXNx'
    }
    @staticmethod
    def download_all_files(show_ui_spinner=False) -> Dict[str, str]:        
        status_container = st.empty() if show_ui_spinner else None
        
        if status_container:
            status_container.info("Iniciando descarga de archivos de red desde Google Drive...")
        
        downloaded_files = {}
        logger.info("Iniciando intento de descarga de datos por defecto desde Google Drive.")
        
        for name, file_id in DataDownloader.FILE_IDS.items():
            output = f'{name}.csv'
            url = f'https://drive.google.com/uc?id={file_id}'
            try: 
                gdown.download(url, output, quiet=True, fuzzy=True)
                downloaded_files[name] = output
                logger.info(f"Éxito: '{output}' descargado.")
            except Exception as e: 
                logger.warning(f"Falló la descarga de '{output}'. Error: {e}")
        
        if status_container:
            status_container.empty()
            if len(downloaded_files) == len(DataDownloader.FILE_IDS):
                st.success("Todos los archivos descargados exitosamente!")
            else:
                 st.warning("Falló la descarga de algunos archivos. Revise la terminal para detalles.")
            
        return downloaded_files

class DataLoader:
    @staticmethod
    def load_from_files(files: Dict[str, str]) -> Optional[NetworkData]:
        try:
            return NetworkData(
                plantas=pd.read_csv(files['plantas']), centros=pd.read_csv(files['centros']),
                clientes=pd.read_csv(files['clientes']), productos=pd.read_csv(files['productos']),
                costos=pd.read_csv(files['costos'])
            )
        except Exception as e:
            logger.error(f"❌ Error crítico al cargar los DataFrames. Verifique la estructura de los CSV. Error: {e}")
            return None

    @staticmethod
    def load_default() -> Optional[NetworkData]:
        downloaded_files = DataDownloader.download_all_files(show_ui_spinner=True)
        
        if len(downloaded_files) == len(DataDownloader.FILE_IDS):
            logger.info("Todos los archivos descargados/disponibles localmente.")
            return DataLoader.load_from_files(downloaded_files)
        
        # Fallback: Busca archivos existentes en el directorio
        logger.warning("Intentando FALLBACK: Buscando archivos CSV en el directorio local.")
        
        fallback_files = {}
        
        for name_key in DataDownloader.FILE_IDS.keys():
            df_key = name_key.replace('.csv', '')
            output = f'{df_key}.csv'

            if os.path.exists(output):
                fallback_files[df_key] = output
                logger.info(f"Fallback: '{output}' encontrado localmente.")
            else:
                logger.error(f"Fallback Error: '{output}' no encontrado localmente.")

        if len(fallback_files) == len(DataDownloader.FILE_IDS):
            logger.info("Todos los archivos necesarios encontrados en el directorio local.")
            return DataLoader.load_from_files(fallback_files)
        else:
            logger.error("La carga de datos falló. Ni descarga ni fallback fueron exitosos.")
            return None

class DataProcessor:
    def __init__(self, data: NetworkData):
        self.data = data
        
        try:
            self.P = list(data.plantas['Planta'].unique())
            self.C = list(data.centros['Centro'].unique())
            self.J = list(data.clientes['Cliente'].unique())
            self.K = list(data.productos['Producto'].unique())
        except KeyError:
            self.P, self.C, self.J, self.K = [], [], [], []

    def get_sets(self): return {'plantas': self.P, 'centros': self.C, 'clientes': self.J, 'productos': self.K}

    def get_params(self):
        try:
            demanda = self.data.clientes.set_index(['Cliente', 'Producto'])['Demanda'].to_dict()
            cap_produccion = self.data.plantas.set_index(['Planta', 'Producto'])['Capacidad_Produccion'].to_dict()
            cap_almacenamiento = self.data.centros.set_index(['Centro', 'Producto'])['Capacidad_Almacenamiento'].to_dict()
            costo_produccion = self.data.plantas.set_index(['Planta', 'Producto'])['Costo_Produccion'].to_dict()
            
            # --- Logica de Costos ---
            
            # Costo P->C
            costo_planta_centro = {}
            df_pc_filtered = self.data.costos.dropna(subset=['Planta', 'Centro', 'Producto', 'Costo_Plant_Centro'])
            for (p, c, k), group in df_pc_filtered.groupby(['Planta', 'Centro', 'Producto']):
                costo_planta_centro[(p, c, k)] = group['Costo_Plant_Centro'].iloc[0]

            # Costo C->J
            costo_centro_cliente = {}
            df_cj_filtered = self.data.costos.dropna(subset=['Centro', 'Cliente', 'Producto', 'Costo_Centro_Cliente'])
            for (c, j, k), group in df_cj_filtered.groupby(['Centro', 'Cliente', 'Producto']):
                costo_centro_cliente[(c, j, k)] = group['Costo_Centro_Cliente'].iloc[0]
            # -------------------------------------------------------------
            
            return demanda, cap_produccion, cap_almacenamiento, costo_produccion, costo_planta_centro, costo_centro_cliente
        except Exception as e:
            logger.error(f"Error al procesar los parámetros. Verifique las columnas: {e}")
            return {}, {}, {}, {}, {}, {}


class DistributionNetworkLP:
    def __init__(self, processor: DataProcessor, params):
        self.processor = processor
        self.model = None
        self.results = None
        
        self.sets = processor.get_sets()
        self.P, self.C, self.J, self.K = self.sets['plantas'], self.sets['centros'], self.sets['clientes'], self.sets['productos']
        self.demanda, self.cap_prod, self.cap_alm, self.costo_prod, self.costo_pc, self.costo_cj = params

    def build_model(self):
        modelo = ConcreteModel(name="Red_Distribucion_Multinivel")
        
        if not (self.P and self.C and self.J and self.K):
            logger.warning("No se puede construir el modelo: Los conjuntos de datos están vacíos.")
            return False

        # Conjuntos
        modelo.P, modelo.C, modelo.J, modelo.K = Set(initialize=self.P), Set(initialize=self.C), Set(initialize=self.J), Set(initialize=self.K)
        
        # Variables
        modelo.x = Var(modelo.P, modelo.C, modelo.K, domain=NonNegativeReals)
        modelo.y = Var(modelo.C, modelo.J, modelo.K, domain=NonNegativeReals)

        # Función Objetivo (IDÉNTICA AL NOTEBOOK)
        def funcion_objetivo(m):
            costo_plantas = sum((self.costo_prod.get((p, k), 0) + self.costo_pc.get((p, c, k), 0)) * m.x[p, c, k] for p in m.P for c in m.C for k in m.K)
            costo_centros = sum(self.costo_cj.get((c, j, k), 0) * m.y[c, j, k] for c in m.C for j in m.J for k in m.K)
            return costo_plantas + costo_centros
        modelo.objetivo = Objective(rule=funcion_objetivo, sense=minimize)

        # Restricciones
        def restriccion_demanda(m, j, k):
            req = self.demanda.get((j, k), 0)
            if req > 0: return sum(m.y[c, j, k] for c in m.C) == req
            return Constraint.Skip
        modelo.satisfacer_demanda = Constraint(modelo.J, modelo.K, rule=restriccion_demanda)

        def restriccion_balance(m, c, k):
            return sum(m.x[p, c, k] for p in m.P) == sum(m.y[c, j, k] for j in m.J)
        modelo.balance_centros = Constraint(modelo.C, modelo.K, rule=restriccion_balance)

        def restriccion_capacidad_planta(m, p, k):
            cap = self.cap_prod.get((p, k), 0)
            if cap > 0: return sum(m.x[p, c, k] for c in m.C) <= cap
            return Constraint.Skip
        modelo.capacidad_plantas = Constraint(modelo.P, modelo.K, rule=restriccion_capacidad_planta)

        def restriccion_capacidad_centro(m, c, k):
            cap = self.cap_alm.get((c, k), 0)
            if cap > 0: return sum(m.y[c, j, k] for j in m.J) <= cap
            return Constraint.Skip
        modelo.capacidad_centros = Constraint(modelo.C, modelo.K, rule=restriccion_capacidad_centro)
        
        self.model = modelo
        return True
        
    def solve(self, solver_name='glpk'):
        solver = SolverFactory(solver_name)
        try:
            self.results = solver.solve(self.model, tee=False)
            return self.results.solver.termination_condition == TerminationCondition.optimal
        except Exception: return False

    def get_solution_dataframes(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        flujos_pc = []; flujos_cj = []
        if self.results and self.results.solver.termination_condition == TerminationCondition.optimal:
            for p in self.P:
                for c in self.C:
                    for k in self.K:
                        valor = self.model.x[p, c, k].value
                        if valor is not None and valor > 0.01: flujos_pc.append({'Planta': p, 'Centro': c, 'Producto': k, 'Cantidad': round(valor, 2)})
            for c in self.C:
                for j in self.J:
                    for k in self.K:
                        valor = self.model.y[c, j, k].value
                        if valor is not None and valor > 0.01: flujos_cj.append({'Centro': c, 'Cliente': j, 'Producto': k, 'Cantidad': round(valor, 2)})

        return pd.DataFrame(flujos_pc), pd.DataFrame(flujos_cj)

    def get_cost_breakdown(self) -> Dict:
        if not self.results or self.results.solver.termination_condition != TerminationCondition.optimal: return {}
        
        costo_prod = sum(self.costo_prod.get((p, k), 0) * (self.model.x[p, c, k].value or 0) for p in self.P for c in self.C for k in self.K)
        costo_pc = sum(self.costo_pc.get((p, c, k), 0) * (self.model.x[p, c, k].value or 0) for p in self.P for c in self.C for k in self.K)
        costo_cj = sum(self.costo_cj.get((c, j, k), 0) * (self.model.y[c, j, k].value or 0) for c in self.C for j in self.J for k in self.K)
        costo_total = self.model.objetivo()

        return {'Costo Total': costo_total, 'Producción': costo_prod, 'Transporte Planta → Centro': costo_pc, 'Transporte Centro → Cliente': costo_cj}

def plot_flow_distribution_st(lp_model: 'DistributionNetworkLP'):
    modelo = lp_model.model
    P, C, J, K = lp_model.P, lp_model.C, lp_model.J, lp_model.K

    fig, axes = plt.subplots(2, 1, figsize=(14, 12))

    # --- Flujo Planta → Centro ---
    data_pc = []
    for p in P:
        for c in C:
            # Acceso seguro al valor de la variable
            flujo = sum(modelo.x[p, c, k].value or 0 for k in K) 
            if flujo > 0.01:
                data_pc.append({'Planta': p, 'Centro': c, 'Flujo': flujo})

    if data_pc:
        df_pc = pd.DataFrame(data_pc)
        df_pivot = df_pc.pivot(index='Planta', columns='Centro', values='Flujo').fillna(0)
        df_pivot.plot(kind='bar', stacked=True, ax=axes[0], cmap='tab10',
                     edgecolor='black', linewidth=0.5)
        axes[0].set_title('Flujo: Planta → Centro de Distribución',
                         fontsize=14, fontweight='bold', pad=15)
        axes[0].set_ylabel('Unidades Enviadas', fontsize=12)
        axes[0].set_xlabel('Planta', fontsize=12)
        axes[0].legend(title='Centro', bbox_to_anchor=(1.02, 1), loc='upper left')
        axes[0].tick_params(axis='x', rotation=45 if len(P) > 3 else 0)
    else:
        axes[0].text(0.5, 0.5, 'Sin flujo Planta → Centro', ha='center', va='center', fontsize=14)

    # --- Flujo Centro → Cliente ---
    data_cj = []
    for c in C:
        for j in J:
            flujo = sum(modelo.y[c, j, k].value or 0 for k in K)
            if flujo > 0.01:
                data_cj.append({'Centro': c, 'Cliente': j, 'Flujo': flujo})

    if data_cj:
        df_cj = pd.DataFrame(data_cj)
        num_clientes = len(J)

        if num_clientes > 10:
            totales_cliente = df_cj.groupby('Cliente')['Flujo'].sum()
            top_clientes = totales_cliente.nlargest(9).index
            df_cj['Cliente_Grupo'] = df_cj['Cliente'].apply(
                lambda x: x if x in top_clientes else 'Otros'
            )
            df_pivot = df_cj.groupby(['Centro', 'Cliente_Grupo'])['Flujo'].sum().unstack(fill_value=0)
        else:
            df_pivot = df_cj.pivot(index='Centro', columns='Cliente', values='Flujo').fillna(0)

        df_pivot.plot(kind='bar', stacked=True, ax=axes[1], cmap='tab20',
                     edgecolor='black', linewidth=0.5)
        axes[1].set_title('Flujo: Centro de Distribución → Cliente',
                         fontsize=14, fontweight='bold', pad=15)
        axes[1].set_ylabel('Unidades Enviadas', fontsize=12)
        axes[1].set_xlabel('Centro', fontsize=12)
        axes[1].legend(title='Cliente', bbox_to_anchor=(1.02, 1), loc='upper left')
        axes[1].tick_params(axis='x', rotation=45 if len(C) > 5 else 0)
    else:
        axes[1].text(0.5, 0.5, 'Sin flujo Centro → Cliente', ha='center', va='center', fontsize=14)

    plt.tight_layout(rect=[0, 0, 0.88, 1])
    # Mostrar la figura en Streamlit
    plot_col.pyplot(fig, use_container_width=True) # Mostrar el gráfico en Streamlit

def plot_utilization(lp_model: 'DistributionNetworkLP'):
    modelo = lp_model.model
    P, C, J, K = lp_model.P, lp_model.C, lp_model.J, lp_model.K
    cap_produccion = lp_model.cap_prod
    cap_almacenamiento = lp_model.cap_alm
    demanda = lp_model.demanda

    fig, axes = plt.subplots(3, 1, figsize=(18, 6))

    # --- Utilización de Plantas ---
    util_plantas = []
    for p in P:
        produccion = sum(modelo.x[p, c, k].value or 0 for c in C for k in K)
        capacidad = sum(cap_produccion.get((p, k), 0) for k in K)
        util = (produccion / capacidad * 100) if capacidad > 0 else 0
        util_plantas.append(util)

    bars = axes[0].bar(P, util_plantas, color='steelblue', alpha=0.8, edgecolor='black')
    axes[0].set_title('Utilización: Plantas', fontsize=13, fontweight='bold')
    axes[0].set_ylabel('Utilización (%)', fontsize=11)
    axes[0].set_ylim([0, 110])
    axes[0].axhline(y=100, color='red', linestyle='--', alpha=0.7, label='Máximo')
    axes[0].legend()
    axes[0].tick_params(axis='x', rotation=45 if len(P) > 3 else 0)

    # Agregar porcentajes
    for bar, util in zip(bars, util_plantas):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                    f'{util:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')

    # --- Utilización de Centros ---
    util_centros = []
    for c in C:
        # Nota: La utilización de centros en el notebook original usaba el flujo de entrada (x)
        flujo = sum(modelo.x[p, c, k].value or 0 for p in P for k in K)
        capacidad = sum(cap_almacenamiento.get((c, k), 0) for k in K)
        util = (flujo / capacidad * 100) if capacidad > 0 else 0
        util_centros.append(util)

    bars = axes[1].bar(C, util_centros, color='coral', alpha=0.8, edgecolor='black')
    axes[1].set_title('Utilización: Centros', fontsize=13, fontweight='bold')
    axes[1].set_ylabel('Utilización (%)', fontsize=11)
    axes[1].set_ylim([0, 110])
    axes[1].axhline(y=100, color='red', linestyle='--', alpha=0.7)
    axes[1].tick_params(axis='x', rotation=45)

    for bar, util in zip(bars, util_centros):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                    f'{util:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')

    # --- Satisfacción de Clientes ---
    satisfaccion = []
    clientes_list = []
    for j in J:
        recibido = sum(modelo.y[c, j, k].value or 0 for c in C for k in K)
        demandado = sum(demanda.get((j, k), 0) for k in K)
        sat = (recibido / demandado * 100) if demandado > 0 else 0
        satisfaccion.append(sat)
        clientes_list.append(j)

    # Calcular estadísticas
    satisfaccion_arr = np.array(satisfaccion)
    rango_satisfaccion = satisfaccion_arr.max() - satisfaccion_arr.min()

    if len(J) > 20 and rango_satisfaccion > 5:
        # Histograma para muchos clientes con variación
        axes[2].hist(satisfaccion, bins=15, color='seagreen', alpha=0.8, edgecolor='black')
        axes[2].set_xlabel('Satisfacción (%)', fontsize=11)
        axes[2].set_ylabel('Número de Clientes', fontsize=11)
        axes[2].axvline(x=100, color='red', linestyle='--', alpha=0.7, label='100%')
        axes[2].set_title('Distribución de Satisfacción', fontsize=13, fontweight='bold')
    else:
        # Gráfico de barras individual
        if len(J) <= 15:
            # Mostrar todos los clientes
            bars = axes[2].bar(clientes_list, satisfaccion, color='seagreen',
                             alpha=0.8, edgecolor='black')
            axes[2].tick_params(axis='x', rotation=45)

            # Agregar porcentajes solo si no hay muchos clientes
            for bar, sat in zip(bars, satisfaccion):
                axes[2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                           f'{sat:.1f}%', ha='center', va='bottom', fontsize=8, fontweight='bold')
        else:
            # Top 10 clientes con menor satisfacción + promedio (para > 15 clientes)
            sat_dict = dict(zip(clientes_list, satisfaccion))
            top_10_menores = sorted(sat_dict.items(), key=lambda x: x[1])[:10]
            clientes_mostrar = [x[0] for x in top_10_menores]
            sat_mostrar = [x[1] for x in top_10_menores]

            bars = axes[2].bar(clientes_mostrar, sat_mostrar, color='seagreen',
                             alpha=0.8, edgecolor='black')
            axes[2].tick_params(axis='x', rotation=45)
            axes[2].set_title('Top 10 Clientes con Menor Satisfacción', fontsize=13, fontweight='bold')

        axes[2].set_ylabel('Satisfacción (%)', fontsize=11)
        axes[2].set_ylim([0, 110])
        axes[2].axhline(y=100, color='red', linestyle='--', alpha=0.7, label='100%')
        axes[2].set_title('Satisfacción: Clientes', fontsize=13, fontweight='bold')

    axes[2].legend()
    plt.tight_layout()
    # Aplicar formateador de enteros a todos los ejes Y de los gráficos de barras
    for i in range(3):
        formatter = ScalarFormatter(useOffset=False)
        formatter.set_scientific(False)
        axes[i].yaxis.set_major_formatter(formatter)

    plot_col.pyplot(fig, use_container_width=True) # Mostrar el gráfico en Streamlit

def plot_cost_breakdown(lp_model: 'DistributionNetworkLP', cost_data: Dict):

    # Extraer variables y parámetros del objeto lp_model y cost_data
    modelo = lp_model.model
    P, C, J, K = lp_model.P, lp_model.C, lp_model.J, lp_model.K
    
    costo_produccion = lp_model.costo_prod
    costo_planta_centro = lp_model.costo_pc
    costo_centro_cliente = lp_model.costo_cj
        
    costo_prod = sum(
        costo_produccion.get((p, k), 0) * (modelo.x[p, c, k].value or 0)
        for p in P for c in C for k in K
    )

    costo_pc = sum(
        costo_planta_centro.get((p, c, k), 0) * (modelo.x[p, c, k].value or 0)
        for p in P for c in C for k in K
    )

    costo_cj = sum(
        costo_centro_cliente.get((c, j, k), 0) * (modelo.y[c, j, k].value or 0)
        for c in C for j in J for k in K
    )
    
    # --------------------------------------------------

    # --- 2. Creación y Formato del Gráfico ---
    
    fig, ax = plt.subplots(figsize=(10, 6))

    categorias = ['Producción', 'Transporte\nPlanta→Centro', 'Transporte\nCentro→Cliente']
    costos = [costo_prod, costo_pc, costo_cj]
    colores = ['#3498db', '#e74c3c', '#2ecc71']

    bars = ax.bar(categorias, costos, color=colores, alpha=0.85,
                  edgecolor='black', linewidth=1.5)

    # Formato
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', linestyle='--', alpha=0.5)

    total = sum(costos)
    for bar, costo in zip(bars, costos):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'${costo:,.0f}\n({(costo/total*100):.1f}%)',
               ha='center', va='bottom', fontweight='bold', fontsize=11)

    ax.set_title('Desglose de Costos Totales', fontsize=15, fontweight='bold', pad=20)
    ax.set_ylabel('Costo ($)', fontsize=12)
    
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, p: f'${x:,.0f}'))

    fig.suptitle(f'Costo Total Óptimo: ${total:,.2f}',
                 fontsize=13, fontweight='bold', y=0.98,
                 bbox=dict(boxstyle='round', facecolor='gold', alpha=0.3))

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    plot_col.pyplot(fig, use_container_width=True) # Mostrar el gráfico en Streamlit

def display_final_summary(cost_data: Dict):
    total = cost_data['Costo Total']
    prod = cost_data['Producción']
    pc = cost_data['Transporte Planta → Centro']
    cj = cost_data['Transporte Centro → Cliente']

    st.markdown("## RESULTADOS DE LA OPTIMIZACIÓN")
    st.markdown("---")
    st.markdown("✓ **Solución ÓPTIMA encontrada**")

    st.markdown(f"**💰 COSTO TOTAL MÍNIMO:** **${total:,.2f}**")

    # Tabla de desglose
    df_summary = pd.DataFrame({
        'Concepto': ['Costo de Producción', 'Costo Transporte (P→C)', 'Costo Transporte (C→J)', 'TOTAL:'],
        'Costo': [f"${prod:,.2f}", f"${pc:,.2f}", f"${cj:,.2f}", f"**${total:,.2f}**"]
    })

    st.markdown("### 📊 DESGLOSE DE COSTOS:")
    st.dataframe(df_summary, hide_index=True, width=1000) 

    # Mostrar porcentajes
    if total > 0.01:
        st.markdown(f"""
        **Distribución Porcentual:**
        * Producción: **{(prod/total*100):.1f}%**
        * Transporte P→C: **{(pc/total*100):.1f}%**
        * Transporte C→J: **{(cj/total*100):.1f}%**
        """)
    st.markdown("---")

# --- FUNCIÓN PARA INICIALIZAR ESTRUCTURAS DE COLUMNAS (NECESARIO PARA EL EDITOR) ---
def get_empty_df_structure(df_key: str) -> pd.DataFrame:
    if df_key == 'plantas':
        return pd.DataFrame(columns=['Planta', 'Producto', 'Capacidad_Produccion', 'Costo_Produccion'])
    elif df_key == 'centros':
        return pd.DataFrame(columns=['Centro', 'Producto', 'Capacidad_Almacenamiento'])
    elif df_key == 'clientes':
        return pd.DataFrame(columns=['Cliente', 'Producto', 'Demanda'])
    elif df_key == 'productos':
        return pd.DataFrame(columns=['Producto'])
    elif df_key == 'costos':
        return pd.DataFrame(columns=['Planta', 'Centro', 'Cliente', 'Producto', 'Costo_Plant_Centro', 'Costo_Centro_Cliente'])
    return pd.DataFrame()

@st.cache_resource
def load_default_data_real() -> Optional[NetworkData]:
    return DataLoader.load_default()

# Reset para informacion nueva
def reset_solution_state():
    if 'lp_model' in st.session_state:
        del st.session_state.lp_model
    if 'solution_status' in st.session_state:
        del st.session_state.solution_status
    if 'data_hash_at_solve' in st.session_state:
        del st.session_state.data_hash_at_solve

# GUI
def main_app():
    st.set_page_config(layout="wide")
    st.title("Red Multinivel de Distribución Multiproducto")
    st.markdown("---")

    # --- INICIALIZACIÓN DE VARIABLES DE ESTADO ---
    data_keys = ['plantas', 'centros', 'clientes', 'costos', 'productos']

    # Carga Inicial Diferida (Inicia con DataFrames vacíos)
    if 'network_data' not in st.session_state:
        empty_dfs = {key: get_empty_df_structure(key) for key in data_keys}
        st.session_state.network_data = NetworkData(**empty_dfs)
        st.info("Intentar Documentos disponibles en Google Drive para empezar")
            
    if 'active_tab' not in st.session_state:
        st.session_state.active_tab = "Datos Para Modelo"

    # Hash calculado aquí (ámbito superior)
    try:
        current_data_hash = st.session_state.network_data.get_hash()
    except Exception:
        current_data_hash = 0 
            
    # Sidebar
    st.sidebar.header("Modificar Datos")
    
    # Botón de descarga manual
    if st.sidebar.button("Datos de Google Drive", key="download_button"):
        with st.spinner("Descargando archivos y cargando DataFrames..."):
            downloaded_data = DataLoader.load_default()
            if downloaded_data:
                st.session_state.network_data = downloaded_data
                reset_solution_state()
                st.success("Descarga y carga de datos completada.")
                st.rerun()
            else:
                st.error("Falló la descarga. Intente subir los archivos manualmente.")

    # Widget de subida de archivos
    uploaded_files = st.sidebar.file_uploader(
        "Subir uno o más archivos CSV", 
        type=['csv'], 
        accept_multiple_files=True,
        key="global_file_uploader"
    )
    
    expected_keys = [f'{k}.csv' for k in data_keys] 
    
    if st.sidebar.button("Guardar CSVs Nuevos", key="apply_upload_button"):
        if not uploaded_files:
            st.sidebar.warning("Por favor, seleccione al menos un archivo para subir.")
        else:
            with st.spinner("Cargando y reemplazando datos..."):
                temp_data = st.session_state.network_data 
                files_processed_count = 0
                
                for file in uploaded_files:
                    file_name = file.name.lower()
                    df_key = file_name.replace('.csv', '')
                    
                    if f"{df_key}.csv" in expected_keys:
                        try:
                            df = pd.read_csv(file)
                            if not df.equals(temp_data.__getattribute__(df_key)):
                                temp_data.update_data(df_key, df) 
                                files_processed_count += 1
                                logger.info(f"Archivo '{file.name}' cargado y actualizado.")
                            
                        except Exception as e:
                            st.sidebar.error(f"Error al leer '{file.name}'. Verifique que sea un CSV válido.")
                            logger.error(f"Error al cargar archivo CSV: {e}")
                            
                    else:
                        st.sidebar.warning(f"Archivo '{file.name}' ignorado. Nombre no reconocido.")

                if files_processed_count > 0:
                    st.session_state.network_data = temp_data
                    reset_solution_state()
                    st.session_state.active_tab = "Datos Para Modelo" 
                    st.rerun()
                else:
                    st.sidebar.info("No se subió ningún archivo válido para actualizar.")

    st.sidebar.markdown("---")

    # --- Control de Pestañas y Rendering ---
    tab_labels = [" Datos Para Modelo ", " Resultados y Flujos "]
    
    if st.session_state.active_tab == "Datos Para Modelo": active_tab_index = 0
    elif st.session_state.active_tab == "Resultados Óptimos": active_tab_index = 1
    else: active_tab_index = 0 
        
    tab1, tab2 = st.tabs(tab_labels)

    with tab1:
        
        # Usamos columnas para colocar el botón de ejecución en la parte superior derecha
        header_col, button_col = st.columns([3, 1])
        
        with header_col:
            st.header(tab_labels[0])
            st.info("💡 Edita las tablas de datos de la red directamente. Los cambios se aplicarán inmediatamente a la data en memoria.")

        with button_col:
            st.markdown("##") # Espacio para alinear el botón
            
            # Advertencia de datos modificados
            if 'lp_model' in st.session_state and st.session_state.get('data_hash_at_solve') != current_data_hash:
                st.warning("⚠️ Datos modificados. Vuelva a ejecutar.")

            # BOTÓN DE EJECUCIÓN SUPERIOR
            if st.button("🚀 Ejecutar Modelo", key="run_model_button", use_container_width=True):
                # Lógica de ejecución
                solver_option = 'glpk' 

                processor = DataProcessor(st.session_state.network_data)
                params = processor.get_params()
                
                with st.spinner("Paso 1: Construyendo y resolviendo el modelo..."):
                    lp_model = DistributionNetworkLP(processor, params)
                    if not lp_model.build_model():
                         st.error("Error al construir el modelo. Verifique los datos.")
                         st.stop()
                         
                    # Cálculo de producción total para el KPI
                    produccion_total = sum(
                        lp_model.model.x[p, c, k].value or 0
                        for p in lp_model.P for c in lp_model.C for k in lp_model.K
                    )
                    capacidad_total_produccion = sum(lp_model.cap_prod.values())
                    
                    util_produccion = 0.0
                    if capacidad_total_produccion > 0:
                        util_produccion = (produccion_total / capacidad_total_produccion) * 100

                    # 2. Resolver el modelo
                    if lp_model.solve(solver_option):
                        st.session_state.lp_model = lp_model 
                        st.session_state.cost_data = lp_model.get_cost_breakdown()
                        st.session_state.solution_status = "optimal"
                        st.session_state.util_produccion = util_produccion 
                        st.session_state.data_hash_at_solve = current_data_hash 
                        
                        st.balloons()
                        st.session_state.active_tab = "Resultados Óptimos" 
                        st.rerun() 
                    else:
                        st.session_state.solution_status = "infeasible"
                        st.error("❌ No se encontró una solución óptima para los datos proporcionados. Verifique capacidades y demandas.")
        
        st.markdown("---") # Separador visual

        # --- Editor de DataFrames ---
        for key in data_keys:
            current_df = st.session_state.network_data.__getattribute__(key)
            
            # Si el DataFrame está vacío, usamos la estructura vacía para que el editor funcione
            if current_df.empty:
                current_df = get_empty_df_structure(key)
                
            with st.expander(f"📝 Editar tabla: **{key.capitalize()}** ({len(current_df)} filas)"):
                
                df_edited = st.data_editor(current_df, width='stretch', num_rows="dynamic", key=f"editor_{key}")
                
                if not df_edited.equals(current_df):
                    st.session_state.network_data.update_data(key, df_edited)
                    reset_solution_state() 
                    st.toast(f"Tabla '{key}' modificada. Por favor, VUELVA A EJECUTAR el modelo.", icon="✍️")
                        
    with tab2:
        st.header(tab_labels[1])

        if 'lp_model' in st.session_state and st.session_state.solution_status == "optimal":
            
            if st.session_state.get('data_hash_at_solve') != current_data_hash:
                st.warning("La solución que se muestra es obsoleta. Por favor, ejecute el modelo en la pestaña '1️⃣ Datos Para Modelo' con los datos actualizados.")
                st.stop()

            util_val = st.session_state.get('util_produccion', 0.0)
            cost_data = st.session_state.cost_data
            lp_model = st.session_state.lp_model 
            
            display_final_summary(cost_data) 
                
            st.markdown("---")
            
            st.subheader("Análisis de Costos y Recursos")
            
            plot_flow_distribution_st(lp_model)
            
            plot_utilization(lp_model)
            
            plot_cost_breakdown(lp_model, cost_data)
                                    
            st.subheader("Flujos Óptimos de la Red")
            df_pc, df_cj = lp_model.get_solution_dataframes()

            st.markdown("#### Flujos Planta → Centro (Detalle)")
            st.dataframe(df_pc, width='stretch') 

            st.markdown("#### Flujos Centro → Cliente (Detalle)")
            st.dataframe(df_cj, width='stretch') 

        else:
            # Mensaje de instrucción si el modelo no ha sido ejecutado con éxito
            st.info("Modelo no ejecutado. Vaya a la pestaña '1️⃣ Datos Para Modelo' y presione el botón para generar los resultados.")


if __name__ == "__main__":
    main_app()