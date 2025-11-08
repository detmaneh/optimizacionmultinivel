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

# --- Configuración Visual Profesional / Tecnológica ---
warnings.filterwarnings('ignore')
sns.set_theme(style="whitegrid", palette="deep") 
plt.rcParams['figure.dpi'] = 100 
plt.style.use('dark_background') # Estilo oscuro para look tecnológico

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
        
        # Iterar sobre las tablas del modelo
        for key in ['plantas', 'centros', 'clientes', 'productos', 'costos']:
            df = getattr(self, key)
            if df is None or df.empty:
                continue

            # Usar la forma más estable de Pandas para hashing:
            # 1. Convertir todo a string/tupla para evitar problemas de tipos/overflow.
            # 2. Convertir el DataFrame a un hash estable.
            
            # Convierte el DF a una tupla de tuplas (estable) y aplica hash
            df_hash = hash(tuple(map(tuple, df.values)))
            
            # Asegurar que hash_val se actualice de manera segura
            hash_val += df_hash
            
        return hash_val

    def update_data(self, key: str, df: pd.DataFrame):
        if hasattr(self, key):
            expected_cols = get_empty_df_structure(key).columns.tolist()
            if set(df.columns) != set(expected_cols):
                 df = df.reindex(columns=expected_cols, fill_value=None)
                 
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
            status_container.info("🌐 Iniciando Descarga de Assets de Red desde Repositorio...")
        
        downloaded_files = {}
        logger.info("Iniciando intento de descarga de datos por defecto desde Google Drive.")
        
        for name, file_id in DataDownloader.FILE_IDS.items():
            output = f'{name}.csv'
            url = f'https://drive.google.com/uc?id={file_id}'
            try: 
                gdown.download(url, output, quiet=True, fuzzy=True)
                downloaded_files[name] = output
                logger.info(f"✅ Éxito: '{output}' descargado.")
            except Exception as e: 
                logger.warning(f"❌ Falló la descarga de '{output}'. Error: {e}")
        
        if status_container:
            status_container.empty()
            if len(downloaded_files) == len(DataDownloader.FILE_IDS):
                st.success("✅ Todos los archivos descargados. Sistema Listo.")
            else:
                 st.warning("⚠️ Falló la descarga de algunos archivos. Revise la terminal para detalles.")
            
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
            self.P = [str(x) for x in data.plantas['Planta'].unique() if pd.notna(x)]
            self.C = [str(x) for x in data.centros['Centro'].unique() if pd.notna(x)]
            self.J = [str(x) for x in data.clientes['Cliente'].unique() if pd.notna(x)]
            self.K = [str(x) for x in data.productos['Producto'].unique() if pd.notna(x)]
        except KeyError:
            self.P, self.C, self.J, self.K = [], [], [], []

    def get_sets(self): return {'plantas': self.P, 'centros': self.C, 'clientes': self.J, 'productos': self.K}

    def get_params(self):
        try:
            demanda = self.data.clientes.set_index(['Cliente', 'Producto'])['Demanda'].apply(lambda x: float(x) if pd.notna(x) else 0).to_dict()
            cap_produccion = self.data.plantas.set_index(['Planta', 'Producto'])['Capacidad_Produccion'].apply(lambda x: float(x) if pd.notna(x) else 0).to_dict()
            cap_almacenamiento = self.data.centros.set_index(['Centro', 'Producto'])['Capacidad_Almacenamiento'].apply(lambda x: float(x) if pd.notna(x) else 0).to_dict()
            costo_produccion = self.data.plantas.set_index(['Planta', 'Producto'])['Costo_Produccion'].apply(lambda x: float(x) if pd.notna(x) else 0).to_dict()
            
            costo_planta_centro = {}
            df_pc_filtered = self.data.costos.dropna(subset=['Planta', 'Centro', 'Producto', 'Costo_Plant_Centro'])
            for (p, c, k), group in df_pc_filtered.groupby(['Planta', 'Centro', 'Producto']):
                costo_planta_centro[(str(p), str(c), str(k))] = float(group['Costo_Plant_Centro'].iloc[0])

            costo_centro_cliente = {}
            df_cj_filtered = self.data.costos.dropna(subset=['Centro', 'Cliente', 'Producto', 'Costo_Centro_Cliente'])
            for (c, j, k), group in df_cj_filtered.groupby(['Centro', 'Cliente', 'Producto']):
                costo_centro_cliente[(str(c), str(j), str(k))] = float(group['Costo_Centro_Cliente'].iloc[0])
            
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
            logger.warning("Modelo inválido: Conjuntos de datos vacíos o incompletos.")
            return False

        # Conjuntos
        modelo.P, modelo.C, modelo.J, modelo.K = Set(initialize=self.P), Set(initialize=self.C), Set(initialize=self.J), Set(initialize=self.K)
        
        # Variables
        modelo.x = Var(modelo.P, modelo.C, modelo.K, domain=NonNegativeReals)
        modelo.y = Var(modelo.C, modelo.J, modelo.K, domain=NonNegativeReals)

        # Función Objetivo
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
        except Exception as e:
            logger.error(f"Error al resolver el modelo con {solver_name}: {e}")
            return False

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
        
        costo_prod = sum(
            self.costo_prod.get((p, k), 0) * (self.model.x[p, c, k].value or 0)
            for p in self.P for c in self.C for k in self.K
        )
        costo_pc = sum(
            self.costo_pc.get((p, c, k), 0) * (self.model.x[p, c, k].value or 0)
            for p in self.P for c in self.C for k in self.K
        )
        costo_cj = sum(
            self.costo_cj.get((c, j, k), 0) * (self.model.y[c, j, k].value or 0)
            for c in self.C for j in self.J for k in self.K
        )
        costo_total = self.model.objetivo()

        return {'Costo Total': costo_total, 'Producción': costo_prod, 'Transporte Planta → Centro': costo_pc, 'Transporte Centro → Cliente': costo_cj}

# --- FUNCIONES DE GRÁFICO ACTUALIZADAS PARA USAR DATOS FILTRADOS ---

def plot_flow_distribution_st_filtered(solution_data: Dict):
    col_plot, _ = st.columns([0.5, 0.5]) # Usar el 50% de ancho

    with col_plot:
        st.subheader("Flujos de la Red (Filtrado)")
        df_pc_filtered = solution_data['df_pc']
        df_cj_filtered = solution_data['df_cj']
        lp_model = solution_data['lp_model']
        
        P, C, J = lp_model.P, lp_model.C, lp_model.J

        fig, axes = plt.subplots(2, 1, figsize=(10, 8)) # Ajustar el tamaño del gráfico
        
        # Flujo Planta → Centro
        if not df_pc_filtered.empty:
            df_pivot = df_pc_filtered.groupby(['Planta', 'Centro'])['Cantidad'].sum().unstack(fill_value=0)
            df_pivot.plot(kind='bar', stacked=True, ax=axes[0], cmap='viridis', edgecolor='white', linewidth=0.5)
            axes[0].set_title('Flujo: Planta → Centro de Distribución', fontsize=12, fontweight='bold', color='lightblue')
            axes[0].set_ylabel('Unidades Enviadas', fontsize=10, color='silver')
            axes[0].set_xlabel('Planta', fontsize=10, color='silver')
            axes[0].legend(title='Centro', loc='upper left', bbox_to_anchor=(1.02, 1), fontsize=8)
            axes[0].tick_params(axis='x', rotation=45 if len(P) > 3 else 0, colors='silver')
            axes[0].tick_params(axis='y', colors='silver')
        else:
            axes[0].text(0.5, 0.5, 'Sin flujo P → C (Flujo cero o filtrado)', ha='center', va='center', fontsize=10, color='red')

        # Flujo Centro → Cliente
        if not df_cj_filtered.empty:
            df_cj_filtered['Cliente_Grupo'] = df_cj_filtered['Cliente'] 
            num_clientes = len(df_cj_filtered['Cliente'].unique())
            
            # Agrupar clientes si hay muchos
            if num_clientes > 10:
                totales_cliente = df_cj_filtered.groupby('Cliente')['Cantidad'].sum()
                top_clientes = totales_cliente.nlargest(9).index
                df_cj_filtered['Cliente_Grupo'] = df_cj_filtered['Cliente'].apply(
                    lambda x: x if x in top_clientes else 'Otros'
                )
                df_pivot = df_cj_filtered.groupby(['Centro', 'Cliente_Grupo'])['Cantidad'].sum().unstack(fill_value=0)
            else:
                df_pivot = df_cj_filtered.groupby(['Centro', 'Cliente'])['Cantidad'].sum().unstack(fill_value=0)

            df_pivot.plot(kind='bar', stacked=True, ax=axes[1], cmap='plasma', edgecolor='white', linewidth=0.5)
            axes[1].set_title('Flujo: Centro de Distribución → Cliente', fontsize=12, fontweight='bold', color='lightblue')
            axes[1].set_ylabel('Unidades Enviadas', fontsize=10, color='silver')
            axes[1].set_xlabel('Centro', fontsize=10, color='silver')
            axes[1].legend(title='Cliente', loc='upper left', bbox_to_anchor=(1.02, 1), fontsize=8)
            axes[1].tick_params(axis='x', rotation=45 if len(C) > 5 else 0, colors='silver')
            axes[1].tick_params(axis='y', colors='silver')

        else:
            axes[1].text(0.5, 0.5, 'Sin flujo C → J (Flujo cero o filtrado)', ha='center', va='center', fontsize=10, color='red')

        plt.tight_layout(rect=[0, 0, 0.88, 1])
        st.pyplot(fig, use_container_width=True)


def plot_utilization_filtered(solution_data: Dict):
    col_plot, _ = st.columns([0.5, 0.5]) # Usar el 50% de ancho

    with col_plot:
        st.subheader("Utilización de Recursos (Flujo Filtrado)")
        df_pc_filtered = solution_data['df_pc']
        df_cj_filtered = solution_data['df_cj']
        lp_model = solution_data['lp_model']
        
        P, C, J, K = lp_model.P, lp_model.C, lp_model.J, lp_model.K
        demanda = lp_model.demanda

        fig, axes = plt.subplots(3, 1, figsize=(10, 8)) # Ajustar el tamaño del gráfico
        
        # Utilización de Plantas
        util_plantas = []
        plantas_list = []
        flujo_por_planta = df_pc_filtered.groupby('Planta')['Cantidad'].sum().to_dict()

        for p in P:
            produccion = flujo_por_planta.get(p, 0)
            capacidad = sum(lp_model.cap_prod.get((p, k), 0) for k in K)
            util = (produccion / capacidad * 100) if capacidad > 0 else 0
            util_plantas.append(util)
            plantas_list.append(p)
            
        bars = axes[0].bar(plantas_list, util_plantas, color='cyan', alpha=0.8, edgecolor='white')
        axes[0].set_title('Utilización: Plantas', fontsize=12, fontweight='bold', color='lightblue')
        axes[0].set_ylabel('Utilización (%)', fontsize=10, color='silver')
        axes[0].set_ylim([0, 110])
        axes[0].axhline(y=100, color='red', linestyle='--', alpha=0.7, label='Máximo')
        axes[0].legend()
        axes[0].tick_params(axis='x', rotation=45 if len(P) > 3 else 0, colors='silver')
        axes[0].tick_params(axis='y', colors='silver')

        for bar, util in zip(bars, util_plantas):
            axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                        f'{util:.1f}%', ha='center', va='bottom', fontsize=7, fontweight='bold', color='white')

        # Utilización de Centros
        util_centros = []
        centros_list = []
        flujo_por_centro = df_pc_filtered.groupby('Centro')['Cantidad'].sum().to_dict()

        for c in C:
            flujo = flujo_por_centro.get(c, 0)
            capacidad = sum(lp_model.cap_alm.get((c, k), 0) for k in K)
            util = (flujo / capacidad * 100) if capacidad > 0 else 0
            util_centros.append(util)
            centros_list.append(c)

        bars = axes[1].bar(centros_list, util_centros, color='magenta', alpha=0.8, edgecolor='white')
        axes[1].set_title('Utilización: Centros', fontsize=12, fontweight='bold', color='lightblue')
        axes[1].set_ylabel('Utilización (%)', fontsize=10, color='silver')
        axes[1].set_ylim([0, 110])
        axes[1].axhline(y=100, color='red', linestyle='--', alpha=0.7)
        axes[1].tick_params(axis='x', rotation=45, colors='silver')
        axes[1].tick_params(axis='y', colors='silver')

        for bar, util in zip(bars, util_centros):
            axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                        f'{util:.1f}%', ha='center', va='bottom', fontsize=7, fontweight='bold', color='white')

        # Satisfacción de Clientes
        satisfaccion = []
        clientes_list = []
        recibido_por_cliente = df_cj_filtered.groupby('Cliente')['Cantidad'].sum().to_dict()
        
        for j in J:
            recibido = recibido_por_cliente.get(j, 0)
            demandado = sum(demanda.get((j, k), 0) for k in K)
            sat = (recibido / demandado * 100) if demandado > 0 else 0
            satisfaccion.append(sat)
            clientes_list.append(j)
        
        satisfaccion_arr = np.array(satisfaccion)
        rango_satisfaccion = satisfaccion_arr.max() - satisfaccion_arr.min()

        if len(J) > 20 and rango_satisfaccion > 5:
            # Histograma
            axes[2].hist(satisfaccion, bins=15, color='lime', alpha=0.8, edgecolor='white')
            axes[2].set_xlabel('Satisfacción (%)', fontsize=10, color='silver')
            axes[2].set_ylabel('Número de Clientes', fontsize=10, color='silver')
            axes[2].axvline(x=100, color='red', linestyle='--', alpha=0.7, label='100%')
            axes[2].set_title('Distribución de Satisfacción', fontsize=12, fontweight='bold', color='lightblue')
        else:
            # Gráfico de barras individual
            if len(J) <= 15:
                bars = axes[2].bar(clientes_list, satisfaccion, color='lime', alpha=0.8, edgecolor='white')
                axes[2].tick_params(axis='x', rotation=45, colors='silver')

                for bar, sat in zip(bars, satisfaccion):
                    axes[2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                               f'{sat:.1f}%', ha='center', va='bottom', fontsize=6, fontweight='bold', color='white')
            else:
                sat_dict = dict(zip(clientes_list, satisfaccion))
                clientes_con_demanda = {k: v for k, v in sat_dict.items() if sum(demanda.get((k, prod), 0) for prod in K) > 0}
                top_10_menores = sorted(clientes_con_demanda.items(), key=lambda x: x[1])[:10]
                
                if top_10_menores:
                    clientes_mostrar = [x[0] for x in top_10_menores]
                    sat_mostrar = [x[1] for x in top_10_menores]
                else:
                    clientes_mostrar = ["N/A"]
                    sat_mostrar = [0]


                bars = axes[2].bar(clientes_mostrar, sat_mostrar, color='lime', alpha=0.8, edgecolor='white')
                axes[2].tick_params(axis='x', rotation=45, colors='silver')
                axes[2].set_title('Top 10 Clientes con Menor Satisfacción', fontsize=12, fontweight='bold', color='lightblue')

            axes[2].set_ylabel('Satisfacción (%)', fontsize=10, color='silver')
            axes[2].set_ylim([0, 110])
            axes[2].axhline(y=100, color='red', linestyle='--', alpha=0.7, label='100%')
            axes[2].set_title('Satisfacción: Clientes', fontsize=12, fontweight='bold', color='lightblue')
            axes[2].tick_params(axis='y', colors='silver')


        axes[2].legend()
        plt.tight_layout()
        for i in range(3):
            formatter = ScalarFormatter(useOffset=False)
            formatter.set_scientific(False)
            axes[i].yaxis.set_major_formatter(formatter)

        st.pyplot(fig, use_container_width=True)


def plot_cost_breakdown(lp_model: 'DistributionNetworkLP', cost_data: Dict):
    col_plot, _ = st.columns([0.5, 0.5]) # Usar el 50% de ancho

    with col_plot:
        st.subheader("Desglose de Costos (Resultado Óptimo)")

        costo_prod = cost_data['Producción']
        costo_pc = cost_data['Transporte Planta → Centro']
        costo_cj = cost_data['Transporte Centro → Cliente']
        costo_total = cost_data['Costo Total']
        
        fig, ax = plt.subplots(figsize=(10, 6))

        categorias = ['Producción', 'Transporte\nPlanta→Centro', 'Transporte\nCentro→Cliente']
        costos = [costo_prod, costo_pc, costo_cj]
        colores = ['#4A90E2', '#FF4136', '#00FF7F'] # Colores vibrantes

        bars = ax.bar(categorias, costos, color=colores, alpha=0.85,
                      edgecolor='white', linewidth=1.5)

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(axis='y', linestyle='--', alpha=0.5, color='gray')

        total = sum(costos)
        for bar, costo in zip(bars, costos):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'${costo:,.0f}\n({(costo/total*100):.1f}%)',
                   ha='center', va='bottom', fontweight='bold', fontsize=9, color='white')

        ax.set_title('Desglose de Costos por Etapa', fontsize=14, fontweight='bold', pad=20, color='lightblue')
        ax.set_ylabel('Costo ($)', fontsize=12, color='silver')
        
        ax.yaxis.set_major_formatter(FuncFormatter(lambda x, p: f'${x:,.0f}'))
        ax.tick_params(axis='x', colors='silver')
        ax.tick_params(axis='y', colors='silver')


        fig.suptitle(f'Costo Total Mínimo: ${costo_total:,.2f}',
                     fontsize=12, fontweight='bold', y=0.98,
                     bbox=dict(boxstyle='round', facecolor='#2c3e50', edgecolor='#4A90E2', alpha=0.9))

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        
        st.pyplot(fig)

def display_final_summary(cost_data: Dict):
    total = cost_data['Costo Total']
    prod = cost_data['Producción']
    pc = cost_data['Transporte Planta → Centro']
    cj = cost_data['Transporte Centro → Cliente']

    st.markdown("## 📈 Estado de Optimización: [EJECUTADO]")
    st.markdown("---")
    st.success("✅ **Solución ÓPTIMA Encontrada y Verificada**")

    st.markdown(f"**💰 VALOR ÓPTIMO DE LA FUNCIÓN OBJETIVO:** **${total:,.2f}**")

    # Tabla de desglose
    df_summary = pd.DataFrame({
        'Concepto': ['Costo de Producción', 'Costo Transporte (P→C)', 'Costo Transporte (C→J)', 'TOTAL:'],
        'Costo': [f"${prod:,.2f}", f"${pc:,.2f}", f"${cj:,.2f}", f"**${total:,.2f}**"]
    })

    st.markdown("### 📊 Desglose Analítico:")
    st.dataframe(df_summary, hide_index=True, width=1000) 

    if total > 0.01:
        st.markdown(f"""
        **Distribución Porcentual:**
        * 🏭 Producción: **{(prod/total*100):.1f}%**
        * 🚚 Transporte P→C: **{(pc/total*100):.1f}%**
        * 📦 Transporte C→J: **{(cj/total*100):.1f}%**
        """)
    st.markdown("---")


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
    # Establecer la configuración de la página
    st.set_page_config(
        page_title="Network Optimization Platform", 
        layout="wide", 
        initial_sidebar_state="expanded"
    )

    st.title("🌐 Plataforma de Optimización de Red Logística Multinivel")
    st.markdown("---")

    # --- INICIALIZACIÓN DE VARIABLES DE ESTADO ---
    data_keys = ['plantas', 'centros', 'clientes', 'costos', 'productos']

    if 'network_data' not in st.session_state:
        empty_dfs = {key: get_empty_df_structure(key) for key in data_keys}
        st.session_state.network_data = NetworkData(**empty_dfs)
        st.balloons() 
        st.info("💡 Sugerencia: Use el panel lateral para 'Datos de Google Drive' o cargue sus propios archivos CSV.")
            
    if 'active_tab' not in st.session_state:
        st.session_state.active_tab = "Datos Para Modelo"

    try:
        current_data_hash = st.session_state.network_data.get_hash()
    except Exception:
        current_data_hash = 0 
            
    # Sidebar
    st.sidebar.header("🛠️ Configuración de Data Assets")
    
    # Botón de descarga manual
    if st.sidebar.button("Descargar Datos de Repositorio", key="download_button"):
        with st.spinner("Descargando y preparando DataFrames..."):
            downloaded_data = DataLoader.load_default()
            if downloaded_data:
                st.session_state.network_data = downloaded_data
                reset_solution_state()
                st.sidebar.success("✅ Archivos cargados.")
                st.rerun()
            else:
                st.sidebar.error("❌ Falló la descarga.")

    # Widget de subida de archivos
    uploaded_files = st.sidebar.file_uploader(
        "Upload CSV Assets", 
        type=['csv'], 
        accept_multiple_files=True,
        key="global_file_uploader"
    )
    
    expected_keys = [f'{k}.csv' for k in data_keys] 
    
    if st.sidebar.button("Aplicar CSVs Cargados", key="apply_upload_button"):
        if not uploaded_files:
            st.sidebar.warning("Seleccione archivos para continuar.")
        else:
            with st.spinner("Procesando y validando DataFrames..."):
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
                            
                        except Exception as e:
                            st.sidebar.error(f"Error al leer '{file.name}'. Verifique formato CSV.")
                            
                if files_processed_count > 0:
                    st.session_state.network_data = temp_data
                    reset_solution_state()
                    st.session_state.active_tab = "Datos Para Modelo" 
                    st.sidebar.success(f"✅ {files_processed_count} archivos actualizados.")
                    st.rerun()
                else:
                    st.sidebar.info("No se detectaron archivos nuevos válidos.")

    st.sidebar.markdown("---")

    # --- Control de Pestañas y Rendering ---
    tab_labels = [" 1️⃣ Configuración de Parámetros ", " 2️⃣ Ejecución y Resultados "]
        
    tab1, tab2 = st.tabs(tab_labels)

    # --- INICIO DE MODIFICACIÓN EN TABLA 1 (PRE-FILTROS) ---
    with tab1:
        
        st.markdown("### ⚙️ Filtrado de Parámetros de Red")
        
        current_data = st.session_state.network_data
        
        # Obtener opciones para los filtros
        productos_opt = ['Todos'] + sorted(list(current_data.productos['Producto'].unique())) if not current_data.productos.empty else ['Todos']
        plantas_opt = ['Todos'] + sorted(list(current_data.plantas['Planta'].unique())) if not current_data.plantas.empty else ['Todos']
        centros_opt = ['Todos'] + sorted(list(current_data.centros['Centro'].unique())) if not current_data.centros.empty else ['Todos']
        clientes_opt = ['Todos'] + sorted(list(current_data.clientes['Cliente'].unique())) if not current_data.clientes.empty else ['Todos']

        col_prefilt1, col_prefilt2, col_prefilt3, col_prefilt4 = st.columns(4)

        with col_prefilt1:
            prod_filter = st.multiselect("Producto(s) [K]", productos_opt, default=['Todos'])
        with col_prefilt2:
            plant_filter = st.multiselect("Planta(s) [P]", plantas_opt, default=['Todos'])
        with col_prefilt3:
            center_filter = st.multiselect("Centro(s) [C]", centros_opt, default=['Todos'])
        with col_prefilt4:
            client_filter = st.multiselect("Cliente(s) [J]", clientes_opt, default=['Todos'])
            
        # Determinar los valores finales a usar para el filtrado
        P_to_use = plant_filter if 'Todos' not in plant_filter else plantas_opt[1:]
        C_to_use = center_filter if 'Todos' not in center_filter else centros_opt[1:]
        J_to_use = client_filter if 'Todos' not in client_filter else clientes_opt[1:]
        K_to_use = prod_filter if 'Todos' not in prod_filter else productos_opt[1:]
        
        # --- APLICAR FILTROS Y CREAR DATA SUBSET PARA EL MODELO Y EL EDITOR ---
        filtered_data_dict = {
            'productos': current_data.productos[current_data.productos['Producto'].isin(K_to_use)],
            'plantas': current_data.plantas[
                (current_data.plantas['Planta'].isin(P_to_use)) & 
                (current_data.plantas['Producto'].isin(K_to_use))
            ],
            'centros': current_data.centros[
                (current_data.centros['Centro'].isin(C_to_use)) & 
                (current_data.centros['Producto'].isin(K_to_use))
            ],
            'clientes': current_data.clientes[
                (current_data.clientes['Cliente'].isin(J_to_use)) & 
                (current_data.clientes['Producto'].isin(K_to_use))
            ],
            'costos': current_data.costos[
                (current_data.costos['Planta'].isin(P_to_use)) & 
                (current_data.costos['Centro'].isin(C_to_use)) & 
                (current_data.costos['Cliente'].isin(J_to_use)) & 
                (current_data.costos['Producto'].isin(K_to_use))
            ].dropna(how='all', subset=['Costo_Plant_Centro', 'Costo_Centro_Cliente']), # Limpiar filas de costos vacías
        }
        
        filtered_network_data = NetworkData(**filtered_data_dict)
        filtered_data_hash = filtered_network_data.get_hash() 
        
        st.markdown("---")

        header_col, button_col = st.columns([3, 1])
        
        # with header_col:
        #     st.header("Data Assets del Modelo (Visualización Filtrada)")
        #     st.code(f"HASH_MODEL_DATA: {filtered_data_hash}", language='markdown')

        # with header_col:
        st.markdown("##") 
        
        if 'lp_model' in st.session_state and st.session_state.get('data_hash_at_solve') != filtered_data_hash:
            st.warning("⚠️ Data desactualizada. Re-ejecución requerida.")

        # BOTÓN DE EJECUCIÓN 
        if st.button("▶️ Ejecutar Algoritmo de Optimización", key="run_model_button", use_container_width=True):
            solver_option = 'glpk' 
            processor = DataProcessor(filtered_network_data) 
            params = processor.get_params()
            
            with st.spinner("Ejecutando Pyomo y Solver GLPK..."):
                lp_model = DistributionNetworkLP(processor, params)
                if not lp_model.build_model():
                        st.error("❌ Error en Construcción del Modelo. Verifique Data Filtrada.")
                        st.stop()
                        
                if lp_model.solve(solver_option):
                    st.session_state.lp_model = lp_model 
                    st.session_state.cost_data = lp_model.get_cost_breakdown()
                    st.session_state.solution_status = "optimal"
                    st.session_state.data_hash_at_solve = filtered_data_hash 
                    
                    st.snow()
                    st.session_state.active_tab = "Resultados Óptimos" 
                    st.rerun() 
                else:
                    st.session_state.solution_status = "infeasible"
                    st.error("❌ SOLUCIÓN NO ENCONTRADA: Modelo Infactible o Datos Insuficientes.")
                    st.stop()
        
        st.markdown("---") 

        # --- Editor de DataFrames (MOSTRANDO LA VERSIÓN FILTRADA) ---
        for key in data_keys:
            current_df = filtered_data_dict[key] 
            
            if current_df.empty:
                current_df = get_empty_df_structure(key)
                
            with st.expander(f"📝 Configurar Tabla: **{key.capitalize()}** ({len(current_df)} registros)"):
                
                df_edited = st.data_editor(current_df, width='stretch', num_rows="dynamic", key=f"editor_filtered_{key}_{filtered_data_hash}")
                
                if not df_edited.equals(current_df):
                    st.session_state.network_data.update_data(key, df_edited)

                    reset_solution_state() 
                    st.toast(f"Tabla '{key}' modificada. Se requiere una re-ejecución del modelo.", icon="✍️")
                    st.rerun() 
                        
    with tab2:
        st.header("📊 Análisis de Resultados Óptimos")

        if 'lp_model' in st.session_state and st.session_state.solution_status == "optimal":
            
            if st.session_state.get('data_hash_at_solve') != filtered_data_hash:
                st.warning("⚠️ Advertencia: La solución actual es obsoleta. Ejecute el modelo con la Data Asset en la Pestaña 1.")
                st.stop()

            # --- PREPARACIÓN DE DATOS PARA FILTROS DE VISUALIZACIÓN ---
            lp_model = st.session_state.lp_model
            cost_data = st.session_state.cost_data
            df_pc_orig, df_cj_orig = lp_model.get_solution_dataframes()
            
            # 1. Calcular Costos Unitarios/Totales
            df_pc_merged = df_pc_orig.copy()
            df_pc_merged['Costo_Unitario'] = df_pc_merged.apply(
                lambda row: lp_model.costo_pc.get((row['Planta'], row['Centro'], row['Producto']), 0) + lp_model.costo_prod.get((row['Planta'], row['Producto']), 0),
                axis=1
            )
            df_pc_merged['Costo_Total'] = df_pc_merged['Cantidad'] * df_pc_merged['Costo_Unitario']

            df_cj_merged = df_cj_orig.copy()
            df_cj_merged['Costo_Unitario'] = df_cj_merged.apply(
                lambda row: lp_model.costo_cj.get((row['Centro'], row['Cliente'], row['Producto']), 0),
                axis=1
            )
            df_cj_merged['Costo_Total'] = df_cj_merged['Cantidad'] * df_cj_merged['Costo_Unitario']

            # --- INTERFAZ DE FILTROS DE RESULTADOS ---
            st.markdown("### 🔍 Filtros Dinámicos de Visualización (Post-Optimización)")
            col_filt1, col_filt2, col_filt3, col_filt4 = st.columns([1, 1, 1, 1])

            productos_disp = ['Todos'] + sorted(lp_model.K)
            selected_producto = col_filt1.selectbox("Filtro: Producto", productos_disp)
            
            plantas_disp = ['Todos'] + sorted(lp_model.P)
            selected_planta = col_filt2.selectbox("Filtro: Planta", plantas_disp)

            centros_disp = ['Todos'] + sorted(lp_model.C)
            selected_centro = col_filt3.selectbox("Filtro: Centro", centros_disp)

            clientes_disp = ['Todos'] + sorted(lp_model.J)
            selected_cliente = col_filt4.selectbox("Filtro: Cliente", clientes_disp)
            
            max_costo = max(df_pc_merged['Costo_Unitario'].max() if not df_pc_merged.empty else 0,
                            df_cj_merged['Costo_Unitario'].max() if not df_cj_merged.empty else 0)
            
            costo_range = st.slider(
                "Rango de Costo Unitario ($) para Flujo/Visualización",
                min_value=0.0,
                max_value=max_costo + 0.5, 
                value=(0.0, max_costo + 0.5),
                step=0.01,
                format="$%.2f"
            ) if max_costo > 0 else (0.0, 1000.0) 
            
            st.markdown("---")
            
            # --- APLICACIÓN DE FILTROS DE VISUALIZACIÓN ---
            
            df_pc_filtered = df_pc_merged.copy()
            if selected_producto != 'Todos':
                df_pc_filtered = df_pc_filtered[df_pc_filtered['Producto'] == selected_producto]
            if selected_planta != 'Todos':
                df_pc_filtered = df_pc_filtered[df_pc_filtered['Planta'] == selected_planta]
            if selected_centro != 'Todos':
                df_pc_filtered = df_pc_filtered[df_pc_filtered['Centro'] == selected_centro]
            df_pc_filtered = df_pc_filtered[
                (df_pc_filtered['Costo_Unitario'] >= costo_range[0]) & 
                (df_pc_filtered['Costo_Unitario'] <= costo_range[1])
            ]

            df_cj_filtered = df_cj_merged.copy()
            if selected_producto != 'Todos':
                df_cj_filtered = df_cj_filtered[df_cj_filtered['Producto'] == selected_producto]
            if selected_centro != 'Todos':
                df_cj_filtered = df_cj_filtered[df_cj_filtered['Centro'] == selected_centro]
            if selected_cliente != 'Todos':
                df_cj_filtered = df_cj_filtered[df_cj_filtered['Cliente'] == selected_cliente]
            df_cj_filtered = df_cj_filtered[
                (df_cj_filtered['Costo_Unitario'] >= costo_range[0]) & 
                (df_cj_filtered['Costo_Unitario'] <= costo_range[1])
            ]
            
            solution_data = {
                'df_pc': df_pc_filtered,
                'df_cj': df_cj_filtered,
                'lp_model': lp_model
            }
            
            display_final_summary(cost_data) 
            
            # Gráficos en columnas (50% de ancho)
            
            col_cost_pie, col_util = st.columns([0.5, 0.5])
            
            with col_cost_pie:
                plot_cost_breakdown(lp_model, cost_data)
                
            with col_util:
                plot_utilization_filtered(solution_data)
            
            # Flujos
            plot_flow_distribution_st_filtered(solution_data)
                                    
            st.markdown("### 📋 Flujos Óptimos Detallados (Filtrado)")

            st.markdown("#### Flujos Planta → Centro")
            st.dataframe(df_pc_filtered[['Planta', 'Centro', 'Producto', 'Cantidad', 'Costo_Unitario', 'Costo_Total']], width='stretch', height=200) 

            st.markdown("#### Flujos Centro → Cliente")
            st.dataframe(df_cj_filtered[['Centro', 'Cliente', 'Producto', 'Cantidad', 'Costo_Unitario', 'Costo_Total']], width='stretch', height=200) 

        else:
            st.info("Modelo no ejecutado. Vaya a la pestaña '1️⃣ Configuración de Parámetros' para iniciar la optimización.")


if __name__ == "__main__":
    main_app()