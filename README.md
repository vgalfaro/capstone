# Capstone Itaú: Valorización de Opcionalidad de Prepago en Créditos Corporativos

Repositorio del proyecto Capstone Itaú 1 2025-2, para la valorización de la opcionalidad de prepago como una opción americana para créditos hipotecarios utilizando modelos discretos de tasas de interés.

## 📋 Descripción del Proyecto

Este proyecto aplica técnicas de finanzas cuantitativas para:
- **Modelar tasas de interés** mediante el modelo Ho-Lee y análisis de volatilidad EWMA
- **Valorizar opciones de prepago** en créditos corporativos
- **Análisis de componentes principales (PCA)** para reducción de dimensionalidad
- **Construcción de árboles binomiales** para pricing de derivados

## 📁 Estructura del Repositorio
```
capstone/
├── holee.ipynb                    # Modelo Ho-Lee - Implementación base
├── holee_EWMA.ipynb               # Modelo Ho-Lee con volatilidad EWMA (volatilidad variable)
├── holee_simple.ipynb             # Versión simplificada del modelo
├── holee_EWMA_test.ipynb          # Testing del modelo EWMA
├── PCA.ipynb                      # Análisis de componentes principales
├── aux_funct.py                   # Funciones auxiliares (volatilidad, árbol binomial)
├── df.txt                         # Factores de descuento (archivo de datos, raíz)
├── r1m.txt                        # Tasas de interés (1 mes, raíz)
├── sigma_EWMA.txt                 # Volatilidad EWMA calculada (raíz)
├── (otros archivos .xlsx/.csv)    # Posibles archivos de datos adicionales en la raíz
└── README.md                      # Este archivo
```

## 🔧 Tecnologías y Librerías

- **Python 3.x**
- **NumPy**: Cálculos numéricos y álgebra lineal
- **Pandas**: Manipulación de datos
- **Matplotlib**: Visualización de datos
- **Scikit-learn**: Análisis PCA
- **LaTeX**: Documentación del informe

## 📊 Descripción de Notebooks

### `holee.ipynb` 
Implementación completa del modelo Ho-Lee con:
- Construcción del árbol binomial de tasas
- Cálculo de precios Arrow-Debreu
- Calibración de drift
- Validación contra curva de mercado

### `holee_EWMA.ipynb`
Mejora del modelo base integrando:
- Estimación de volatilidad con EWMA (Exponentially Weighted Moving Average)
- Factor de decaimiento β = 0.94
- Normalización anualizada de volatilidad

### `holee_simple.ipynb`
Versión simplificada para entendimiento básico del modelo.

### `holee_EWMA_test.ipynb`
Suite de testing y validación del modelo EWMA.

### `PCA.ipynb`
Análisis exploratorio mediante:
- Descomposición en componentes principales
- Análisis de correlación de tasas
- Visualización de varianza explicada
- Reducción de dimensionalidad

## 🔬 Funciones Auxiliares (`aux_funct.py`)

### `ewma_volatility(file_name)`
Calcula la volatilidad variable en el tiempo usando EWMA:
- Lee tasas de retorno desde archivo
- Normaliza a frecuencia anualizada

### `HoLee(sigma, N, dt, r0, df_filename)`
Implementa el modelo Ho-Lee:
- **sigma**: Volatilidad de tasas
- **N**: Número de períodos
- **dt**: Intervalo de tiempo
- **r0**: Tasa inicial
- **df_filename**: Nombre de archivo con factores de descuento

Genera:
- Matriz de tasas `r[N+1, N+1]`
- Precios Arrow-Debreu `Q[N+1, N+1]`
- Vector de drifts `theta[N+1]`

## 📈 Datos

Los archivos de datos contienen:
- **df.txt**: Factores de descuento de mercado
- **r1m.txt**: Tasas spot a 1 mes y horizontes posteriores
- **sigma_EWMA.txt**: Volatilidad calculada mediante EWMA
- **Archivos Excel**: Datos de swaps e instrumentos financieros para calibración

Nota: Actualmente no existe una carpeta `Datos/` en el repositorio; los archivos de datos relevantes (archivos `.txt`, `.xlsx`, `.csv`) se encuentran en la raíz del proyecto. Al referenciar estos archivos desde los scripts o notebooks, use la ruta relativa desde la raíz, por ejemplo `r1m.txt` o `df.txt`.


## 🚀 Uso

1. **Preparar datos**: Asegurarse que los archivos .txt con tasas y factores de descuento estén disponibles

2. **Ejecutar análisis**:
   ```python
   from aux_funct import ewma_volatility, HoLee
   
   # Los archivos de datos están en la raíz del repositorio
   sigma = ewma_volatility('r1m.txt')
   r, Q, theta = HoLee(sigma, N=50, dt=1/12, r0=0.03, df_filename='df.txt')
   ```

3. **Revisar notebooks**: Abrir los notebooks de Jupyter para ver implementaciones completas y visualizaciones

<!-- ## 📋 Requisitos

```
numpy>=1.19
pandas>=1.2
matplotlib>=3.3
scikit-learn>=0.24
jupyter>=1.0
``` -->
