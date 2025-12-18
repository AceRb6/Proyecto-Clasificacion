Detección de Riesgo Suicida en Redes Sociales mediante NLP y Machine Learning
Este proyecto desarrolla un sistema de clasificación de texto basado en Procesamiento de Lenguaje Natural (NLP) para identificar ideación suicida en publicaciones de redes sociales. Utiliza modelos de Deep Learning (Sentence Transformers) para la representación semántica y algoritmos de Machine Learning clásicos para la clasificación, integrando variables demográficas inferidas heurísticamente.

Descripción del Proyecto
La detección temprana de señales de riesgo suicida en plataformas digitales es un desafío crítico debido a la ambigüedad del lenguaje y el volumen de datos. Este proyecto aborda el problema mediante un pipeline que:

Extrae y limpia datos de foros de salud mental.

Imputa información demográfica (Género y Edad) mediante Supervisión Débil (Weak Supervision) y modelos auxiliares.

Genera vectores semánticos densos (Embeddings) para capturar el contexto emocional.

Clasifica el riesgo utilizando modelos robustos y explicables.

Dataset
El conjunto de datos consta de 232,074 registros recopilados de Reddit, específicamente de los subreddits:

r/SuicideWatch (Clase: suicide)


r/depression (Clase: non-suicide) 

Estructura de datos:

text: Contenido completo de la publicación.

class: Etiqueta binaria (suicide / non-suicide).

gender (Inferido): Male, Female, Unknown.

age_group (Inferido): Adolescent, Young Teen, Young Adult, Adult.

Arquitectura y Metodología
1. Extracción de Metadatos (Heurística + ML Auxiliar)
Ante la falta de etiquetas demográficas explícitas, se implementó una estrategia en dos fases:

Fase Heurística (Regex): Extracción basada en patrones lingüísticos explícitos (ej. "I am a 23 year old man").

Fase de Imputación (Machine Learning): Entrenamiento de modelos Random Forest con TF-IDF para clasificar usuarios sin etiqueta explícita.


Técnicas aplicadas: SMOTE para balanceo de clases minoritarias (Adultos) y class_weight='balanced'.

Ingeniería de Características (Feature Engineering)

Representación de Texto: Uso del modelo pre-entrenado all-MiniLM-L6-v2 (SentenceTransformer) para generar embeddings densos de 384 dimensiones.

Análisis de Sentimiento: Cálculo de Polaridad y Subjetividad utilizando TextBlob para añadir contexto emocional explícito.

Modelado Predictivo
Se evaluaron tres arquitecturas principales para la tarea de clasificación final:

Random Forest Classifier: Seleccionado por su robustez ante el ruido de las etiquetas demográficas y capacidad de manejo de datos heterogéneos.

SVM (Kernel Lineal): Evaluado para probar la separabilidad lineal de los embeddings (Accuracy alcanzado: 0.93).

XGBoost: Utilizado como baseline de alto rendimiento (Gradient Boosting).

Resultados y Evaluación
El modelo final alcanza un rendimiento robusto, validando la calidad de los embeddings semánticos.

Modelo	Accuracy	F1-Score (Macro)	Observaciones
SVM (Lineal)	0.93	0.93	Mejor separación de clases en el hiperplano.
XGBoost	0.93	0.93	Rendimiento idéntico al SVM, mayor costo computacional.
Random Forest	0.92	0.92	Mayor tolerancia al ruido en variables de Edad/Género.
Visualización de Separabilidad (UMAP)
La proyección en 3D de los embeddings muestra dos clústeres definidos con una zona de interfaz correspondiente a la ambigüedad lingüística inherente (ej. expresiones de cansancio vs. depresión), justificando el "techo" de precisión del 93%.

Instalación y Uso
Clonar el repositorio:
git clone https://github.com/usuario/suicide-risk-detection.git

Instalar dependencias:
pip install -r requirements.txt

Ejecutar el pipeline de entrenamiento:
Python
