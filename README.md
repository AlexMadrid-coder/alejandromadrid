# Research Assistant in UV

Bienvenido al repositorio que he creado para mis prácticas extracurriculares como 'Research Assistant' para la Universidad de Valencia. Este repositorio contiene los 3 temas sobre los que he trabajado: la aplicación de la librería de python PandasAI par el tratamiento y preprocesado de datos, el uso de AIDE para resolver complejos problemas de python y la aplicabilidad de algoritmos tipo Transformer sobre datos tabulares.

## Table de Contenidos

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [License](#license)
- [Contact](#contact)

## Overview

Este repositorio contiene mi investigación e implementación de los siguientes temas:

1. **PandasAI**: Aplicado al procesamiento, resolución y preprocesado en datasets.
2. **AIDE**: Utilizado para resolver complejos problemas de clasificación en Python
3. **Transformer Algorithms**: Implementación de algoritmos transformer evaluando rendimiento y precisión en diversos datasets.

## Project Structure

El repositorio se organiza de la siguiente manera:

- Primero tenemos las carpetas de proyecto o carpetas de investigación que son aquellas con datasets sobre las que he aplicado las diversas librerías:
    - AIDE-[nombre_del_dataset] : En este tipo de proyectos he aplicado la librería AIDE para resolver el problema.
    - PANDASAI-[nombre_del_dataset] : En este tipo de proyectos he aplicado la librería PandasAI para resolver el problema.
    - TRANSFORMER-[nombre_del_dataset] : En este tipo de proyectos he aplicado algoritmos transformer para resovler el problema.
      
- Lo segundo que encontramos son los ficheros de 'requirements' donde tengo las librerías para cada uno de los temas a investigar. Decidí crear 3 entornos virtuales distintos, uno para cada tema, para mantener una instalación por tema más pequeña por si cualquier persona quería revisar mi trabajo hecho sobre un tema no tener que ir revisando notebook por notebook que librerías tengo o no tengo. En todas estoy usando la versión Python3.10.0 que ofrece Pyenv.

- Lo tercero que encontramos es el fichero pandasai.log que no es otro que aquel en el que guardamos tanto los input-prompts como los output-prompts de la librería PandasAI para tener información de como hacemos query así como para agilizar el proceso de las siguientes consultas monotema.


## Installation

Para replicar los experimentos y ejecutar este código sigue estos pasos: 

1. Clonar el repositorio:
    ```bash
    git clone https://github.com/yourusername/repository-name.git
    ```
2. Navega al directorio del repositorio:
    ```bash
    cd repository-name
    ```
3. Crea un entorno virtual:
    ```bash
    python -m venv env
    ```
4. Activa el entorno virtual:
    - Para Windows:
        ```bash
        .\env\Scripts\activate
        ```
    - En macOS y Linux:
        ```bash
        source env/bin/activate
        ```
5. Instala los paquetes necesarios:
    ```bash
    pip install -r .python-[librería-a-usar]-requirements.txt
    ```

## Usage

Las instrucciones para usar cada librería están detalladas en cada una de sus carpetas. En general sería:

1. Entra en la carpeta que quieres (e.g., `PandasAI-[problem_name]`).
2. Abre el notebook de jupyter (.ipynb) y explora resultados y procedimientos.
3. Ejecuta los scripts para contemplar por tus propios ojos los procedimientos.

## Results

The results of applying the different techniques to various problems are documented within each folder. You can find detailed explanations, visualizations, and conclusions in the Jupyter notebooks and markdown files provided.

## License

Este proyecto está bajo la licencia MIT. Ver el fichero [LICENSE](LICENSE) para más detalles

## Contact

Para cualqueir petición de contacto o comentario, puedes contactarme en [amaga2@alumni.uv.es](mailto:amaga2@alumni.uv.es).

