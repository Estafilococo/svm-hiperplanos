Máquinas de Vectores de Soporte (SVM) – Hiperplanos de Separación

Este proyecto explora el **algoritmo de Máquinas de Vectores de Soporte (SVM)**, haciendo hincapié en la teoría de los **hiperplanos de separación**. Se abordan los fundamentos matemáticos, la implementación en Python (desde cero y usando Scikit-Learn), así como ejemplos de visualización.

## Descripción General

En este repositorio se encuentra un **notebook de Google Colab** que desarrolla de forma detallada la teoría de las SVM, desde su formulación matemática (problema de optimización) hasta la implementación práctica paso a paso. Se incluyen:

- Explicaciones teóricas y demostraciones matemáticas enfocadas al **margen de separación** de las SVM.
- Implementación manual de un SVM lineal usando **descenso subgradiente** (sin bibliotecas de alto nivel para el entrenamiento).
- Ejemplos de uso y comparación con la **API de Scikit-Learn**.
- Visualizaciones en 2D y 3D para ilustrar los hiperplanos, márgenes y vectores de soporte.

---

## Estructura de Archivos

- **svm_hyperplanes.ipynb**: Notebook principal de Google Colab con:
  1. Desarrollo teórico.
  2. Implementación de un SVM lineal “desde cero”.
  3. Ejemplos de entrenamiento y visualización.

---

## Requisitos y Dependencias

- **Python 3.7+**  
- Bibliotecas principales:
  - **NumPy** (≥ 1.19)
  - **Matplotlib** (≥ 3.2)
  - **Scikit-Learn** (≥ 0.24)
  - **mpl_toolkits.mplot3d** (para visualización 3D, viene con Matplotlib)

---

## Resumen Teórico

### Formulación Primal y Dual de SVM

Las SVM buscan un **hiperplano** que separe dos clases (etiquetadas +1 y -1) con el **máximo margen**. En la formulación primal de “margen duro” (sin errores):
\[
\begin{aligned}
\min_{\mathbf{w},b} \quad & \frac{1}{2}\|\mathbf{w}\|^2, \\
\text{sujeto a}\quad & y_i(\mathbf{w}^\top \mathbf{x}_i + b) \ge 1,\;\; i=1,\dots,n.
\end{aligned}
\]

Cuando no existe separación perfecta, se introduce un **margen suave** con variables de holgura \(\xi_i\) y un parámetro \(C\) que penaliza errores:
\[
\min_{\mathbf{w},b,\boldsymbol{\xi}} \frac{1}{2}\|\mathbf{w}\|^2 + C \sum_{i=1}^n \xi_i.
\]

La formulación **dual** reescribe el problema en términos de multiplicadores de Lagrange \(\alpha_i\), permitiendo usar **kernels** para extender SVM a problemas no lineales.

### Márgen Suave vs. Márgen Duro

- **Márgen duro**: Asume que los datos son linealmente separables sin errores. 
- **Márgen suave**: Permite errores y viola­cio­nes del margen mediante \(\xi_i\). Más robusto ante ruido.

### Pérdida Hinge

El entrenamiento de SVM se puede abordar como minimización de la pérdida:
\[
L(\mathbf{w},b) = \frac{1}{2}\|\mathbf{w}\|^2 + C \sum_{i=1}^n \max\bigl\{0,\,1 - y_i(\mathbf{w}^\top \mathbf{x}_i + b)\bigr\}.
\]
Esta **hinge loss** penaliza las muestras mal clasificadas o que quedan dentro del margen.

### Parámetro C

Controla el **trade-off** entre un margen más amplio (menor overfitting) y menos errores de clasificación.  
- **C grande**: castiga fuertemente los errores, reduciendo el margen.  
- **C pequeño**: prioriza un margen amplio, admite más errores en entrenamiento.

---

## Implementación en Python

### Generación de Datos Sintéticos

Se crean conjuntos de datos bidimensionales (o tridimensionales) en torno a distintas distribuciones para probar la separación de clases y el efecto de los hiperparámetros. Se muestra cómo usar `numpy.random.randn` para generar ejemplos cerca de centros predefinidos.

### Entrenamiento desde Cero

Se implementa una clase `SVM` con actualización por **descenso de subgradiente**:
```python
class SVM:
    def __init__(...):
        ...
    def fit(self, X, y):
        ...
    def predict(self, X):
        ...
```
Para mostrar el proceso de optimización manual sin recurrir a bibliotecas externas de optimización.

### Comparación con Scikit-Learn

Se entrena un `SVC(kernel="linear", C=...)` de Scikit-Learn para comparar resultados (precisión, vectores de soporte, fronteras) con la implementación manual.

---

## Visualización de Resultados

Se crean gráficos en **2D** y **3D** para ilustrar:
- El **hiperplano** de separación (\(\mathbf{w}\cdot\mathbf{x} + b = 0\)).
- Las **líneas de margen** (\(\pm 1\)).
- Posición de los **vectores de soporte**.  
En el caso 3D, se traza la superficie correspondiente al plano y se observan los datos distribuidos en el espacio.

También se compara el efecto de distintos valores de \(C\) (alto vs. bajo) en la forma del hiperplano y el número de vectores de soporte.


---

## Referencias y Bibliografía

- Cortes, C., Vapnik, V. (1995). *Support-vector networks*. **Machine Learning**, 20(3), 273–297.  
- Burges, C. J. C. (1998). *A Tutorial on Support Vector Machines for Pattern Recognition*. **Data Mining and Knowledge Discovery**, 2(2), 121–167.  
- Schölkopf, B., Smola, A. (2002). *Learning with Kernels*. MIT Press.  
- Cristianini, N., Shawe-Taylor, J. (2000). *An Introduction to Support Vector Machines*. Cambridge University Press.  
- Mountrakis, G., Im, J., & Ogole, C. (2011). *Support vector machines in remote sensing: A review*. **ISPRS Journal of Photogrammetry and Remote Sensing**, 66(3), 247–259.  
- (Entre otras referencias incluidas en el notebook)

