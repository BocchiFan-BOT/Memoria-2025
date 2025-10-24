# **Monitor inteligente detector de aglomeraciones**

Sistema de **monitoreo en tiempo real** diseñado para **detectar y contar personas** utilizando **visión por computador**. La solución se integra con una **interfaz web** que permite la **visualización en vivo** de las cámaras, los **conteos** actuales y la generación de **reportes históricos**.

### **Tecnologías Utilizadas**

| Backend | Detección | Frontend |
| :---- | :---- | :---- |
| **FastAPI** (Python) | **YOLOv8** | **React** |

## **Requisitos**

Asegúrate de tener instaladas las siguientes herramientas:

* **Python** (versión $\\geq 3.9$)  
* **Node.js** (versión $\\geq 16$)  
* **Git**
* **MySQL Workbenck**

## **Configuración e instalación**

### **1\. Backend (FastAPI)**

El backend maneja la lógica de la aplicación, el procesamiento de video con YOLOv8 y la comunicación con la base de datos.

\# Montar BD
- Correr MySQL Workbench con el puerto por defecto (3306)
- Crear base de datos vacia ejecutando : CREATE DATABASE memoria;
- Importar script .sql (crea la tabla automaticamete)

\# Configurar el archivo .env
Editar el archivo con las credenciales que corresponda (ajustar contraseña)

\# Navegar al directorio del backend  
cd backend

\# 1\. Crear y activar el entorno virtual  
\# PowerShell:  
.\\venv\\Scripts\\Activate.ps1  
\# O CMD:  
\# backend\\venv\\Scripts\\activate.bat  
\# O Linux/macOS:  
\# source venv/bin/activate

\# 2\. Instalar las dependencias de Python  
pip install \-r requirements.txt

\# Importante:   
\# Editar el archivo .env con las credenciales reales de la base de datos.

\# 3\. Verificar la conexión a la base de datos (Opcional)  
python \-m app.database.database

\# 4\. Ejecutar el servidor (se reinicia automáticamente con \--reload)  
uvicorn app.main:app \--reload \--port 8000

### **2\. Frontend (React)**

El frontend proporciona la interfaz de usuario para la visualización y gestión del sistema.

\# Navegar al directorio del frontend  
cd frontend  
\# 1\. Instalar las dependencias de Node.js  
npm install  
\# 2\. Iniciar la aplicación web  
npm start

### **Credenciales por Defecto**

| Campo | Valor |
| :---- | :---- |
| **Usuario** | admin |
| **Contraseña** | admin |


