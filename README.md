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

## **Configuración e instalación**

### **1\. Backend (FastAPI)**

El backend maneja la lógica de la aplicación, el procesamiento de video con YOLOv8 y la comunicación con la base de datos.

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

\# 3\. Configurar variables de entorno  
\# Copiar el archivo de ejemplo para crear el archivo .env  
cp .env.example .env 

\# Importante:   
\# Editar el archivo .env con las credenciales reales de la base de datos.

\# 4\. Verificar la conexión a la base de datos (Opcional)  
python \-m app.database.database

\# 5\. Ejecutar el servidor (se reinicia automáticamente con \--reload)  
uvicorn app.main:app \--reload \--port 8000

### **2\. Frontend (React)**

El frontend proporciona la interfaz de usuario para la visualización y gestión del sistema.

\# Navegar al directorio del frontend  
cd frontend  
\# 1\. Instalar las dependencias de Node.js  
npm install  
\# 2\. Iniciar la aplicación web  
npm start

## **Autenticación y uso de JWT**

El sistema implementa autenticación mediante JSON Web Tokens (JWT) para proteger las rutas administrativas del backend. Este token sirve para identificar al usuario en las peticiones siguientes sin tener que enviar usuario y contraseña cada vez.

### **Cómo probar rutas protegidas (JWT)**

**Nota:** Esto es solo para **desarrolladores/administradores** al probar el backend manualmente. En el uso real, el **frontend** hace el login y envía el token **automáticamente**; los usuarios finales **no** usan la consola.

#### **Ejecutar Prueba Completa (Login \+ Uso del Token)**

**PowerShell (Windows):**

\# 1\. Obtener el token (Login)  
$login\_response \= Invoke-RestMethod \-Uri "\[http://127.0.0.1:8000/auth/login\](http://127.0.0.1:8000/auth/login)" \`  
  \-Method POST \`  
  \-ContentType "application/json" \`  
  \-Body '{"username":"admin","password":"admin"}'

$TOKEN \= $login\_response.access\_token  
Write-Host "Token obtenido correctamente. Probando ruta protegida..."

\# 2\. Usar el token en una ruta protegida (Ejemplo: /camaras/sync-file)  
Invoke-RestMethod \-Uri "\[http://127.0.0.1:8000/camaras/sync-file\](http://127.0.0.1:8000/camaras/sync-file)" \`  
  \-Method POST \`  
  \-Headers @{ Authorization \= "Bearer $TOKEN" }

Write-Host "Ruta protegida ejecutada con éxito."

### **Credenciales por Defecto**

| Campo | Valor |
| :---- | :---- |
| **Usuario** | admin |
| **Contraseña** | admin |

