# ─────────────────────────────────────────────────────────
#  Dockerfile
#  Proyecto: Predicción de Abandono de Clientes (Churn)
#  Modo: Scripts Python (.py)
# ─────────────────────────────────────────────────────────

# Imagen base oficial de Python (slim = más liviana)
FROM python:3.10-slim

# Evitar prompts interactivos durante la instalación
ENV DEBIAN_FRONTEND=noninteractive

# Directorio de trabajo dentro del contenedor
WORKDIR /app

# ── Instalar dependencias del sistema ────────────────────
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    git \
    curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# ── Copiar e instalar dependencias de Python ─────────────
# Se copia primero solo el requirements para aprovechar
# la caché de Docker (si el código cambia pero no las libs,
# no reinstala todo)
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ── Copiar el resto del proyecto ─────────────────────────
COPY . .

# ── Crear carpetas de salida si no existen ───────────────
RUN mkdir -p outputs/figuras outputs/modelos

# ── Variable de entorno para que matplotlib no use GUI ───
# Necesario en Docker porque no hay pantalla (modo headless)
# Los gráficos se guardan como .png en outputs/figuras/
ENV MPLBACKEND=Agg
 
# ── Comando por defecto: bash interactivo ────────────────
# Permite correr cualquier script manualmente desde la terminal
# Uso: docker compose run churn python src/exploration/explore.py
CMD ["bash"]