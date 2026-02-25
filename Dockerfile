FROM python:3.9

# Configurar un usuario sin permisos root (Requisito estricto de Hugging Face)
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"

WORKDIR /app

# 1. Copiar tu archivo de dependencias (notar que usamos requirements_api.txt)
COPY --chown=user ./requirements_api.txt requirements.txt
RUN pip install --no-cache-dir --upgrade -r requirements.txt

# 2. Copiar el resto del código (incluyendo api.py y la carpeta artifacts)
COPY --chown=user . /app

# 3. Arrancar uvicorn apuntando a 'api' en lugar de 'app' 
# (Asumiendo que dentro de api.py tienes una variable 'app = FastAPI()')
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "7860"]