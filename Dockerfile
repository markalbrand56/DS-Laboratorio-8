FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Puerto por defecto de Streamlit
EXPOSE 8501

# Ejecutamos la aplicación Streamlit
CMD ["streamlit", "run", "website.py"]
