# ----------- Backend (FastAPI) --------------
    FROM python:3.10 AS backend

    WORKDIR /app
    
    # Copier et installer les dépendances
    COPY requirements.txt ./
    RUN pip install --no-cache-dir -r requirements.txt
    
    # Copier tout le code de l'application
    COPY . .
    
    RUN python create_db.py
    # Exécuter le script de création de la base de données (au démarrage)
    CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
    