FROM python:3.10 AS backend

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]

# ----------- Frontend --------------
    FROM nginx:alpine AS frontend

    RUN rm -rf /usr/share/nginx/html/*
    
    COPY index.html /usr/share/nginx/html/index.html
    
    EXPOSE 80
    
    CMD ["nginx", "-g", "daemon off;"]
    