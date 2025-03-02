# ----------- Frontend (Nginx) --------------
    FROM nginx:alpine AS frontend

    # Supprimer la configuration par défaut de Nginx
    RUN rm -rf /usr/share/nginx/html/*
    
    # Copier les fichiers du frontend
    COPY index.html /usr/share/nginx/html/index.html
    COPY styles.css /usr/share/nginx/html/styles.css
    COPY script.js /usr/share/nginx/html/script.js
    
    # Exposer le port du frontend
    EXPOSE 80
    
    # Lancer Nginx
    CMD ["nginx", "-g", "daemon off;"]
    