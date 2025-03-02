    FROM nginx:alpine AS frontend

    RUN rm -rf /usr/share/nginx/html/*
    
    COPY index.html /usr/share/nginx/html/index.html
    COPY styles.css /usr/share/nginx/html/styles.css
    COPY script.js /usr/share/nginx/html/script.js
    
    EXPOSE 80
    
    CMD ["nginx", "-g", "daemon off;"]
    
