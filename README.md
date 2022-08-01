# Getting Started
TODO: Guide users through getting your code up and running on their own system. In this section you can talk about:
1.	Installation process
2.	Software dependencies
3.	Latest releases

# Build 
Para crear el entorno de ejecución:
```
python -m venv .venv
.venv/Script/activate
pip install -r requirements.txt
```

Para lanzar el servidor con docker
```
docker-compose -p machine-learning up
```

Para construir la imagen de la app
```
docker-compose -p machine-learning build
```