# docker-flask-sklearn
Basic example of a python flask api for machine learning with Docker, Flask, sklearn and Bokeh for visualization

## Requirements  
Docker

## Docker commands
docker build . -t [name]  
interactive: 	docker run -p 3000:5000 -it [name]  
detached: 		docker run -d -p 3000:5000 [name]  
stop detached:	docker stop [CONTAINER ID]  

## API examples
localhost:3000/  
localhost:3000/chart  
localhost:3000/data  
localhost:3000/predict?AGE=0.1  
localhost:3000/predict?AGE=0.1&RM=6  
