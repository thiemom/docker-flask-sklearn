# docker-flask-sklearn
Basic example of a python flask api for machine learning with Docker and sklearn

## requirements  
Docker

## How to run on Docker
docker build . -t [name]
interactive: 	docker run -p 3000:5000 -it [name]
detached: 		docker run -p -d 3000:5000 [name]

## API example
localhost:3000/
localhost:3000/predict?x=10
