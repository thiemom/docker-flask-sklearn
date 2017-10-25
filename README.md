# docker-flask-sklearn
Basic example of a python flask api for machine learning with Docker, Flask, sklearn and Bokeh for visualization

## requirements  
Docker

## How to run on Docker
docker build . -t [name]
interactive: 	docker run -p 3000:5000 -it [name]
detached: 		docker run -p -d 3000:5000 [name]

## API examples
localhost:3000/
localhost:3000/chart
localhost:3000/predict?x=0.1
