FROM python:3

WORKDIR /app/

COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

COPY /app/. /app/

EXPOSE 5000

ENTRYPOINT python ./app.py
