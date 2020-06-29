FROM python:3

WORKDIR /usr/src/app

COPY ./requirements.txt requirements.txt

RUN pip install -r requirements.txt

COPY ./app.py app.py

COPY ./models/.gitkeep models/.gitkeep

COPY ./csv/.gitkeep csv/.gitkeep

EXPOSE 5000

RUN export FLASK_APP=app.py

CMD ["flask", "run", "--host=0.0.0.0"]