FROM python:3.10

RUN mkdir /workspace
COPY . /workspace/

WORKDIR /workspace
RUN pip install -r requirements.txt

ENV HOME=/workspace

CMD [ "python" , "app.py" ]
