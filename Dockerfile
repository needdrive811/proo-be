FROM continuumio/miniconda3

RUN conda install scikit-learn

# Grab requirements.txt.
ADD ./requirements.txt /tmp/requirements.txt

# Install dependencies
RUN pip install -qr /tmp/requirements.txt

ARG CACHE_DATE=not_a_date

# Add our code
ADD ./webapp /opt/webapp/
WORKDIR /opt/webapp

CMD waitress-serve --listen=*:$PORT wsgi:application

#CMD waitress-serve --listen=*:5000 wsgi:application