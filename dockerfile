ARG TENSORFLOW_VERSION

#from tensorflow/tensorflow:latest
FROM tensorflow/tensorflow:${TENSORFLOW_VERSION}


RUN apt-get -qq update && \
    apt-get install -y -qq --no-install-recommends --purge \
    poppler-utils && \
    apt-get purge --auto-remove && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    rm -rf /tmp/*

# Create directory for the app user
RUN mkdir -p /home/app

# Create the app user
RUN groupadd app && useradd -g app app

# Create the home directory
ENV HOME=/home/app
ENV APP_HOME=/home/app/machine-learning
RUN mkdir -p ${APP_HOME}/app
WORKDIR $APP_HOME

# Install Requirements
COPY requirements.txt .
COPY .env .env
RUN pip install -r requirements.txt


COPY ./app ${APP_HOME}/app
RUN chown -R app:app $HOME

USER app
#CMD ["python", "./app/app.py"]
