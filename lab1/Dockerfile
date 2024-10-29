FROM public.ecr.aws/lambda/python:3.8

ARG bucket_name
ENV BUCKET_NAME $bucket_name

RUN mkdir -p /model
ENV MODEL_PATH /model

WORKDIR ${LAMBDA_TASK_ROOT}

COPY inference/requirements.txt ./
RUN pip install -r requirements.txt

COPY inference/download_model.py ./
RUN python3 download_model.py

COPY inference/app.py ./
CMD [ "app.handler" ]
