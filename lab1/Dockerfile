FROM public.ecr.aws/lambda/python:3.12

WORKDIR /var/task

RUN mkdir -p /model
ENV MODEL_PATH /var/task/model

COPY inference/* .
RUN pip install -r requirements.txt
RUN python download_model.py

CMD ["app.lambda_handler"]
