FROM public.ecr.aws/lambda/python:3.12.2024.11.19.19

# 시간 동기화
ENV TZ=Asia/Seoul
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone


COPY requirements.txt .
COPY lambda_function.py .
COPY core .
COPY utile .
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Set the CMD to your handler (could also be done as a parameter override outside of the Dockerfile)
CMD [ "lambda_function.handler" ]
