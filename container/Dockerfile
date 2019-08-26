ARG REGION=us-west-2
ARG DEVICE=gpu
ARG AWS_PYTORCH_SAGEMAKER_ECR_ACCOUNT=520713654638
FROM ${AWS_PYTORCH_SAGEMAKER_ECR_ACCOUNT}.dkr.ecr.$REGION.amazonaws.com/sagemaker-pytorch:1.0.0-${DEVICE}-py3

ENV path="/opt/ml/code:${path}"

#ENV sagemaker_submit_directory /opt/ml/code

COPY ./src/requirements_dev.txt /opt/ml/code/

RUN pip install -r /opt/ml/code/requirements_dev.txt


