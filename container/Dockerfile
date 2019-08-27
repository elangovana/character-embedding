ARG REGION=us-west-2
ARG DEVICE=cpu
ARG AWS_PYTORCH_SAGEMAKER_ECR_ACCOUNT=520713654638
FROM ${AWS_PYTORCH_SAGEMAKER_ECR_ACCOUNT}.dkr.ecr.$REGION.amazonaws.com/sagemaker-pytorch:1.0.0-${DEVICE}-py3

#NOTE: Do not create the /opt/ml/code directory.. Else downloading the code from s3 fails during runtime...
#ENV sagemaker_submit_directory /opt/ml/code

COPY requirements_dev.txt /tmp

RUN pip install -r /tmp/requirements_dev.txt && rm /tmp/requirements_dev.txt




