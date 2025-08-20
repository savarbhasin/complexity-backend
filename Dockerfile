# Use AWS Lambda Python runtime as the base image
FROM public.ecr.aws/lambda/python:3.10

# Install system dependencies
RUN yum update -y && \
    yum install -y gcc g++ make && \
    yum clean all

# Copy requirements file
COPY requirements.txt ${LAMBDA_TASK_ROOT}

# Install Python dependencies
# Use --no-cache-dir to reduce image size
# Use --no-deps for pydantic-core to avoid conflicts
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . ${LAMBDA_TASK_ROOT}

# Set the CMD to your handler (this will be overridden by Lambda)
CMD ["lambda_handler.handler"]
