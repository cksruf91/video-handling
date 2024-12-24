import sys

import boto3
import botocore


def handler(event, context):
    print(f'boto3 version: {boto3.__version__}')
    print(f'botocore version: {botocore.__version__}')
    print(f'event data: {event}')
    print(f'context data: {context}')
    return 'Hello from AWS Lambda using Python' + sys.version + '!'
