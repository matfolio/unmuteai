import json
import io
import os
import boto3
import csv
import numpy as np

# Replacing this section with the created endpoint.
ENDPOINT_NAME = 'sagemaker-scikit-learn-******************'
client = boto3.client('runtime.sagemaker')
def lambda_handler(event, context):
    print("Received event: " + json.dumps(event, indent=2))
    
    data = json.loads(json.dumps(event))
    payload = data['data']   
    #ContentType='application/json',
    #payload = np.array(payload, dtype='S')
    response = client.invoke_endpoint(EndpointName=ENDPOINT_NAME,ContentType='application/json', Body=payload)
    result = json.loads(response['Body'].read().decode())
    print(result)
    return result
