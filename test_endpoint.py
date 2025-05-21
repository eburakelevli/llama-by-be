import boto3
import json
import os
import sys
import time
from botocore.config import Config
from dotenv import load_dotenv

# Load environment variables from .env file if it exists
load_dotenv()

# Get required configuration from environment variables
ENDPOINT_NAME = os.getenv('SAGEMAKER_ENDPOINT_NAME')
AWS_REGION = os.getenv('AWS_DEFAULT_REGION')

# Validate required environment variables
if not ENDPOINT_NAME:
    print("Error: SAGEMAKER_ENDPOINT_NAME environment variable is required", file=sys.stderr)
    sys.exit(1)
if not AWS_REGION:
    print("Error: AWS_DEFAULT_REGION environment variable is required", file=sys.stderr)
    sys.exit(1)

def check_endpoint_health():
    """Check if the endpoint is healthy and ready to accept requests"""
    sagemaker = boto3.client('sagemaker', region_name=AWS_REGION)
    try:
        response = sagemaker.describe_endpoint(EndpointName=ENDPOINT_NAME)
        status = response['EndpointStatus']
        print(f"\nEndpoint Details:")
        print(f"Endpoint Name: {ENDPOINT_NAME}")
        print(f"Overall Status: {status}")
        print(f"Creation Time: {response.get('CreationTime')}")
        print(f"Last Modified Time: {response.get('LastModifiedTime')}")
        
        print("\nVariant Details:")
        for variant in response.get('ProductionVariants', []):
            print(f"\nVariant Name: {variant.get('VariantName')}")
            print(f"Status: {variant.get('CurrentVariantStatus')}")
            print(f"Instance Type: {variant.get('InstanceType')}")
            print(f"Initial Instance Count: {variant.get('InitialInstanceCount')}")
            print(f"Current Instance Count: {variant.get('CurrentInstanceCount')}")
            print(f"Desired Instance Count: {variant.get('DesiredInstanceCount')}")
            
            if 'VariantStatus' in variant:
                print("Variant Status Details:")
                for status_detail in variant['VariantStatus']:
                    print(f"  - {status_detail.get('Status')}: {status_detail.get('StatusMessage')}")
        
        if status != 'InService':
            print(f"\nWarning: Endpoint is not InService (current status: {status})")
            return False
            
        # Check variant status
        all_variants_healthy = True
        for variant in response.get('ProductionVariants', []):
            variant_status = variant.get('CurrentVariantStatus')
            if variant_status != 'InService':
                print(f"\nWarning: Variant {variant.get('VariantName')} is not InService (current status: {variant_status})")
                all_variants_healthy = False
                
        return all_variants_healthy
    except Exception as e:
        print(f"Error checking endpoint health: {str(e)}")
        return False

def test_endpoint():
    # First check endpoint health
    print("Checking endpoint health...")
    if not check_endpoint_health():
        print("Endpoint is not healthy. Please check the SageMaker console for details.")
        return

    # Create a SageMaker runtime client with longer timeout settings
    config = Config(
        connect_timeout=60,    # 60 seconds to establish connection
        read_timeout=300,      # 5 minutes to read response
        retries={'max_attempts': 1}  # Don't retry, fail fast
    )
    
    runtime = boto3.client('sagemaker-runtime',
                          region_name=AWS_REGION,
                          config=config)
    
    # Test payload
    payload = {
        "inputs": "test"
    }
    
    print(f"\nTesting SageMaker endpoint: {ENDPOINT_NAME}")
    print(f"AWS Region: {AWS_REGION}")
    print("Sending test request to endpoint...")
    
    try:
        start_time = time.time()
        # Invoke the endpoint
        response = runtime.invoke_endpoint(
            EndpointName=ENDPOINT_NAME,
            ContentType='application/json',
            Body=json.dumps(payload)
        )
        
        # Parse and print the response
        response_body = response['Body'].read().decode()
        end_time = time.time()
        print(f"\nRequest completed in {end_time - start_time:.2f} seconds")
        print("Response:", response_body)
        
    except Exception as e:
        print("\nError occurred!")
        print("Error type:", type(e).__name__)
        print("Error message:", str(e))
        
        if isinstance(e, boto3.exceptions.SageMakerRuntime.ClientError):
            print("\nError Response Details:")
            print("Error Code:", e.response['Error']['Code'])
            print("Error Message:", e.response['Error']['Message'])
        elif hasattr(e, 'response'):
            print("\nError Response Details:")
            print("Status Code:", e.response.get('ResponseMetadata', {}).get('HTTPStatusCode'))
            print("Error Code:", e.response.get('Error', {}).get('Code'))
            print("Error Message:", e.response.get('Error', {}).get('Message'))
            
            if 'Error' in e.response:
                print("\nFull Error Response:")
                print(json.dumps(e.response['Error'], indent=2))
        else:
            print("\nNo additional error details available")

if __name__ == "__main__":
    test_endpoint() 