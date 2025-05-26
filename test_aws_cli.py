#!/usr/bin/env python3
import subprocess
import json
import os
import sys

def test_aws_cli():
    """Test AWS CLI configuration and SageMaker endpoint access."""
    print("Testing AWS CLI configuration...")
    
    # 1. Check AWS CLI version
    try:
        version = subprocess.run(['aws', '--version'], capture_output=True, text=True, check=True)
        print(f"AWS CLI version: {version.stdout.strip()}")
    except Exception as e:
        print(f"Error checking AWS CLI version: {e}")
        return False

    # 2. Check AWS credentials
    try:
        identity = subprocess.run(
            ['aws', 'sts', 'get-caller-identity'],
            capture_output=True,
            text=True,
            check=True
        )
        print("\nAWS Identity:")
        print(json.dumps(json.loads(identity.stdout), indent=2))
    except Exception as e:
        print(f"Error checking AWS identity: {e}")
        return False

    # 3. Test SageMaker endpoint
    endpoint_name = os.environ.get('SAGEMAKER_ENDPOINT_NAME')
    if not endpoint_name:
        print("Error: SAGEMAKER_ENDPOINT_NAME not set")
        return False

    print(f"\nTesting SageMaker endpoint: {endpoint_name}")
    
    # Create a simple test payload
    test_payload = {
        "inputs": "Hello, this is a test message. Please respond with a short greeting."
    }
    
    # Save payload to temporary file
    with open('test_payload.json', 'w') as f:
        json.dump(test_payload, f)
    
    try:
        # Try the sagemaker-runtime service first
        print("\nTrying sagemaker-runtime service...")
        cmd = [
            'aws', 'sagemaker-runtime', 'invoke-endpoint',
            '--endpoint-name', endpoint_name,
            '--content-type', 'application/json',
            '--body', 'file://test_payload.json',
            '--cli-binary-format', 'raw-in-base64-out'
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        print(f"Exit code: {result.returncode}")
        print(f"STDOUT: {result.stdout}")
        print(f"STDERR: {result.stderr}")
        
        if result.returncode == 0:
            print("\nSuccess with sagemaker-runtime!")
            return True
            
        # If sagemaker-runtime fails, try the sagemaker service
        print("\nTrying sagemaker service...")
        cmd = [
            'aws', 'sagemaker', 'invoke-endpoint',
            '--endpoint-name', endpoint_name,
            '--content-type', 'application/json',
            '--body', 'file://test_payload.json',
            '--cli-binary-format', 'raw-in-base64-out'
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        print(f"Exit code: {result.returncode}")
        print(f"STDOUT: {result.stdout}")
        print(f"STDERR: {result.stderr}")
        
        return result.returncode == 0
        
    except Exception as e:
        print(f"Error testing SageMaker endpoint: {e}")
        return False
    finally:
        if os.path.exists('test_payload.json'):
            os.unlink('test_payload.json')

if __name__ == "__main__":
    success = test_aws_cli()
    sys.exit(0 if success else 1) 