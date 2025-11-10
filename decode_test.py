#!/usr/bin/env python3

import requests
import os
import base64

# Get credentials from environment
creds = os.getenv("ASOR_CLIENT_CREDENTIALS", "")
if ":" in creds:
    client_id, client_secret = creds.split(":", 1)
else:
    print("Invalid credentials format")
    exit(1)

print(f"Original Client ID: {client_id}")
print(f"Original Client Secret: {client_secret}")

# Try to decode the client_id if it looks like base64
try:
    decoded_client_id = base64.b64decode(client_id).decode('utf-8')
    print(f"Decoded Client ID: {decoded_client_id}")
    
    # Test with decoded credentials
    token_url = "https://wcpdev-services1.wd103.myworkday.com/ccx/oauth2/awsasor_wcpdev1/token"
    
    print("\n=== Testing with decoded client ID ===")
    data = {
        "grant_type": "client_credentials",
        "client_id": decoded_client_id,
        "client_secret": client_secret
    }

    headers = {
        "Content-Type": "application/x-www-form-urlencoded"
    }

    response = requests.post(token_url, data=data, headers=headers)
    print(f"Status: {response.status_code}")
    print(f"Response: {response.text}")
    
except Exception as e:
    print(f"Could not decode client_id or request failed: {e}")
