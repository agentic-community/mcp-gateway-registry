#!/usr/bin/env python3
"""
ASOR Authorization Code Generator

This script generates the OAuth authorization URL and extracts the authorization code
from the redirect URL for ASOR federation.
"""

import urllib.parse
import webbrowser
import os
import re

# ASOR OAuth configuration
CLIENT_ID = "ZjgyZGVjMzAtMTY5Zi00Mzc1LThlNWUtYzc5OGU0NDdjMzJi"
REDIRECT_URI = "https://localhost:7860/callback"
SCOPE = "Agent System of Record"
BASE_URL = "https://wcpdev.wd103.myworkday.com/awsasor_wcpdev1"

def generate_auth_url():
    """Generate the OAuth authorization URL."""
    params = {
        "response_type": "code",
        "client_id": CLIENT_ID,
        "redirect_uri": REDIRECT_URI,
        "scope": SCOPE
    }
    
    auth_url = f"{BASE_URL}/authorize?" + urllib.parse.urlencode(params)
    return auth_url

def extract_code_from_url(redirect_url):
    """Extract authorization code from redirect URL."""
    parsed = urllib.parse.urlparse(redirect_url)
    params = urllib.parse.parse_qs(parsed.query)
    return params.get('code', [None])[0]

def update_env_file(auth_code):
    """Update .env file with ASOR_AUTH_CODE and remove old ASOR_ACCESS_TOKEN."""
    env_file = ".env"
    
    # Read existing .env file
    lines = []
    if os.path.exists(env_file):
        with open(env_file, 'r') as f:
            lines = f.readlines()
    
    # Remove old ASOR tokens and add new auth code
    new_lines = []
    auth_code_added = False
    
    for line in lines:
        # Skip old ASOR token lines
        if line.startswith('ASOR_ACCESS_TOKEN=') or line.startswith('ASOR_AUTH_CODE='):
            continue
        new_lines.append(line)
    
    # Add new auth code
    new_lines.append(f"ASOR_AUTH_CODE={auth_code}\n")
    
    # Write back to .env file
    with open(env_file, 'w') as f:
        f.writelines(new_lines)
    
    print(f"✅ Updated {env_file} with ASOR_AUTH_CODE")

def main():
    print("🔑 ASOR Authorization Code Generator")
    print("=" * 50)
    
    # Generate and display auth URL
    auth_url = generate_auth_url()
    print(f"1. Opening authorization URL in browser...")
    print(f"   URL: {auth_url}")
    print()
    
    # Open browser
    try:
        webbrowser.open(auth_url)
        print("✅ Browser opened automatically")
    except:
        print("❌ Could not open browser automatically")
        print("   Please copy and paste the URL above into your browser")
    
    print()
    print("2. After authorizing, you'll be redirected to a URL like:")
    print("   https://localhost:7860/callback?code=AUTHORIZATION_CODE&state=...")
    print()
    
    # Get redirect URL from user
    redirect_url = input("3. Paste the full redirect URL here: ").strip()
    
    # Extract code
    auth_code = extract_code_from_url(redirect_url)
    
    if auth_code:
        print()
        print("✅ Authorization code extracted successfully!")
        print("=" * 50)
        
        # Update .env file
        update_env_file(auth_code)
        print(f"Updated the env file, but run the following command to explicitly set the ASOR_AUTH_CODE")
        print(f"export ASOR_AUTH_CODE='{auth_code}'")
        print()
        print("🔄 Restart the registry to use the new auth code:")
        print("docker-compose restart registry")
        print()
    else:
        print("❌ Could not extract authorization code from URL")
        print("   Make sure you pasted the complete redirect URL")

if __name__ == "__main__":
    main()
