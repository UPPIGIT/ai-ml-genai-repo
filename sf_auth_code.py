import requests
import json

def get_salesforce_token(client_id, client_secret, domain='login'):
    """
    Get Salesforce access token using client credentials (OAuth 2.0).
    
    Args:
        client_id: Connected App Consumer Key
        client_secret: Connected App Consumer Secret
        domain: 'login' for production/developer, 'test' for sandbox
    
    Returns:
        dict: Authentication response with access_token, instance_url, etc.
    """
    url = f"https://{domain}.salesforce.com/services/oauth2/token"
    
    payload = {
        'grant_type': 'client_credentials',
        'client_id': client_id,
        'client_secret': client_secret
    }
    
    headers = {
        'Content-Type': 'application/x-www-form-urlencoded'
    }
    
    try:
        response = requests.post(url, data=payload, headers=headers)
        response.raise_for_status()
        
        auth_data = response.json()
        
        print("✓ Authentication successful!")
        print(f"Access Token: {auth_data['access_token'][:30]}...")
        print(f"Instance URL: {auth_data['instance_url']}")
        print(f"Token Type: {auth_data.get('token_type', 'Bearer')}")
        
        return auth_data
    
    except requests.exceptions.HTTPError as e:
        print(f"✗ Authentication failed: {e}")
        print(f"Response: {response.text}")
        return None
    except Exception as e:
        print(f"✗ Error: {e}")
        return None


def make_salesforce_request(access_token, instance_url, endpoint, method='GET', data=None):
    """
    Make an authenticated request to Salesforce API.
    
    Args:
        access_token: OAuth access token
        instance_url: Salesforce instance URL
        endpoint: API endpoint (e.g., '/services/data/v59.0/query')
        method: HTTP method (GET, POST, PATCH, DELETE)
        data: Request payload for POST/PATCH
    
    Returns:
        dict: API response
    """
    url = f"{instance_url}{endpoint}"
    
    headers = {
        'Authorization': f'Bearer {access_token}',
        'Content-Type': 'application/json'
    }
    
    try:
        if method.upper() == 'GET':
            response = requests.get(url, headers=headers)
        elif method.upper() == 'POST':
            response = requests.post(url, headers=headers, json=data)
        elif method.upper() == 'PATCH':
            response = requests.patch(url, headers=headers, json=data)
        elif method.upper() == 'DELETE':
            response = requests.delete(url, headers=headers)
        else:
            raise ValueError(f"Unsupported HTTP method: {method}")
        
        response.raise_for_status()
        return response.json() if response.content else {}
    
    except requests.exceptions.HTTPError as e:
        print(f"✗ API request failed: {e}")
        print(f"Response: {response.text}")
        return None
    except Exception as e:
        print(f"✗ Error: {e}")
        return None


def query_salesforce(access_token, instance_url, soql_query):
    """
    Execute a SOQL query.
    
    Args:
        access_token: OAuth access token
        instance_url: Salesforce instance URL
        soql_query: SOQL query string
    
    Returns:
        dict: Query results
    """
    endpoint = f"/services/data/v59.0/query?q={requests.utils.quote(soql_query)}"
    return make_salesforce_request(access_token, instance_url, endpoint, method='GET')


# Example usage
if __name__ == "__main__":
    # Configuration
    CLIENT_ID = 'your_consumer_key_here'
    CLIENT_SECRET = 'your_consumer_secret_here'
    DOMAIN = 'login'  # Use 'test' for sandbox
    
    # Get access token
    auth = get_salesforce_token(CLIENT_ID, CLIENT_SECRET, DOMAIN)
    
    if auth:
        access_token = auth['access_token']
        instance_url = auth['instance_url']
        
        # Example 1: Query accounts
        print("\n--- Querying Accounts ---")
        results = query_salesforce(
            access_token, 
            instance_url, 
            'SELECT Id, Name FROM Account LIMIT 5'
        )
        
        if results:
            print(f"Total records: {results.get('totalSize', 0)}")
            for record in results.get('records', []):
                print(f"  - {record['Name']} (ID: {record['Id']})")
        
        # Example 2: Get API versions
        print("\n--- Available API Versions ---")
        versions = make_salesforce_request(
            access_token,
            instance_url,
            '/services/data/',
            method='GET'
        )
        
        if versions:
            print(f"Latest version: {versions[-1]['version']}")
        
        # Example 3: Get object metadata
        print("\n--- Account Object Metadata ---")
        metadata = make_salesforce_request(
            access_token,
            instance_url,
            '/services/data/v59.0/sobjects/Account/describe',
            method='GET'
        )
        
        if metadata:
            print(f"Object: {metadata['name']}")
            print(f"Label: {metadata['label']}")
            print(f"Fields: {len(metadata['fields'])}")
