# Microsoft Entra ID App Registration Configuration

## Issue: Missing Email and Groups Claims

After implementing Entra ID authentication, you may find that the userinfo endpoint does not return `email` or `groups` claims. This is because Microsoft Entra ID requires explicit configuration to include these claims.

## Current Symptoms

From the auth server logs:
```
Raw user info from entra: {'sub': '...', 'name': 'Debbie Philips', 'family_name': 'Philips', 'given_name': 'Debbie', 'picture': '...'}
Mapped user info: {'username': None, 'email': None, 'name': 'Debbie Philips', 'groups': []}
```

**Missing:**
- `email` claim
- `preferred_username` claim
- `groups` claim

## Solution: Configure App Registration

### Step 1: Add API Permissions for Groups

1. Go to [Azure Portal](https://portal.azure.com)
2. Navigate to **Azure Active Directory** → **App registrations**
3. Select your app: `mcp-gateway-web` (Client ID: `150f50ad-d0ca-4d7a-bb4b-75fbaf31acc1`)
4. Click **API permissions** in the left menu
5. Click **Add a permission**
6. Select **Microsoft Graph**
7. Select **Delegated permissions**
8. Search for and add:
   - `GroupMember.Read.All` - Read groups user belongs to
   - `User.Read` - Should already be present
   - `email` - Read user's email address
   - `profile` - Read user's basic profile
9. Click **Add permissions**
10. Click **Grant admin consent for [Your Tenant]** - This is **REQUIRED**

### Step 2: Configure Optional Claims

1. In your app registration, click **Token configuration** in the left menu
2. Click **Add optional claim**
3. Select **ID** token type
4. Add these claims:
   - `email` - User's email address
   - `preferred_username` - User's UPN (User Principal Name)
   - `groups` - Security group Object IDs
5. Click **Add**
6. When prompted "Turn on the Microsoft Graph email, profile permission", click **Add**

### Step 3: Configure Group Claims

1. Still in **Token configuration**
2. Click **Add groups claim**
3. Select **Security groups**
4. Under "Customize token properties by type", select:
   - **ID**: Check "Group ID"
   - **Access**: Check "Group ID"
5. Click **Add**

### Step 4: Verify Manifest (Optional)

You can verify the configuration in the app manifest:

1. Click **Manifest** in the left menu
2. Look for `optionalClaims`:

```json
"optionalClaims": {
  "idToken": [
    {
      "name": "email",
      "source": null,
      "essential": false,
      "additionalProperties": []
    },
    {
      "name": "preferred_username",
      "source": null,
      "essential": false,
      "additionalProperties": []
    },
    {
      "name": "groups",
      "source": null,
      "essential": false,
      "additionalProperties": []
    }
  ],
  "accessToken": [
    {
      "name": "groups",
      "source": null,
      "essential": false,
      "additionalProperties": []
    }
  ]
}
```

3. Look for `groupMembershipClaims`:
```json
"groupMembershipClaims": "SecurityGroup"
```

### Step 5: Alternative - Use Microsoft Graph API

If you cannot enable `GroupMember.Read.All` permission, you can modify the code to fetch groups via Microsoft Graph API:

```python
# In auth_server/providers/entra.py, add a method:
def get_user_groups(self, access_token: str) -> List[str]:
    """Fetch user's group memberships from Microsoft Graph."""
    try:
        headers = {'Authorization': f'Bearer {access_token}'}
        # Request group Object IDs
        response = requests.get(
            'https://graph.microsoft.com/v1.0/me/memberOf?$select=id,displayName',
            headers=headers,
            timeout=10
        )
        response.raise_for_status()

        data = response.json()
        # Return list of group Object IDs
        return [group['id'] for group in data.get('value', [])]
    except Exception as e:
        logger.error(f"Failed to fetch user groups: {e}")
        return []
```

Then call this in the callback handler instead of relying on the groups claim.

## Testing the Configuration

After making these changes:

1. Wait 5-10 minutes for Azure AD to propagate the changes
2. Clear your browser cookies for `localhost`
3. Try logging in again
4. Check the auth server logs for:
```
Raw user info from entra: {'sub': '...', 'email': 'DebbiePhilips@AWS139.onmicrosoft.com', 'preferred_username': 'DebbiePhilips@AWS139.onmicrosoft.com', 'groups': ['16c7e67e-...', '62c07ac1-...'], ...}
```

## Expected Result

After configuration, you should see:
- `email`: `DebbiePhilips@AWS139.onmicrosoft.com`
- `preferred_username`: `DebbiePhilips@AWS139.onmicrosoft.com`
- `groups`: `['16c7e67e-e8ae-498c-ba2e-0593c0159e43', '62c07ac1-03d0-4924-90c7-a0255f23bd1d']`

The user will be mapped to scopes based on their group membership in `scopes.yml`:
- Admin group (`16c7e67e-...`): `mcp-registry-admin`, `mcp-servers-unrestricted/read`, `mcp-servers-unrestricted/execute`
- Users group (`62c07ac1-...`): `mcp-registry-user`, `mcp-servers-restricted/read`

## References

- [Microsoft Entra ID optional claims](https://learn.microsoft.com/en-us/entra/identity-platform/optional-claims)
- [Configure group claims](https://learn.microsoft.com/en-us/entra/identity-platform/optional-claims#configure-groups-optional-claims)
- [Microsoft Graph permissions](https://learn.microsoft.com/en-us/graph/permissions-reference)
