#!/usr/bin/env python3
"""
Script to fix auth test patterns in test_server_routes.py

Removes manual patching and ensures proper use of auth_override_helper.
"""

import re


def fix_test_file():
    file_path = "/home/ubuntu/mcp-gateway-registry-MAIN/tests/unit/api/test_server_routes.py"

    with open(file_path, "r") as f:
        content = f.read()

    # Pattern 1: Remove "with patch" blocks for admin users (single line)
    # Match: with patch("registry.api.server_routes.enhanced_auth", return_value=admin_user_context):
    pattern1 = r'        with patch\("registry\.api\.server_routes\.(enhanced_auth|nginx_proxied_auth)", return_value=admin_user_context\):\n'
    content = re.sub(pattern1, "", content)

    # Pattern 2: Remove multiline patch blocks with user_has_ui_permission_for_service (admin)
    pattern2 = r'        with patch\("registry\.api\.server_routes\.(enhanced_auth|nginx_proxied_auth)", return_value=admin_user_context\), \\\n             patch\("registry\.api\.server_routes\.user_has_ui_permission_for_service", return_value=True\):\n'
    content = re.sub(pattern2, "", content)

    # Pattern 3: Handle tests with regular_user_context - add auth_override_helper and call it
    # First, find regular_user_context tests and add auth_override_helper param
    # Pattern: def test_xxx(self, ..., regular_user_context)
    # Need to add auth_override_helper after regular_user_context if not present

    def add_auth_helper_param(match):
        func_sig = match.group(0)
        # Check if auth_override_helper already in signature
        if "auth_override_helper" in func_sig:
            return func_sig
        # Add auth_override_helper after regular_user_context
        return func_sig.replace(
            "regular_user_context\n", "regular_user_context,\n        auth_override_helper\n"
        )

    pattern_func = r"    def test_\w+\([^)]+regular_user_context\n    \):"
    content = re.sub(pattern_func, add_auth_helper_param, content, flags=re.MULTILINE)

    # Pattern 4: For regular_user_context tests, replace patch blocks with auth_override_helper call
    # Match: with patch(...enhanced_auth...regular_user_context), \
    #             patch(...user_has_ui_permission...):
    #            # Act
    # Replace with: # Arrange - override auth to regular user
    #               auth_override_helper(regular_user_context)
    #               # Act

    pattern4 = r'        with patch\("registry\.api\.server_routes\.(enhanced_auth|nginx_proxied_auth)", return_value=regular_user_context\), \\\n             patch\("registry\.api\.server_routes\.user_has_ui_permission_for_service", return_value=(True|False)\):\n            # Act'

    def replace_regular_auth(match):
        permission_val = match.group(3)
        if permission_val == "True":
            with_patch = 'with patch("registry.api.server_routes.user_has_ui_permission_for_service", return_value=True):\n'
            indent = "            "
        else:
            with_patch = 'with patch("registry.api.server_routes.user_has_ui_permission_for_service", return_value=False):\n'
            indent = "            "

        return f"        # Arrange - override auth to regular user\n        auth_override_helper(regular_user_context)\n        {with_patch}{indent}# Act"

    content = re.sub(pattern4, replace_regular_auth, content)

    # Fix remaining indentation issues
    # Lines that were indented for "with patch" context should be de-indented
    lines = content.split("\n")
    fixed_lines = []
    in_test_method = False
    skip_dedent = False

    for i, line in enumerate(lines):
        # Track if we're in a test method
        if line.strip().startswith("def test_"):
            in_test_method = True
            skip_dedent = False
        elif line.strip().startswith("def ") or (
            line.strip().startswith("class ") and not line.strip().startswith("class Test")
        ):
            in_test_method = False

        # Check if this is a comment we added
        if "# Arrange - override auth" in line or "# Arrange - auth already set" in line:
            skip_dedent = True
        elif line.strip().startswith("# Act"):
            skip_dedent = True
        elif line.strip() == "" or line.strip().startswith("#"):
            pass  # Keep as is
        elif skip_dedent and in_test_method and line.startswith("            "):
            # De-indent by 4 spaces (was indented for with block)
            line = line[4:]

        fixed_lines.append(line)

    content = "\n".join(fixed_lines)

    with open(file_path, "w") as f:
        f.write(content)

    print("Fixed test file")


if __name__ == "__main__":
    fix_test_file()
