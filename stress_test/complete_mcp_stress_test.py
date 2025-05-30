#!/usr/bin/env python3
"""
Complete MCP Gateway Registry Stress Test Suite
Single file that does everything: registers mock servers, tests FAISS, tests UI, generates reports.
"""

import json
import time
import asyncio
import aiohttp
import psutil
import threading
import argparse
import os
import sys
from pathlib import Path
from typing import List, Dict, Any, Tuple
from datetime import datetime
import numpy as np
import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.keys import Keys
from selenium.common.exceptions import TimeoutException, NoSuchElementException

# MCP imports
from mcp import ClientSession
from mcp.client.sse import sse_client

# Configuration
REGISTRY_URL = os.getenv("REGISTRY_URL", "http://localhost:7860")
MCPGW_SERVER_URL = os.getenv("MCPGW_SERVER_URL", "http://ec2-54-89-253-127.compute-1.amazonaws.com/mcpgw/sse")
ADMIN_USER = os.getenv("ADMIN_USER", "admin")
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "your-admin-password")

# Test queries for FAISS - Updated to match actual available tools
STRESS_QUERIES = [
    "stock market data and financial information",
    "get stock prices and trading data",
    "polygon API for financial aggregates", 
    "current time in different timezones",
    "timezone information and time API",
    "quantum flux analysis and energy levels",
    "neural pattern synthesis and AI processing",
    "hyper dimensional mapping coordinates",
    "temporal anomaly detection in time series",
    "user profile analysis and behavior",
    "synthetic data generation and schemas",
    "MCP service registration and management",
    "toggle and control MCP servers",
    "server configuration and details",
    "financial stock ticker symbols and AAPL data",
    "time zone America/New_York current time"
]

class SystemMonitor:
    def __init__(self):
        self.monitoring = False
        self.measurements = []
        self.thread = None
        
    def start(self):
        self.monitoring = True
        self.measurements = []
        self.thread = threading.Thread(target=self._monitor)
        self.thread.daemon = True
        self.thread.start()
        
    def stop(self):
        self.monitoring = False
        if self.thread:
            self.thread.join(timeout=2.0)
        return self.measurements.copy()
        
    def _monitor(self):
        while self.monitoring:
            try:
                process = psutil.Process()
                cpu = psutil.cpu_percent(interval=None)
                memory = psutil.virtual_memory()
                
                self.measurements.append({
                    "timestamp": time.time(),
                    "cpu_percent": cpu,
                    "memory_percent": memory.percent,
                    "memory_used_gb": memory.used / (1024**3),
                    "process_memory_mb": process.memory_info().rss / (1024**2),
                    "process_cpu": process.cpu_percent(),
                    "threads": process.num_threads()
                })
            except:
                pass
            time.sleep(0.5)

class CompleteMCPStressTest:
    def __init__(self, output_base_dir: Path = None):
        # Create a unique folder for this test run in the same directory as the script
        if output_base_dir is None:
            # Get the directory where this script is located
            script_dir = Path(__file__).parent
            output_base_dir = script_dir
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.output_dir = output_base_dir / f"results-{timestamp}"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Test results will be saved to: {self.output_dir}")
        
        self.results = {
            "metadata": {
                "test_start": datetime.now().isoformat(),
                "registry_url": REGISTRY_URL,
                "mcpgw_url": MCPGW_SERVER_URL,
                "admin_user": ADMIN_USER,
                "output_directory": str(self.output_dir)
            },
            "mock_servers": {},
            "faiss_results": [],
            "ui_results": {},
            "ui_search_results": [],
            "performance_summary": {}
        }
        
    def generate_mock_server_data(self, num_servers: int = 50) -> List[Dict[str, Any]]:
        """Generate mock server definitions."""
        print(f"Generating {num_servers} mock server definitions...")
        
        server_types = ["database", "analytics", "ml", "security", "monitoring", "api", "workflow"]
        domains = ["finance", "healthcare", "retail", "logistics", "manufacturing", "education"]
        
        servers = []
        for i in range(1, num_servers + 1):
            server_type = server_types[i % len(server_types)]
            domain = domains[i % len(domains)]
            
            # Generate realistic tool sets
            base_tools = 5 + (i % 15)  # 5-15 tools per server
            tools = []
            
            for j in range(base_tools):
                tool_name = f"{server_type}_{domain}_tool_{j+1}"
                tools.append({
                    "name": tool_name,
                    "description": f"Tool for {server_type} operations in {domain} domain",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "input": {"type": "string", "description": "Input data"},
                            "config": {"type": "object", "description": "Configuration options"}
                        }
                    }
                })
            
            server = {
                "server_name": f"Mock {server_type.title()} Server {i:03d}",
                "path": f"/mock-{server_type}-{i:03d}",
                "proxy_pass_url": f"http://mock-{server_type}-{i:03d}:8000",
                "description": f"Mock {server_type} server for {domain} operations",
                "tags": [server_type, domain, "mock", "stress-test"],
                "num_tools": len(tools),
                "num_stars": (i % 5) + 1,
                "is_python": i % 3 == 0,
                "license": "MIT",
                "tools": tools
            }
            servers.append(server)
            
        print(f"Generated {len(servers)} mock servers with {sum(s['num_tools'] for s in servers)} total tools")
        return servers
    
    async def register_mock_servers(self, servers: List[Dict[str, Any]]) -> int:
        """Register mock servers via API and ensure they are enabled."""
        print(f"📝 Registering {len(servers)} mock servers...")
        
        successful = 0
        
        async with aiohttp.ClientSession() as session:
            # Login first
            data = aiohttp.FormData()
            data.add_field('username', ADMIN_USER)
            data.add_field('password', ADMIN_PASSWORD)
            
            async with session.post(f"{REGISTRY_URL}/login", data=data) as response:
                if response.status not in [200, 201, 302, 303]:
                    raise Exception(f"Login failed: {response.status}")
            
            # Register servers in batches
            batch_size = 10
            for i in range(0, len(servers), batch_size):
                batch = servers[i:i + batch_size]
                print(f"  Registering batch {i//batch_size + 1}/{(len(servers) + batch_size - 1)//batch_size}...")
                
                tasks = []
                for server in batch:
                    tasks.append(self._register_single_server(session, server))
                
                results = await asyncio.gather(*tasks, return_exceptions=True)
                successful += sum(1 for r in results if r is True)
                
                await asyncio.sleep(0.1)  # Small delay between batches
        
        print(f"Successfully registered {successful}/{len(servers)} servers")
        
        # Now enable all registered servers
        if successful > 0:
            print("🔄 Enabling all registered mock servers...")
            enabled_count = await self._enable_all_mock_servers()
            print(f"Enabled {enabled_count} mock servers")
        
        self.results["mock_servers"] = {"total": len(servers), "successful": successful, "enabled": enabled_count if successful > 0 else 0}
        return successful
    
    async def _register_single_server(self, session: aiohttp.ClientSession, server: Dict[str, Any]) -> bool:
        """Register a single server."""
        try:
            data = aiohttp.FormData()
            data.add_field('name', server["server_name"])
            data.add_field('path', server["path"])
            data.add_field('proxy_pass_url', server["proxy_pass_url"])
            data.add_field('description', server["description"])
            data.add_field('tags', ",".join(server["tags"]))
            data.add_field('num_tools', str(server["num_tools"]))
            data.add_field('num_stars', str(server["num_stars"]))
            data.add_field('is_python', str(server["is_python"]).lower())
            data.add_field('license_str', server["license"])
            
            async with session.post(f"{REGISTRY_URL}/register", data=data) as response:
                return response.status in [200, 201]
        except:
            return False
    
    async def _enable_all_mock_servers(self) -> int:
        """Enable all mock servers that were just registered."""
        # Find all mock server files that were created
        mock_server_dir = Path("/home/ubuntu/mcp-gateway-data/servers/")
        mock_files = list(mock_server_dir.glob("mock-*.json"))  # Updated pattern
        
        print(f"  Found {len(mock_files)} mock server files to enable")
        
        enabled_count = 0
        
        # Enable servers in batches using the MCP registry API
        async with aiohttp.ClientSession() as session:
            # Login first
            data = aiohttp.FormData()
            data.add_field('username', ADMIN_USER)
            data.add_field('password', ADMIN_PASSWORD)
            
            async with session.post(f"{REGISTRY_URL}/login", data=data) as response:
                if response.status not in [200, 201, 302, 303]:
                    print("Could not login to enable servers")
                    return 0
            
            # Process in batches
            batch_size = 25  # Smaller batches for reliability
            for i in range(0, len(mock_files), batch_size):
                batch = mock_files[i:i + batch_size]
                print(f"  Enabling batch {i//batch_size + 1}/{(len(mock_files) + batch_size - 1)//batch_size}...")
                
                tasks = []
                for mock_file in batch:
                    try:
                        # Read server data to get the path
                        with open(mock_file, 'r') as f:
                            server_data = json.load(f)
                        server_path = server_data.get("path", "")
                        if server_path:
                            tasks.append(self._enable_single_server(session, server_path))
                    except Exception as e:
                        print(f"Could not read {mock_file}: {e}")
                        continue
                
                if tasks:
                    results = await asyncio.gather(*tasks, return_exceptions=True)
                    batch_enabled = sum(1 for r in results if r is True)
                    enabled_count += batch_enabled
                    print(f"    Enabled {batch_enabled}/{len(batch)} servers in this batch")
                
                await asyncio.sleep(0.2)  # Slightly longer delay between batches
        
        return enabled_count
    
    async def _enable_single_server(self, session: aiohttp.ClientSession, server_path: str) -> bool:
        """Enable a single server via toggle API."""
        try:
            # Use the correct toggle endpoint format: /toggle/{service_path}
            # Remove leading slash from server_path if present
            clean_path = server_path.lstrip('/')
            toggle_url = f"{REGISTRY_URL}/toggle/{clean_path}"
            
            # Send form data with enabled="on" to enable the server
            data = aiohttp.FormData()
            data.add_field('enabled', 'on')
            
            async with session.post(toggle_url, data=data) as response:
                if response.status in [200, 201, 302, 303]:
                    # Check if response indicates server was enabled
                    response_text = await response.text()
                    return "enabled" in response_text.lower() or "success" in response_text.lower() or response.status in [302, 303]
                else:
                    print(f"Toggle failed for {server_path}: HTTP {response.status}")
                    return False
        except Exception as e:
            print(f"Error enabling {server_path}: {e}")
            return False
    
    async def test_faiss_performance(self, concurrent_limits: List[int] = [5, 10, 20]) -> List[Dict[str, Any]]:
        """Test FAISS performance with different concurrency levels."""
        print(f"Testing FAISS performance with concurrency levels: {concurrent_limits}")
        
        all_results = []
        
        for concurrent_limit in concurrent_limits:
            print(f"\n  Testing with {concurrent_limit} concurrent requests...")
            
            # Start system monitoring
            monitor = SystemMonitor()
            monitor.start()
            batch_start = time.time()
            
            # Run queries concurrently
            semaphore = asyncio.Semaphore(concurrent_limit)
            
            async def run_single_query(query: str, index: int):
                async with semaphore:
                    try:
                        start_time = time.time()
                        
                        # Add timeout for individual queries
                        query_timeout = 30  # 30 seconds per query
                        
                        async with asyncio.timeout(query_timeout):
                            async with sse_client(MCPGW_SERVER_URL) as (read, write):
                                async with ClientSession(read, write) as session:
                                    await session.initialize()
                                    
                                    result = await session.call_tool(
                                        "intelligent_tool_finder",
                                        {
                                            "natural_language_query": query,
                                            "username": ADMIN_USER,
                                            "password": ADMIN_PASSWORD,
                                            "top_k_services": 5,
                                            "top_n_tools": 10
                                        }
                                    )
                                    
                                    elapsed = time.time() - start_time
                                    
                                    # Parse tools - intelligent_tool_finder returns individual tool objects
                                    tools = []
                                    if result.content and len(result.content) > 0:
                                        # Each content item represents a tool result
                                        for content_item in result.content:
                                            if hasattr(content_item, 'text'):
                                                try:
                                                    # Try to parse as JSON tool object
                                                    parsed = json.loads(content_item.text)
                                                    if isinstance(parsed, dict) and "tool_name" in parsed:
                                                        tools.append(parsed)
                                                    elif isinstance(parsed, list):
                                                        tools.extend(parsed)
                                                except json.JSONDecodeError:
                                                    # If not JSON, might be plain text result
                                                    if content_item.text.strip():
                                                        tools.append({"raw_response": content_item.text})
                                    
                                    print(f"      Query: {query[:30]}... -> {len(tools)} tools found")
                                    
                                    return {
                                        "query": query,
                                        "elapsed_time": elapsed,
                                        "tools_found": len(tools),
                                        "success": len(tools) > 0,
                                        "concurrent_limit": concurrent_limit
                                    }
                                    
                    except asyncio.TimeoutError:
                        elapsed_time = time.time() - start_time if 'start_time' in locals() else query_timeout
                        print(f"      ⏰ Query timeout: {query[:40]}... - {query_timeout}s")
                        return {
                            "query": query,
                            "elapsed_time": elapsed_time,
                            "tools_found": 0,
                            "success": False,
                            "concurrent_limit": concurrent_limit,
                            "error": f"Timeout after {query_timeout}s"
                        }
                    except Exception as e:
                        elapsed_time = time.time() - start_time if 'start_time' in locals() else 0
                        print(f"      ✗ Query failed: {query[:40]}... - {str(e)}")
                        return {
                            "query": query,
                            "elapsed_time": elapsed_time,
                            "tools_found": 0,
                            "success": False,
                            "concurrent_limit": concurrent_limit,
                            "error": str(e)
                        }
            
            # Run all queries
            tasks = [run_single_query(query, i) for i, query in enumerate(STRESS_QUERIES)]
            batch_results = await asyncio.gather(*tasks)
            
            # Stop monitoring
            batch_end = time.time()
            system_data = monitor.stop()
            
            # Add system metrics
            if system_data:
                cpu_values = [m["cpu_percent"] for m in system_data]
                memory_values = [m["memory_percent"] for m in system_data]
                
                system_metrics = {
                    "avg_cpu_percent": np.mean(cpu_values),
                    "peak_cpu_percent": np.max(cpu_values),
                    "avg_memory_percent": np.mean(memory_values),
                    "peak_memory_percent": np.max(memory_values),
                    "batch_duration": batch_end - batch_start
                }
                
                for result in batch_results:
                    result["system_metrics"] = system_metrics
            
            all_results.extend(batch_results)
            
            successful = len([r for r in batch_results if r["success"]])
            avg_time = np.mean([r["elapsed_time"] for r in batch_results if r["success"]]) if successful > 0 else 0
            
            print(f"    ✓ Completed: {successful}/{len(batch_results)} successful, avg: {avg_time:.3f}s")
        
        self.results["faiss_results"] = all_results
        return all_results
    
    def test_ui_performance(self, headless: bool = True) -> Dict[str, Any]:
        """Test UI performance using Selenium with comprehensive coverage."""
        print("🌐 Testing UI performance...")
        
        # Setup Chrome driver with unique user data directory
        chrome_options = Options()
        if headless:
            chrome_options.add_argument("--headless")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("--window-size=1920,1080")
        chrome_options.add_argument("--disable-web-security")
        chrome_options.add_argument("--allow-running-insecure-content")
        chrome_options.add_argument("--disable-extensions")
        chrome_options.add_argument("--disable-plugins")
        chrome_options.add_argument("--disable-images")
        # Use unique user data directory to avoid conflicts
        import tempfile
        temp_dir = tempfile.mkdtemp()
        chrome_options.add_argument(f"--user-data-dir={temp_dir}")
        
        ui_results = {
            "login_time": 0,
            "page_load_time": 0,
            "servers_found": 0,
            "search_results": [],
            "navigation_tests": [],
            "element_tests": [],
            "errors": []
        }
        
        driver = None
        try:
            print("  Initializing Chrome driver...")
            driver = webdriver.Chrome(options=chrome_options)
            driver.set_page_load_timeout(30)  # Reduce timeout
            driver.implicitly_wait(3)  # Reduce implicit wait
            
            # Test 1: Login performance
            print("  Testing login...")
            start_time = time.time()
            try:
                driver.get(f"{REGISTRY_URL}/login")
                
                username_field = WebDriverWait(driver, 5).until(
                    EC.presence_of_element_located((By.NAME, "username"))
                )
                password_field = driver.find_element(By.NAME, "password")
                
                username_field.clear()
                username_field.send_keys(ADMIN_USER)
                password_field.clear()
                password_field.send_keys(ADMIN_PASSWORD)
                password_field.send_keys(Keys.RETURN)
                
                # Wait for successful login (redirect to main page)
                WebDriverWait(driver, 5).until(
                    lambda d: "/login" not in d.current_url
                )
                login_time = time.time() - start_time
                ui_results["login_time"] = login_time
                print(f"    ✓ Login successful: {login_time:.3f}s")
                
            except Exception as e:
                ui_results["errors"].append(f"Login failed: {str(e)}")
                print(f"    ✗ Login failed: {e}")
            
            # Test 2: Main page load performance
            print("  Testing main page load...")
            start_time = time.time()
            try:
                driver.get(REGISTRY_URL)
                
                # Wait for page to be fully loaded with timeout
                WebDriverWait(driver, 10).until(
                    lambda d: d.execute_script("return document.readyState") == "complete"
                )
                page_load_time = time.time() - start_time
                ui_results["page_load_time"] = page_load_time
                print(f"    ✓ Page loaded: {page_load_time:.3f}s")
                
            except Exception as e:
                ui_results["errors"].append(f"Page load failed: {str(e)}")
                print(f"    ✗ Page load failed: {e}")
            
            # Test 3: Count servers and page elements
            print("  Analyzing page elements...")
            try:
                # Try multiple selectors for server switches
                server_elements = []
                selectors_to_try = [
                    ".switch",
                    "input[type='checkbox']",
                    ".server-item",
                    ".server-row",
                    "[data-server]",
                    ".toggle-switch"
                ]
                
                for selector in selectors_to_try:
                    elements = driver.find_elements(By.CSS_SELECTOR, selector)
                    if elements:
                        server_elements = elements
                        print(f"    Found {len(elements)} elements with selector '{selector}'")
                        break
                
                ui_results["servers_found"] = len(server_elements)
                
                # Count other page elements with timeout protection
                ui_results["element_tests"] = [
                    {"element": "buttons", "count": len(driver.find_elements(By.TAG_NAME, "button"))},
                    {"element": "forms", "count": len(driver.find_elements(By.TAG_NAME, "form"))},
                    {"element": "inputs", "count": len(driver.find_elements(By.TAG_NAME, "input"))},
                    {"element": "links", "count": len(driver.find_elements(By.TAG_NAME, "a"))},
                    {"element": "tables", "count": len(driver.find_elements(By.TAG_NAME, "table"))},
                ]
                
            except Exception as e:
                ui_results["errors"].append(f"Element analysis failed: {str(e)}")
                print(f"    ✗ Element analysis failed: {e}")
            
            # Test 4: Enhanced search functionality testing with robust timeout protection
            print("  Testing search functionality...")
            search_results = []
            
            try:
                # Try multiple search input selectors
                search_input = None
                search_selectors = [
                    "input[name='query']",
                    "input[placeholder*='search' i]",
                    "input[type='search']",
                    "#search",
                    ".search-input",
                    "input[id*='search' i]",
                    "input[class*='search' i]"
                ]
                
                for selector in search_selectors:
                    try:
                        search_input = WebDriverWait(driver, 2).until(
                            EC.presence_of_element_located((By.CSS_SELECTOR, selector))
                        )
                        print(f"    Found search input with selector: {selector}")
                        break
                    except:
                        continue
                
                if search_input:
                    # Test only 2 queries to avoid hanging, with aggressive timeouts
                    test_queries = STRESS_QUERIES[:2]
                    
                    for i, query in enumerate(test_queries):
                        print(f"    Testing search query {i+1}: {query[:30]}...")
                        
                        try:
                            # Set a hard timeout for this entire search operation
                            import signal
                            
                            def timeout_handler(signum, frame):
                                raise TimeoutError("Search operation timed out")
                            
                            # Set 5-second timeout for the entire search operation
                            signal.signal(signal.SIGALRM, timeout_handler)
                            signal.alarm(5)
                            
                            try:
                                start_time = time.time()
                                
                                # Clear and enter search query using JavaScript to avoid hanging
                                driver.execute_script("arguments[0].value = '';", search_input)
                                driver.execute_script("arguments[0].value = arguments[1];", search_input, query)
                                
                                # Get initial page state
                                initial_url = driver.current_url
                                initial_page_source_length = len(driver.page_source)
                                
                                # Try multiple submission methods with immediate fallback
                                search_submitted = False
                                
                                # Method 1: JavaScript form submission (most reliable)
                                try:
                                    driver.execute_script("""
                                        var input = arguments[0];
                                        var form = input.closest('form');
                                        if (form) {
                                            form.submit();
                                        } else {
                                            // Trigger enter key event
                                            var event = new KeyboardEvent('keydown', {key: 'Enter', keyCode: 13});
                                            input.dispatchEvent(event);
                                        }
                                    """, search_input)
                                    search_submitted = True
                                except Exception as e:
                                    print(f"      JavaScript submission failed: {e}")
                                
                                # Method 2: Direct keys if JS failed
                                if not search_submitted:
                                    try:
                                        search_input.send_keys(Keys.RETURN)
                                        search_submitted = True
                                    except Exception as e:
                                        print(f"      Keys.RETURN failed: {e}")
                                
                                # Method 3: Click submit button if available
                                if not search_submitted:
                                    try:
                                        submit_btn = driver.find_element(By.CSS_SELECTOR, "button[type='submit'], input[type='submit']")
                                        driver.execute_script("arguments[0].click();", submit_btn)
                                        search_submitted = True
                                    except Exception as e:
                                        print(f"      Submit button click failed: {e}")
                                
                                if search_submitted:
                                    # Wait briefly for any changes (max 2 seconds)
                                    changes_detected = False
                                    for wait_iteration in range(20):  # 20 * 0.1s = 2s max
                                        time.sleep(0.1)
                                        
                                        # Check if page changed
                                        current_url = driver.current_url
                                        current_page_length = len(driver.page_source)
                                        
                                        if (current_url != initial_url or 
                                            abs(current_page_length - initial_page_source_length) > 100):
                                            changes_detected = True
                                            break
                                    
                                    search_time = time.time() - start_time
                                    
                                    # Quick scan for results without hanging
                                    results = []
                                    result_selectors = [
                                        ".search-result", ".tool-result", ".result-item",
                                        "[data-tool]", ".tool-card", ".server-result",
                                        "tr[data-server]", ".tool-list-item", "tbody tr"
                                    ]
                                    
                                    # Use JavaScript to count results quickly
                                    for selector in result_selectors:
                                        try:
                                            count = driver.execute_script(f"""
                                                return document.querySelectorAll('{selector}').length;
                                            """)
                                            if count > 0:
                                                results.extend([f"result_{i}" for i in range(count)])
                                                break
                                        except:
                                            continue
                                    
                                    # Remove duplicates
                                    results = list(set(results))
                                    
                                    search_results.append({
                                        "query": query,
                                        "search_time": search_time,
                                        "results_found": len(results),
                                        "success": len(results) > 0 or changes_detected,
                                        "changes_detected": changes_detected,
                                        "submission_method": "javascript" if search_submitted else "failed"
                                    })
                                    
                                    print(f"      ✓ Search completed: {len(results)} results, {search_time:.3f}s, changes: {changes_detected}")
                                    
                                else:
                                    search_results.append({
                                        "query": query,
                                        "search_time": 0,
                                        "results_found": 0,
                                        "success": False,
                                        "error": "Could not submit search form"
                                    })
                                    print(f"      ✗ Could not submit search")
                                
                            finally:
                                # Cancel the alarm
                                signal.alarm(0)
                                
                        except TimeoutError:
                            search_results.append({
                                "query": query,
                                "search_time": 5.0,
                                "results_found": 0,
                                "success": False,
                                "error": "Search operation timed out after 5 seconds"
                            })
                            print(f"      ⏰ Search timed out after 5 seconds")
                            
                        except Exception as e:
                            search_results.append({
                                "query": query,
                                "search_time": 0,
                                "results_found": 0,
                                "success": False,
                                "error": str(e)
                            })
                            print(f"      ✗ Search failed: {e}")
                        
                        # Reset for next query - navigate back to main page
                        try:
                            driver.get(REGISTRY_URL)
                            WebDriverWait(driver, 3).until(
                                lambda d: d.execute_script("return document.readyState") == "complete"
                            )
                            # Re-find search input
                            search_input = driver.find_element(By.CSS_SELECTOR, search_selectors[0])
                        except:
                            print(f"      Could not reset for next search query")
                            break
                            
                else:
                    ui_results["errors"].append("Could not find search input field")
                    print("    ✗ No search input field found")
                    
            except Exception as e:
                ui_results["errors"].append(f"Search test setup failed: {str(e)}")
                print(f"    ✗ Search test setup failed: {e}")
            
            ui_results["search_results"] = search_results
            
            # Test 5: Quick navigation performance (limited to avoid hanging)
            print("  Testing navigation...")
            navigation_tests = []
            
            # Test only key pages
            test_urls = [
                ("Home", "/"),
                ("Login", "/login")
            ]
            
            for name, path in test_urls:
                try:
                    start_time = time.time()
                    driver.get(f"{REGISTRY_URL}{path}")
                    WebDriverWait(driver, 5).until(
                        lambda d: d.execute_script("return document.readyState") == "complete"
                    )
                    nav_time = time.time() - start_time
                    
                    navigation_tests.append({
                        "page": name,
                        "url": path,
                        "load_time": nav_time,
                        "status_code": 200,
                        "success": True
                    })
                    
                except Exception as e:
                    navigation_tests.append({
                        "page": name,
                        "url": path,
                        "load_time": 0,
                        "error": str(e),
                        "success": False
                    })
            
            ui_results["navigation_tests"] = navigation_tests
            
            # Test 6: Quick JavaScript performance check
            print("  Testing JavaScript performance...")
            try:
                js_performance = driver.execute_script("""
                    return {
                        timing: performance.timing,
                        navigation: performance.navigation,
                        memory: performance.memory || {}
                    };
                """)
                ui_results["js_performance"] = js_performance
            except Exception as e:
                ui_results["errors"].append(f"JavaScript performance test failed: {str(e)}")
            
            print("  ✓ UI tests completed")
            
        except Exception as e:
            ui_results["errors"].append(f"UI test suite failed: {str(e)}")
            print(f"    ✗ UI test suite failed: {e}")
            
        finally:
            # Ensure driver is closed
            if driver:
                try:
                    driver.quit()
                    print("  ✓ Chrome driver closed")
                except:
                    pass
            
            # Clean up temp directory
            try:
                import shutil
                if 'temp_dir' in locals():
                    shutil.rmtree(temp_dir, ignore_errors=True)
            except:
                pass
        
        self.results["ui_results"] = ui_results
        return ui_results
    
    async def test_ui_search_concurrency(self, concurrent_limits: List[int] = [5, 10, 20]) -> List[Dict[str, Any]]:
        """Test UI search performance with different concurrency levels using HTTP requests."""
        print(f"Testing UI Search concurrency with levels: {concurrent_limits}")
        
        all_results = []
        
        # First, check what search endpoints actually exist
        print("  Discovering available search endpoints...")
        available_endpoints = []
        
        async with aiohttp.ClientSession() as session:
            # Test different potential endpoints
            test_endpoints = [
                f"{REGISTRY_URL}",  # Main page with search
                f"{REGISTRY_URL}/search",  # Dedicated search endpoint
                f"{REGISTRY_URL}/tools",  # Tools search
                f"{REGISTRY_URL}/servers"  # Server search
            ]
            
            for endpoint in test_endpoints:
                try:
                    async with session.get(endpoint) as response:
                        if response.status == 200:
                            content = await response.text()
                            # Check if this page has search functionality
                            search_indicators = [
                                'type="search"',
                                'name="query"',
                                'name="q"',
                                'placeholder="search"',
                                'id="search"'
                            ]
                            
                            if any(indicator.lower() in content.lower() for indicator in search_indicators):
                                available_endpoints.append(endpoint)
                                print(f"    ✓ Found search capability at: {endpoint}")
                except:
                    continue
        
        if not available_endpoints:
            print("  No search endpoints found, using main page only")
            available_endpoints = [REGISTRY_URL]
        
        # Create a persistent session with proper authentication
        connector = aiohttp.TCPConnector(limit=100, limit_per_host=50)
        session_timeout = aiohttp.ClientTimeout(total=30, connect=10)
        
        async with aiohttp.ClientSession(
            connector=connector, 
            timeout=session_timeout,
            cookie_jar=aiohttp.CookieJar()
        ) as persistent_session:
            
            # Login once with the persistent session
            try:
                login_data = aiohttp.FormData()
                login_data.add_field('username', ADMIN_USER)
                login_data.add_field('password', ADMIN_PASSWORD)
                
                async with persistent_session.post(f"{REGISTRY_URL}/login", data=login_data) as response:
                    if response.status in [200, 201, 302, 303]:
                        print(f"  ✓ Login successful for search testing")
                    else:
                        print(f"  ✗ Login failed for search testing: {response.status}")
                        # Continue anyway, some endpoints might not require auth
            except Exception as e:
                print(f"  Login failed: {e}, continuing without authentication")
            
            for concurrent_limit in concurrent_limits:
                print(f"\n  Testing UI search with {concurrent_limit} concurrent requests...")
                
                # Start system monitoring
                monitor = SystemMonitor()
                monitor.start()
                batch_start = time.time()
                
                # Create a limited set of search queries for more realistic testing
                test_queries = [
                    "stock",
                    "time", 
                    "API",
                    "data",
                    "server",
                    "tool",
                    "financial",
                    "search"
                ]
                
                # Cycle through queries to get the desired number of concurrent requests
                search_queries = []
                for i in range(concurrent_limit):
                    search_queries.append(test_queries[i % len(test_queries)])
                
                # Run searches with proper concurrency control
                semaphore = asyncio.Semaphore(min(concurrent_limit, 50))  # Cap at 50 to avoid overwhelming
                
                async def run_single_search(query: str, request_index: int):
                    async with semaphore:
                        try:
                            start_time = time.time()
                            
                            # Try the available endpoints
                            search_successful = False
                            results_found = 0
                            response_content = ""
                            
                            for endpoint in available_endpoints:
                                try:
                                    # Use different search methods for different endpoints
                                    if endpoint.endswith('/search'):
                                        # POST to dedicated search endpoint
                                        search_data = aiohttp.FormData()
                                        search_data.add_field('query', query)
                                        search_data.add_field('q', query)
                                        
                                        async with persistent_session.post(endpoint, data=search_data) as response:
                                            if response.status == 200:
                                                response_content = await response.text()
                                                search_successful = True
                                                break
                                    else:
                                        # GET with query parameters
                                        params = {
                                            'query': query,
                                            'q': query,
                                            'search': query
                                        }
                                        
                                        async with persistent_session.get(endpoint, params=params) as response:
                                            if response.status == 200:
                                                response_content = await response.text()
                                                search_successful = True
                                                break
                                
                                except asyncio.TimeoutError:
                                    continue
                                except Exception:
                                    continue
                            
                            # Count results more broadly - look for any server/tool indicators
                            if search_successful and response_content:
                                # Look for various result indicators
                                result_patterns = [
                                    '<tr',  # Table rows (likely servers)
                                    'data-server',  # Server data attributes
                                    'data-tool',  # Tool data attributes
                                    'class="server',  # Server CSS classes
                                    'class="tool',  # Tool CSS classes
                                    '<div class="card',  # Bootstrap cards
                                    'server-item',  # Server items
                                    'tool-item',  # Tool items
                                    '<option',  # Select options
                                    'result-item'  # Generic result items
                                ]
                                
                                content_lower = response_content.lower()
                                for pattern in result_patterns:
                                    count = content_lower.count(pattern.lower())
                                    results_found += count
                                
                                # Remove duplicates by dividing by estimated patterns per item
                                results_found = max(1, results_found // 3) if results_found > 0 else 0
                            
                            elapsed = time.time() - start_time
                            
                            return {
                                "query": query,
                                "elapsed_time": elapsed,
                                "results_found": results_found,
                                "success": search_successful,
                                "concurrent_limit": concurrent_limit,
                                "search_type": "ui_http",
                                "request_index": request_index,
                                "endpoint_used": available_endpoints[0] if available_endpoints else "unknown"
                            }
                            
                        except asyncio.TimeoutError:
                            elapsed_time = time.time() - start_time if 'start_time' in locals() else 30
                            return {
                                "query": query,
                                "elapsed_time": elapsed_time,
                                "results_found": 0,
                                "success": False,
                                "concurrent_limit": concurrent_limit,
                                "error": "Timeout after 30s",
                                "search_type": "ui_http",
                                "request_index": request_index
                            }
                        except Exception as e:
                            elapsed_time = time.time() - start_time if 'start_time' in locals() else 0
                            return {
                                "query": query,
                                "elapsed_time": elapsed_time,
                                "results_found": 0,
                                "success": False,
                                "concurrent_limit": concurrent_limit,
                                "error": str(e),
                                "search_type": "ui_http",
                                "request_index": request_index
                            }
                
                # Run all search queries concurrently
                tasks = [run_single_search(query, i) for i, query in enumerate(search_queries)]
                batch_results = await asyncio.gather(*tasks, return_exceptions=True)
                
                # Filter out exceptions and process results
                valid_results = []
                for result in batch_results:
                    if isinstance(result, dict):
                        valid_results.append(result)
                    else:
                        # Handle exceptions
                        valid_results.append({
                            "query": "unknown",
                            "elapsed_time": 0,
                            "results_found": 0,
                            "success": False,
                            "concurrent_limit": concurrent_limit,
                            "error": str(result),
                            "search_type": "ui_http"
                        })
                
                # Stop monitoring
                batch_end = time.time()
                system_data = monitor.stop()
                
                # Add system metrics
                if system_data:
                    cpu_values = [m["cpu_percent"] for m in system_data]
                    memory_values = [m["memory_percent"] for m in system_data]
                    
                    system_metrics = {
                        "avg_cpu_percent": np.mean(cpu_values),
                        "peak_cpu_percent": np.max(cpu_values),
                        "avg_memory_percent": np.mean(memory_values),
                        "peak_memory_percent": np.max(memory_values),
                        "batch_duration": batch_end - batch_start
                    }
                    
                    for result in valid_results:
                        result["system_metrics"] = system_metrics
                
                all_results.extend(valid_results)
                
                successful = len([r for r in valid_results if r["success"]])
                avg_time = np.mean([r["elapsed_time"] for r in valid_results if r["success"]]) if successful > 0 else 0
                avg_results = np.mean([r["results_found"] for r in valid_results if r["success"]]) if successful > 0 else 0
                
                print(f"    ✓ Completed: {successful}/{len(valid_results)} successful, avg: {avg_time:.3f}s, avg results: {avg_results:.1f}")
                
                # Add a small delay between concurrency levels to let the server recover
                if concurrent_limit < max(concurrent_limits):
                    await asyncio.sleep(2)
        
        return all_results
    
    def generate_performance_charts(self) -> Dict[str, str]:
        """Generate interactive performance analysis charts using Chart.js data."""
        print("Generating interactive performance charts...")
        
        charts = {}
        
        # FAISS Performance Chart Data
        if self.results["faiss_results"]:
            faiss_df = pd.DataFrame(self.results["faiss_results"])
            
            if not faiss_df.empty:
                # Response times by concurrency
                success_df = faiss_df[faiss_df['success'] == True]
                if not success_df.empty:
                    concurrency_groups = success_df.groupby('concurrent_limit')['elapsed_time'].apply(list)
                    
                    # Prepare boxplot data for Chart.js
                    boxplot_data = []
                    labels = []
                    for limit, times in concurrency_groups.items():
                        labels.append(str(limit))
                        boxplot_data.append({
                            'min': float(np.min(times)),
                            'q1': float(np.percentile(times, 25)),
                            'median': float(np.median(times)),
                            'q3': float(np.percentile(times, 75)),
                            'max': float(np.max(times)),
                            'mean': float(np.mean(times))
                        })
                    
                    charts["faiss_boxplot_data"] = json.dumps(boxplot_data)
                    charts["faiss_boxplot_labels"] = json.dumps(labels)
                
                # Success rates
                success_rates = faiss_df.groupby('concurrent_limit').agg({
                    'success': ['count', 'sum']
                })
                success_rates.columns = ['total', 'successful']
                success_rates['success_rate'] = (success_rates['successful'] / success_rates['total'] * 100)
                
                charts["faiss_success_labels"] = json.dumps([str(idx) for idx in success_rates.index.tolist()])
                charts["faiss_success_data"] = json.dumps([round(rate, 1) for rate in success_rates['success_rate'].tolist()])
        
        # UI Performance Chart Data
        if self.results["ui_results"]:
            ui_data = self.results["ui_results"]
            
            # Performance metrics
            metrics = ["login_time", "page_load_time"]
            values = [ui_data.get(metric, 0) for metric in metrics]
            labels = ["Login", "Page Load"]
            
            charts["ui_performance_labels"] = json.dumps(labels)
            charts["ui_performance_data"] = json.dumps([round(v, 3) for v in values])
            
            # Element count data
            element_data = ui_data.get("element_tests", [])
            if element_data:
                element_names = [elem["element"] for elem in element_data]
                element_counts = [elem["count"] for elem in element_data]
                
                charts["ui_elements_labels"] = json.dumps(element_names)
                charts["ui_elements_data"] = json.dumps(element_counts)
            
            # Navigation performance
            nav_data = ui_data.get("navigation_tests", [])
            if nav_data:
                nav_pages = [nav["page"] for nav in nav_data if nav.get("success", False)]
                nav_times = [nav["load_time"] for nav in nav_data if nav.get("success", False)]
                
                if nav_pages and nav_times:
                    charts["ui_nav_labels"] = json.dumps(nav_pages)
                    charts["ui_nav_data"] = json.dumps([round(t, 3) for t in nav_times])
        
        # UI Search Concurrency Chart Data
        ui_search_results = self.results.get("ui_search_results", [])
        if ui_search_results:
            search_df = pd.DataFrame(ui_search_results)
            
            if not search_df.empty:
                # Response times by concurrency
                success_search_df = search_df[search_df['success'] == True]
                if not success_search_df.empty:
                    search_concurrency_groups = success_search_df.groupby('concurrent_limit')['elapsed_time'].apply(list)
                    
                    # Prepare boxplot data
                    search_boxplot_data = []
                    search_labels = []
                    for limit, times in search_concurrency_groups.items():
                        search_labels.append(str(limit))
                        search_boxplot_data.append({
                            'min': float(np.min(times)),
                            'q1': float(np.percentile(times, 25)),
                            'median': float(np.median(times)),
                            'q3': float(np.percentile(times, 75)),
                            'max': float(np.max(times)),
                            'mean': float(np.mean(times))
                        })
                    
                    charts["ui_search_boxplot_data"] = json.dumps(search_boxplot_data)
                    charts["ui_search_boxplot_labels"] = json.dumps(search_labels)
                
                # Success rates for UI search
                search_success_rates = search_df.groupby('concurrent_limit').agg({
                    'success': ['count', 'sum']
                })
                search_success_rates.columns = ['total', 'successful']
                search_success_rates['success_rate'] = (search_success_rates['successful'] / search_success_rates['total'] * 100)
                
                charts["ui_search_success_labels"] = json.dumps([str(idx) for idx in search_success_rates.index.tolist()])
                charts["ui_search_success_data"] = json.dumps([round(rate, 1) for rate in search_success_rates['success_rate'].tolist()])
        
        # System Resource Usage Data
        all_results = self.results.get("faiss_results", []) + self.results.get("ui_search_results", [])
        system_data = []
        for result in all_results:
            if "system_metrics" in result:
                metrics = result["system_metrics"]
                metrics["test_type"] = "FAISS" if result in self.results.get("faiss_results", []) else "UI Search"
                system_data.append(metrics)
        
        if system_data:
            # CPU and Memory usage by test type
            cpu_data = {}
            memory_data = {}
            for data in system_data:
                test_type = data["test_type"]
                if test_type not in cpu_data:
                    cpu_data[test_type] = []
                    memory_data[test_type] = []
                cpu_data[test_type].append(data.get("peak_cpu_percent", 0))
                memory_data[test_type].append(data.get("peak_memory_percent", 0))
            
            if cpu_data:
                test_types = list(cpu_data.keys())
                cpu_averages = [round(np.mean(cpu_data[t]), 1) for t in test_types]
                memory_averages = [round(np.mean(memory_data[t]), 1) for t in test_types]
                
                charts["system_test_types"] = json.dumps(test_types)
                charts["system_cpu_data"] = json.dumps(cpu_averages)
                charts["system_memory_data"] = json.dumps(memory_averages)
            
            # Resource usage timeline
            if len(system_data) > 1:
                cpu_timeline = [round(data.get("peak_cpu_percent", 0), 1) for data in system_data]
                memory_timeline = [round(data.get("peak_memory_percent", 0), 1) for data in system_data]
                
                charts["system_timeline_cpu"] = json.dumps(cpu_timeline)
                charts["system_timeline_memory"] = json.dumps(memory_timeline)
                charts["system_timeline_labels"] = json.dumps([f"Phase {i+1}" for i in range(len(system_data))])
        
        # Response Time Distribution Data
        faiss_results = self.results.get("faiss_results", [])
        ui_search_results = self.results.get("ui_search_results", [])
        
        if faiss_results:
            faiss_times = [r["elapsed_time"] for r in faiss_results if r.get("success", False)]
            if faiss_times:
                # Create histogram data
                hist, bin_edges = np.histogram(faiss_times, bins=20)
                bin_centers = [(bin_edges[i] + bin_edges[i+1]) / 2 for i in range(len(bin_edges)-1)]
                
                charts["faiss_histogram_labels"] = json.dumps([f"{edge:.3f}" for edge in bin_centers])
                charts["faiss_histogram_data"] = json.dumps(hist.tolist())
                charts["faiss_mean_time"] = round(np.mean(faiss_times), 3)
                charts["faiss_median_time"] = round(np.median(faiss_times), 3)
        
        if ui_search_results:
            ui_times = [r["elapsed_time"] for r in ui_search_results if r.get("success", False)]
            if ui_times:
                # Create histogram data
                hist, bin_edges = np.histogram(ui_times, bins=20)
                bin_centers = [(bin_edges[i] + bin_edges[i+1]) / 2 for i in range(len(bin_edges)-1)]
                
                charts["ui_histogram_labels"] = json.dumps([f"{edge:.3f}" for edge in bin_centers])
                charts["ui_histogram_data"] = json.dumps(hist.tolist())
                charts["ui_mean_time"] = round(np.mean(ui_times), 3)
                charts["ui_median_time"] = round(np.median(ui_times), 3)
        
        # Performance Comparison Data
        if faiss_results and ui_search_results:
            faiss_avg = np.mean([r["elapsed_time"] for r in faiss_results if r.get("success", False)])
            ui_avg = np.mean([r["elapsed_time"] for r in ui_search_results if r.get("success", False)])
            
            faiss_success_rate = len([r for r in faiss_results if r.get("success", False)]) / len(faiss_results) * 100
            ui_success_rate = len([r for r in ui_search_results if r.get("success", False)]) / len(ui_search_results) * 100
            
            charts["comparison_categories"] = json.dumps(['FAISS Search', 'UI Search'])
            charts["comparison_response_times"] = json.dumps([round(faiss_avg, 3), round(ui_avg, 3)])
            charts["comparison_success_rates"] = json.dumps([round(faiss_success_rate, 1), round(ui_success_rate, 1)])
        
        print(f"Generated interactive chart data for {len([k for k in charts.keys() if not k.endswith('_data') and not k.endswith('_labels')])} chart types")
        return charts
    
    def generate_html_report(self, charts: Dict[str, str]) -> str:
        """Generate comprehensive HTML report with interactive charts."""
        
        # Calculate summary statistics
        total_faiss_tests = len(self.results.get("faiss_results", []))
        successful_faiss = len([r for r in self.results.get("faiss_results", []) if r.get("success", False)])
        faiss_success_rate = (successful_faiss / total_faiss_tests * 100) if total_faiss_tests > 0 else 0
        
        total_ui_search_tests = len(self.results.get("ui_search_results", []))
        successful_ui_search = len([r for r in self.results.get("ui_search_results", []) if r.get("success", False)])
        ui_search_success_rate = (successful_ui_search / total_ui_search_tests * 100) if total_ui_search_tests > 0 else 0
        
        avg_faiss_time = np.mean([r["elapsed_time"] for r in self.results.get("faiss_results", []) if r.get("success", False)]) if successful_faiss > 0 else 0
        avg_ui_search_time = np.mean([r["elapsed_time"] for r in self.results.get("ui_search_results", []) if r.get("success", False)]) if successful_ui_search > 0 else 0
        
        # Get system metrics
        all_results = self.results.get("faiss_results", []) + self.results.get("ui_search_results", [])
        peak_cpu = 0
        peak_memory = 0
        if all_results:
            cpu_values = []
            memory_values = []
            for result in all_results:
                if "system_metrics" in result:
                    cpu_values.append(result["system_metrics"].get("peak_cpu_percent", 0))
                    memory_values.append(result["system_metrics"].get("peak_memory_percent", 0))
            if cpu_values:
                peak_cpu = max(cpu_values)
                peak_memory = max(memory_values)
        
        html_template = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>MCP Gateway Stress Test Report</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/chartjs-adapter-date-fns/dist/chartjs-adapter-date-fns.bundle.min.js"></script>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
        }}
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            border-radius: 15px;
            box-shadow: 0 20px 40px rgba(0,0,0,0.1);
            overflow: hidden;
        }}
        .header {{
            background: linear-gradient(135deg, #2c3e50 0%, #3498db 100%);
            color: white;
            padding: 30px;
            text-align: center;
        }}
        .header h1 {{
            margin: 0;
            font-size: 2.5em;
            font-weight: 300;
        }}
        .header p {{
            margin: 10px 0 0 0;
            opacity: 0.9;
            font-size: 1.1em;
        }}
        .content {{
            padding: 30px;
        }}
        .summary-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 40px;
        }}
        .summary-card {{
            background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
            padding: 25px;
            border-radius: 10px;
            text-align: center;
            border-left: 5px solid #3498db;
        }}
        .summary-card h3 {{
            margin: 0 0 10px 0;
            color: #2c3e50;
            font-size: 1.1em;
        }}
        .summary-card .value {{
            font-size: 2em;
            font-weight: bold;
            color: #3498db;
            margin: 10px 0;
        }}
        .summary-card .unit {{
            color: #7f8c8d;
            font-size: 0.9em;
        }}
        .chart-section {{
            margin: 40px 0;
            background: #f8f9fa;
            border-radius: 10px;
            padding: 30px;
        }}
        .chart-section h2 {{
            color: #2c3e50;
            margin-bottom: 20px;
            font-size: 1.8em;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
        }}
        .chart-container {{
            position: relative;
            height: 400px;
            margin: 20px 0;
        }}
        .chart-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(500px, 1fr));
            gap: 30px;
            margin: 20px 0;
        }}
        .chart-description {{
            background: #e8f4f8;
            padding: 15px;
            border-radius: 8px;
            margin-bottom: 20px;
            border-left: 4px solid #3498db;
        }}
        .chart-description p {{
            margin: 0;
            color: #2c3e50;
            line-height: 1.6;
        }}
        .footer {{
            background: #34495e;
            color: white;
            text-align: center;
            padding: 20px;
            margin-top: 40px;
        }}
        .status-indicator {{
            display: inline-block;
            width: 12px;
            height: 12px;
            border-radius: 50%;
            margin-right: 8px;
        }}
        .status-success {{ background-color: #27ae60; }}
        .status-warning {{ background-color: #f39c12; }}
        .status-error {{ background-color: #e74c3c; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>MCP Gateway Stress Test Report</h1>
            <p>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>
        
        <div class="content">
            <!-- Summary Statistics -->
            <div class="summary-grid">
                <div class="summary-card">
                    <h3>FAISS Tests</h3>
                    <div class="value">{total_faiss_tests}</div>
                    <div class="unit">
                        <span class="status-indicator {'status-success' if faiss_success_rate > 90 else 'status-warning' if faiss_success_rate > 70 else 'status-error'}"></span>
                        {faiss_success_rate:.1f}% Success Rate
                    </div>
                </div>
                <div class="summary-card">
                    <h3>UI Search Tests</h3>
                    <div class="value">{total_ui_search_tests}</div>
                    <div class="unit">
                        <span class="status-indicator {'status-success' if ui_search_success_rate > 90 else 'status-warning' if ui_search_success_rate > 70 else 'status-error'}"></span>
                        {ui_search_success_rate:.1f}% Success Rate
                    </div>
                </div>
                <div class="summary-card">
                    <h3>Avg FAISS Response</h3>
                    <div class="value">{avg_faiss_time:.3f}</div>
                    <div class="unit">seconds</div>
                </div>
                <div class="summary-card">
                    <h3>Avg UI Search Response</h3>
                    <div class="value">{avg_ui_search_time:.3f}</div>
                    <div class="unit">seconds</div>
                </div>
                <div class="summary-card">
                    <h3>Peak CPU Usage</h3>
                    <div class="value">{peak_cpu:.1f}</div>
                    <div class="unit">percent</div>
                </div>
                <div class="summary-card">
                    <h3>Peak Memory Usage</h3>
                    <div class="value">{peak_memory:.1f}</div>
                    <div class="unit">percent</div>
                </div>
            </div>
"""

        # Add FAISS Performance Charts
        if "faiss_boxplot_data" in charts:
            html_template += f"""
            <div class="chart-section">
                <h2>FAISS Performance Analysis</h2>
                <div class="chart-description">
                    <p>This section analyzes FAISS search performance across different concurrency levels, showing response time distributions and success rates.</p>
                </div>
                <div class="chart-grid">
                    <div class="chart-container">
                        <canvas id="faissBoxplot"></canvas>
                    </div>
                    <div class="chart-container">
                        <canvas id="faissSuccess"></canvas>
                    </div>
                </div>
            </div>
"""

        # Add UI Performance Charts
        if "ui_performance_data" in charts:
            html_template += f"""
            <div class="chart-section">
                <h2>🖥️ UI Performance Analysis</h2>
                <div class="chart-description">
                    <p>Comprehensive analysis of UI performance including login times, page loads, element detection, and navigation performance.</p>
                </div>
                <div class="chart-grid">
                    <div class="chart-container">
                        <canvas id="uiPerformance"></canvas>
                    </div>
"""
            if "ui_elements_data" in charts:
                html_template += """
                    <div class="chart-container">
                        <canvas id="uiElements"></canvas>
                    </div>
"""
            html_template += """
                </div>
"""
            if "ui_nav_data" in charts:
                html_template += """
                <div class="chart-container">
                    <canvas id="uiNavigation"></canvas>
                </div>
"""
            html_template += """
            </div>
"""

        # Add UI Search Concurrency Charts
        if "ui_search_boxplot_data" in charts:
            html_template += f"""
            <div class="chart-section">
                <h2>UI Search Concurrency Analysis</h2>
                <div class="chart-description">
                    <p>Analysis of UI search performance under concurrent load, showing how response times and success rates vary with concurrency levels.</p>
                </div>
                <div class="chart-grid">
                    <div class="chart-container">
                        <canvas id="uiSearchBoxplot"></canvas>
                    </div>
                    <div class="chart-container">
                        <canvas id="uiSearchSuccess"></canvas>
                    </div>
                </div>
            </div>
"""

        # Add System Resource Charts
        if "system_cpu_data" in charts:
            html_template += f"""
            <div class="chart-section">
                <h2>💻 System Resource Usage</h2>
                <div class="chart-description">
                    <p>System resource utilization during testing, showing CPU and memory usage patterns across different test types and over time.</p>
                </div>
                <div class="chart-grid">
                    <div class="chart-container">
                        <canvas id="systemResources"></canvas>
                    </div>
"""
            if "system_timeline_cpu" in charts:
                html_template += """
                    <div class="chart-container">
                        <canvas id="systemTimeline"></canvas>
                    </div>
"""
            html_template += """
                </div>
            </div>
"""

        # Add Response Time Distribution Charts
        if "faiss_histogram_data" in charts or "ui_histogram_data" in charts:
            html_template += f"""
            <div class="chart-section">
                <h2>📈 Response Time Distribution</h2>
                <div class="chart-description">
                    <p>Statistical analysis of response time distributions, showing frequency patterns and key metrics like mean and median response times.</p>
                </div>
                <div class="chart-grid">
"""
            if "faiss_histogram_data" in charts:
                html_template += """
                    <div class="chart-container">
                        <div id="faissHistogram"></div>
                    </div>
"""
            if "ui_histogram_data" in charts:
                html_template += """
                    <div class="chart-container">
                        <div id="uiHistogram"></div>
                    </div>
"""
            html_template += """
                </div>
            </div>
"""

        # Add Performance Comparison Charts
        if "comparison_categories" in charts:
            html_template += f"""
            <div class="chart-section">
                <h2>⚖️ Performance Comparison</h2>
                <div class="chart-description">
                    <p>Direct comparison between FAISS and UI search performance, highlighting differences in response times and success rates.</p>
                </div>
                <div class="chart-grid">
                    <div class="chart-container">
                        <canvas id="comparisonResponseTime"></canvas>
                    </div>
                    <div class="chart-container">
                        <canvas id="comparisonSuccessRate"></canvas>
                    </div>
                </div>
            </div>
"""

        # Close HTML and add JavaScript
        html_template += """
        </div>
        
        <div class="footer">
            <p>Generated by MCP Gateway Stress Test Suite | Interactive Charts powered by Chart.js & Plotly</p>
        </div>
    </div>

    <script>
        // Chart.js default configuration
        Chart.defaults.font.family = "'Segoe UI', Tahoma, Geneva, Verdana, sans-serif";
        Chart.defaults.plugins.legend.position = 'top';
        Chart.defaults.plugins.legend.labels.usePointStyle = true;
        
        // Color schemes
        const colors = {
            primary: '#3498db',
            secondary: '#2ecc71',
            warning: '#f39c12',
            danger: '#e74c3c',
            info: '#9b59b6',
            light: '#ecf0f1',
            dark: '#2c3e50'
        };
"""

        # Add FAISS charts JavaScript
        if "faiss_boxplot_data" in charts:
            html_template += f"""
        // FAISS Boxplot Chart
        const faissBoxplotCtx = document.getElementById('faissBoxplot').getContext('2d');
        const faissBoxplotData = {charts["faiss_boxplot_data"]};
        const faissBoxplotLabels = {charts["faiss_boxplot_labels"]};
        
        new Chart(faissBoxplotCtx, {{
            type: 'bar',
            data: {{
                labels: faissBoxplotLabels,
                datasets: [{{
                    label: 'Min',
                    data: faissBoxplotData.map(d => d.min),
                    backgroundColor: 'rgba(52, 152, 219, 0.3)',
                    borderColor: colors.primary,
                    borderWidth: 1
                }}, {{
                    label: 'Q1',
                    data: faissBoxplotData.map(d => d.q1),
                    backgroundColor: 'rgba(52, 152, 219, 0.5)',
                    borderColor: colors.primary,
                    borderWidth: 1
                }}, {{
                    label: 'Median',
                    data: faissBoxplotData.map(d => d.median),
                    backgroundColor: 'rgba(52, 152, 219, 0.7)',
                    borderColor: colors.primary,
                    borderWidth: 2
                }}, {{
                    label: 'Q3',
                    data: faissBoxplotData.map(d => d.q3),
                    backgroundColor: 'rgba(52, 152, 219, 0.5)',
                    borderColor: colors.primary,
                    borderWidth: 1
                }}, {{
                    label: 'Max',
                    data: faissBoxplotData.map(d => d.max),
                    backgroundColor: 'rgba(52, 152, 219, 0.3)',
                    borderColor: colors.primary,
                    borderWidth: 1
                }}]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                plugins: {{
                    title: {{
                        display: true,
                        text: 'FAISS Response Time Distribution by Concurrency'
                    }},
                    tooltip: {{
                        callbacks: {{
                            afterBody: function(context) {{
                                const index = context[0].dataIndex;
                                const data = faissBoxplotData[index];
                                return `Mean: ${{data.mean.toFixed(3)}}s`;
                            }}
                        }}
                    }}
                }},
                scales: {{
                    x: {{
                        title: {{
                            display: true,
                            text: 'Concurrent Request Limit'
                        }}
                    }},
                    y: {{
                        title: {{
                            display: true,
                            text: 'Response Time (seconds)'
                        }}
                    }}
                }}
            }}
        }});

        // FAISS Success Rate Chart
        const faissSuccessCtx = document.getElementById('faissSuccess').getContext('2d');
        new Chart(faissSuccessCtx, {{
            type: 'line',
            data: {{
                labels: {charts["faiss_success_labels"]},
                datasets: [{{
                    label: 'Success Rate (%)',
                    data: {charts["faiss_success_data"]},
                    borderColor: colors.secondary,
                    backgroundColor: 'rgba(46, 204, 113, 0.1)',
                    borderWidth: 3,
                    fill: true,
                    tension: 0.4,
                    pointBackgroundColor: colors.secondary,
                    pointBorderColor: '#fff',
                    pointBorderWidth: 2,
                    pointRadius: 6
                }}]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                plugins: {{
                    title: {{
                        display: true,
                        text: 'FAISS Success Rate by Concurrency'
                    }}
                }},
                scales: {{
                    x: {{
                        title: {{
                            display: true,
                            text: 'Concurrent Request Limit'
                        }}
                    }},
                    y: {{
                        title: {{
                            display: true,
                            text: 'Success Rate (%)'
                        }},
                        min: 0,
                        max: 100
                    }}
                }}
            }}
        }});
"""

        # Add UI Performance charts JavaScript
        if "ui_performance_data" in charts:
            html_template += f"""
        // UI Performance Chart
        const uiPerformanceCtx = document.getElementById('uiPerformance').getContext('2d');
        new Chart(uiPerformanceCtx, {{
            type: 'doughnut',
            data: {{
                labels: {charts["ui_performance_labels"]},
                datasets: [{{
                    data: {charts["ui_performance_data"]},
                    backgroundColor: [colors.info, colors.warning],
                    borderColor: '#fff',
                    borderWidth: 3
                }}]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                plugins: {{
                    title: {{
                        display: true,
                        text: 'UI Performance Metrics'
                    }},
                    tooltip: {{
                        callbacks: {{
                            label: function(context) {{
                                return context.label + ': ' + context.parsed + 's';
                            }}
                        }}
                    }}
                }}
            }}
        }});
"""

        if "ui_elements_data" in charts:
            html_template += f"""
        // UI Elements Chart
        const uiElementsCtx = document.getElementById('uiElements').getContext('2d');
        new Chart(uiElementsCtx, {{
            type: 'polarArea',
            data: {{
                labels: {charts["ui_elements_labels"]},
                datasets: [{{
                    data: {charts["ui_elements_data"]},
                    backgroundColor: [
                        'rgba(255, 99, 132, 0.7)',
                        'rgba(54, 162, 235, 0.7)',
                        'rgba(255, 205, 86, 0.7)',
                        'rgba(75, 192, 192, 0.7)',
                        'rgba(153, 102, 255, 0.7)'
                    ],
                    borderColor: '#fff',
                    borderWidth: 2
                }}]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                plugins: {{
                    title: {{
                        display: true,
                        text: 'UI Elements Detected'
                    }}
                }}
            }}
        }});
"""

        if "ui_nav_data" in charts:
            html_template += f"""
        // UI Navigation Chart
        const uiNavigationCtx = document.getElementById('uiNavigation').getContext('2d');
        new Chart(uiNavigationCtx, {{
            type: 'radar',
            data: {{
                labels: {charts["ui_nav_labels"]},
                datasets: [{{
                    label: 'Load Time (seconds)',
                    data: {charts["ui_nav_data"]},
                    borderColor: colors.danger,
                    backgroundColor: 'rgba(231, 76, 60, 0.2)',
                    borderWidth: 2,
                    pointBackgroundColor: colors.danger,
                    pointBorderColor: '#fff',
                    pointBorderWidth: 2
                }}]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                plugins: {{
                    title: {{
                        display: true,
                        text: 'Navigation Performance'
                    }}
                }},
                scales: {{
                    r: {{
                        beginAtZero: true,
                        title: {{
                            display: true,
                            text: 'Load Time (seconds)'
                        }}
                    }}
                }}
            }}
        }});
"""

        # Add UI Search charts JavaScript
        if "ui_search_boxplot_data" in charts:
            html_template += f"""
        // UI Search Boxplot Chart
        const uiSearchBoxplotCtx = document.getElementById('uiSearchBoxplot').getContext('2d');
        const uiSearchBoxplotData = {charts["ui_search_boxplot_data"]};
        const uiSearchBoxplotLabels = {charts["ui_search_boxplot_labels"]};
        
        new Chart(uiSearchBoxplotCtx, {{
            type: 'bar',
            data: {{
                labels: uiSearchBoxplotLabels,
                datasets: [{{
                    label: 'Min',
                    data: uiSearchBoxplotData.map(d => d.min),
                    backgroundColor: 'rgba(46, 204, 113, 0.3)',
                    borderColor: colors.secondary,
                    borderWidth: 1
                }}, {{
                    label: 'Q1',
                    data: uiSearchBoxplotData.map(d => d.q1),
                    backgroundColor: 'rgba(46, 204, 113, 0.5)',
                    borderColor: colors.secondary,
                    borderWidth: 1
                }}, {{
                    label: 'Median',
                    data: uiSearchBoxplotData.map(d => d.median),
                    backgroundColor: 'rgba(46, 204, 113, 0.7)',
                    borderColor: colors.secondary,
                    borderWidth: 2
                }}, {{
                    label: 'Q3',
                    data: uiSearchBoxplotData.map(d => d.q3),
                    backgroundColor: 'rgba(46, 204, 113, 0.5)',
                    borderColor: colors.secondary,
                    borderWidth: 1
                }}, {{
                    label: 'Max',
                    data: uiSearchBoxplotData.map(d => d.max),
                    backgroundColor: 'rgba(46, 204, 113, 0.3)',
                    borderColor: colors.secondary,
                    borderWidth: 1
                }}]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                plugins: {{
                    title: {{
                        display: true,
                        text: 'UI Search Response Time Distribution by Concurrency'
                    }},
                    tooltip: {{
                        callbacks: {{
                            afterBody: function(context) {{
                                const index = context[0].dataIndex;
                                const data = uiSearchBoxplotData[index];
                                return `Mean: ${{data.mean.toFixed(3)}}s`;
                            }}
                        }}
                    }}
                }},
                scales: {{
                    x: {{
                        title: {{
                            display: true,
                            text: 'Concurrent Request Limit'
                        }}
                    }},
                    y: {{
                        title: {{
                            display: true,
                            text: 'Response Time (seconds)'
                        }}
                    }}
                }}
            }}
        }});

        // UI Search Success Rate Chart
        const uiSearchSuccessCtx = document.getElementById('uiSearchSuccess').getContext('2d');
        new Chart(uiSearchSuccessCtx, {{
            type: 'line',
            data: {{
                labels: {charts["ui_search_success_labels"]},
                datasets: [{{
                    label: 'Success Rate (%)',
                    data: {charts["ui_search_success_data"]},
                    borderColor: colors.info,
                    backgroundColor: 'rgba(155, 89, 182, 0.1)',
                    borderWidth: 3,
                    fill: true,
                    tension: 0.4,
                    pointBackgroundColor: colors.info,
                    pointBorderColor: '#fff',
                    pointBorderWidth: 2,
                    pointRadius: 6
                }}]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                plugins: {{
                    title: {{
                        display: true,
                        text: 'UI Search Success Rate by Concurrency'
                    }}
                }},
                scales: {{
                    x: {{
                        title: {{
                            display: true,
                            text: 'Concurrent Request Limit'
                        }}
                    }},
                    y: {{
                        title: {{
                            display: true,
                            text: 'Success Rate (%)'
                        }},
                        min: 0,
                        max: 100
                    }}
                }}
            }}
        }});
"""

        # Add System Resource charts JavaScript
        if "system_cpu_data" in charts:
            html_template += f"""
        // System Resources Chart
        const systemResourcesCtx = document.getElementById('systemResources').getContext('2d');
        new Chart(systemResourcesCtx, {{
            type: 'bar',
            data: {{
                labels: {charts["system_test_types"]},
                datasets: [{{
                    label: 'CPU Usage (%)',
                    data: {charts["system_cpu_data"]},
                    backgroundColor: 'rgba(231, 76, 60, 0.7)',
                    borderColor: colors.danger,
                    borderWidth: 2
                }}, {{
                    label: 'Memory Usage (%)',
                    data: {charts["system_memory_data"]},
                    backgroundColor: 'rgba(52, 152, 219, 0.7)',
                    borderColor: colors.primary,
                    borderWidth: 2
                }}]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                plugins: {{
                    title: {{
                        display: true,
                        text: 'System Resource Usage by Test Type'
                    }}
                }},
                scales: {{
                    x: {{
                        title: {{
                            display: true,
                            text: 'Test Type'
                        }}
                    }},
                    y: {{
                        title: {{
                            display: true,
                            text: 'Usage (%)'
                        }},
                        min: 0,
                        max: 100
                    }}
                }}
            }}
        }});
"""

        if "system_timeline_cpu" in charts:
            html_template += f"""
        // System Timeline Chart
        const systemTimelineCtx = document.getElementById('systemTimeline').getContext('2d');
        new Chart(systemTimelineCtx, {{
            type: 'line',
            data: {{
                labels: {charts["system_timeline_labels"]},
                datasets: [{{
                    label: 'CPU Usage (%)',
                    data: {charts["system_timeline_cpu"]},
                    borderColor: colors.danger,
                    backgroundColor: 'rgba(231, 76, 60, 0.1)',
                    borderWidth: 3,
                    fill: true,
                    tension: 0.4,
                    pointBackgroundColor: colors.danger,
                    pointBorderColor: '#fff',
                    pointBorderWidth: 2,
                    pointRadius: 6
                }}, {{
                    label: 'Memory Usage (%)',
                    data: {charts["system_timeline_memory"]},
                    borderColor: colors.primary,
                    backgroundColor: 'rgba(52, 152, 219, 0.1)',
                    borderWidth: 3,
                    fill: true,
                    tension: 0.4,
                    pointBackgroundColor: colors.primary,
                    pointBorderColor: '#fff',
                    pointBorderWidth: 2,
                    pointRadius: 6
                }}]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                plugins: {{
                    title: {{
                        display: true,
                        text: 'Resource Usage Timeline'
                    }}
                }},
                scales: {{
                    x: {{
                        title: {{
                            display: true,
                            text: 'Test Phase'
                        }}
                    }},
                    y: {{
                        title: {{
                            display: true,
                            text: 'Usage (%)'
                        }},
                        min: 0,
                        max: 100
                    }}
                }}
            }}
        }});
"""

        # Add Plotly histograms
        if "faiss_histogram_data" in charts:
            html_template += f"""
        // FAISS Histogram (Plotly)
        const faissHistogramData = [{{
            x: {charts["faiss_histogram_labels"]},
            y: {charts["faiss_histogram_data"]},
            type: 'bar',
            name: 'FAISS Response Times',
            marker: {{
                color: 'rgba(52, 152, 219, 0.7)',
                line: {{
                    color: 'rgba(52, 152, 219, 1)',
                    width: 2
                }}
            }}
        }}];
        
        const faissHistogramLayout = {{
            title: 'FAISS Response Time Distribution',
            xaxis: {{ title: 'Response Time (seconds)' }},
            yaxis: {{ title: 'Frequency' }},
            annotations: [
                {{
                    x: {charts.get("faiss_mean_time", 0)},
                    y: Math.max(...{charts["faiss_histogram_data"]}) * 0.8,
                    text: `Mean: {charts.get("faiss_mean_time", 0)}s`,
                    showarrow: true,
                    arrowhead: 2,
                    arrowcolor: 'red',
                    font: {{ color: 'red' }}
                }},
                {{
                    x: {charts.get("faiss_median_time", 0)},
                    y: Math.max(...{charts["faiss_histogram_data"]}) * 0.6,
                    text: `Median: {charts.get("faiss_median_time", 0)}s`,
                    showarrow: true,
                    arrowhead: 2,
                    arrowcolor: 'green',
                    font: {{ color: 'green' }}
                }}
            ]
        }};
        
        Plotly.newPlot('faissHistogram', faissHistogramData, faissHistogramLayout, {{responsive: true}});
"""

        if "ui_histogram_data" in charts:
            html_template += f"""
        // UI Search Histogram (Plotly)
        const uiHistogramData = [{{
            x: {charts["ui_histogram_labels"]},
            y: {charts["ui_histogram_data"]},
            type: 'bar',
            name: 'UI Search Response Times',
            marker: {{
                color: 'rgba(46, 204, 113, 0.7)',
                line: {{
                    color: 'rgba(46, 204, 113, 1)',
                    width: 2
                }}
            }}
        }}];
        
        const uiHistogramLayout = {{
            title: 'UI Search Response Time Distribution',
            xaxis: {{ title: 'Response Time (seconds)' }},
            yaxis: {{ title: 'Frequency' }},
            annotations: [
                {{
                    x: {charts.get("ui_mean_time", 0)},
                    y: Math.max(...{charts["ui_histogram_data"]}) * 0.8,
                    text: `Mean: {charts.get("ui_mean_time", 0)}s`,
                    showarrow: true,
                    arrowhead: 2,
                    arrowcolor: 'red',
                    font: {{ color: 'red' }}
                }},
                {{
                    x: {charts.get("ui_median_time", 0)},
                    y: Math.max(...{charts["ui_histogram_data"]}) * 0.6,
                    text: `Median: {charts.get("ui_median_time", 0)}s`,
                    showarrow: true,
                    arrowhead: 2,
                    arrowcolor: 'green',
                    font: {{ color: 'green' }}
                }}
            ]
        }};
        
        Plotly.newPlot('uiHistogram', uiHistogramData, uiHistogramLayout, {{responsive: true}});
"""

        # Add Performance Comparison charts
        if "comparison_categories" in charts:
            html_template += f"""
        // Performance Comparison - Response Time
        const comparisonResponseTimeCtx = document.getElementById('comparisonResponseTime').getContext('2d');
        new Chart(comparisonResponseTimeCtx, {{
            type: 'bar',
            data: {{
                labels: {charts["comparison_categories"]},
                datasets: [{{
                    label: 'Average Response Time (seconds)',
                    data: {charts["comparison_response_times"]},
                    backgroundColor: ['rgba(52, 152, 219, 0.7)', 'rgba(46, 204, 113, 0.7)'],
                    borderColor: [colors.primary, colors.secondary],
                    borderWidth: 2
                }}]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                plugins: {{
                    title: {{
                        display: true,
                        text: 'Average Response Time Comparison'
                    }},
                    legend: {{
                        display: false
                    }}
                }},
                scales: {{
                    y: {{
                        title: {{
                            display: true,
                            text: 'Response Time (seconds)'
                        }},
                        beginAtZero: true
                    }}
                }}
            }}
        }});

        // Performance Comparison - Success Rate
        const comparisonSuccessRateCtx = document.getElementById('comparisonSuccessRate').getContext('2d');
        new Chart(comparisonSuccessRateCtx, {{
            type: 'bar',
            data: {{
                labels: {charts["comparison_categories"]},
                datasets: [{{
                    label: 'Success Rate (%)',
                    data: {charts["comparison_success_rates"]},
                    backgroundColor: ['rgba(155, 89, 182, 0.7)', 'rgba(243, 156, 18, 0.7)'],
                    borderColor: [colors.info, colors.warning],
                    borderWidth: 2
                }}]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                plugins: {{
                    title: {{
                        display: true,
                        text: 'Success Rate Comparison'
                    }},
                    legend: {{
                        display: false
                    }}
                }},
                scales: {{
                    y: {{
                        title: {{
                            display: true,
                            text: 'Success Rate (%)'
                        }},
                        min: 0,
                        max: 100
                    }}
                }}
            }}
        }});
"""

        html_template += """
    </script>
</body>
</html>
"""
        
        return html_template
    
    def save_html_report(self, html_content: str) -> str:
        """Save HTML report to file."""
        report_file = self.output_dir / "stress_test_report.html"
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"📋 HTML Report saved to: {report_file}")
        return str(report_file)
    
    def save_results(self):
        """Save all results to JSON file."""
        results_file = self.output_dir / "stress_test_results.json"
        
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"💾 Results saved to: {results_file}")
        return str(results_file)
    
    async def run_complete_test(self, 
                              num_mock_servers: int = 50,
                              concurrent_limits: List[int] = [5, 10, 20],
                              skip_mock_registration: bool = False,
                              skip_faiss: bool = False,
                              headless_ui: bool = True):
        """Run the complete stress test suite."""
        print("Starting Complete MCP Stress Test Suite")
        print("=" * 60)
        
        start_time = time.time()
        
        try:
            # Step 1: Generate and register mock servers (if not skipping)
            if not skip_mock_registration:
                servers = self.generate_mock_server_data(num_mock_servers)
                await self.register_mock_servers(servers)
                
                # Wait for FAISS index to update
                print("⏳ Waiting 10 seconds for FAISS index to update...")
                await asyncio.sleep(10)
            else:
                print("⏩ Skipping mock server registration")
            
            # Step 2: Test FAISS performance
            if not skip_faiss:
                await self.test_faiss_performance(concurrent_limits)
            
            # Step 3: Test UI performance
            self.test_ui_performance(headless_ui)
            
            # Step 4: Test UI search concurrency
            ui_search_results = await self.test_ui_search_concurrency(concurrent_limits)
            self.results["ui_search_results"] = ui_search_results
            
            # Step 5: Generate charts and reports
            charts = self.generate_performance_charts()
            html_content = self.generate_html_report(charts)
            report_path = self.save_html_report(html_content)
            results_path = self.save_results()
            
            # Final summary
            total_time = time.time() - start_time
            print("\n" + "=" * 60)
            print("Complete Stress Test Finished!")
            print(f"Total time: {total_time:.1f} seconds")
            print(f"📋 HTML Report: {report_path}")
            print(f"💾 JSON Results: {results_path}")
            print("=" * 60)
            
            return {
                "success": True,
                "report_path": report_path,
                "results_path": results_path,
                "total_time": total_time
            }
            
        except Exception as e:
            print(f"Test suite failed: {e}")
            return {"success": False, "error": str(e)}

    def add_tools_to_existing_mocks(self) -> int:
        """Add realistic tool lists to existing mock servers."""
        print("Adding tool lists to existing mock servers...")
        
        mock_server_dir = Path("/home/ubuntu/mcp-gateway-data/servers/")
        mock_files = list(mock_server_dir.glob("mock-server-*.json"))
        
        if not mock_files:
            print("No existing mock server files found")
            return 0
        
        print(f"Found {len(mock_files)} mock server files")
        
        # Tool templates for different server types
        tool_templates = {
            "database": [
                {"name": "query_data", "desc": "Execute database queries", "params": ["query", "limit"]},
                {"name": "insert_record", "desc": "Insert new records", "params": ["table", "data"]},
                {"name": "update_record", "desc": "Update existing records", "params": ["table", "id", "data"]},
                {"name": "delete_record", "desc": "Delete records", "params": ["table", "id"]},
                {"name": "create_table", "desc": "Create new tables", "params": ["name", "schema"]},
                {"name": "backup_database", "desc": "Create database backup", "params": ["target"]},
                {"name": "optimize_tables", "desc": "Optimize table performance", "params": ["tables"]},
                {"name": "analyze_performance", "desc": "Analyze query performance", "params": ["timeframe"]},
            ],
            "analytics": [
                {"name": "generate_report", "desc": "Generate analytics reports", "params": ["type", "filters"]},
                {"name": "calculate_metrics", "desc": "Calculate key metrics", "params": ["dataset", "metrics"]},
                {"name": "trend_analysis", "desc": "Perform trend analysis", "params": ["data", "period"]},
                {"name": "correlation_analysis", "desc": "Find data correlations", "params": ["variables"]},
                {"name": "forecast_data", "desc": "Generate forecasts", "params": ["model", "horizon"]},
                {"name": "segment_users", "desc": "Segment user data", "params": ["criteria"]},
                {"name": "ab_test_analysis", "desc": "Analyze A/B test results", "params": ["test_id"]},
                {"name": "cohort_analysis", "desc": "Perform cohort analysis", "params": ["cohort_def"]},
            ],
            "machine_learning": [
                {"name": "train_model", "desc": "Train ML models", "params": ["algorithm", "data", "params"]},
                {"name": "predict", "desc": "Make predictions", "params": ["model", "input"]},
                {"name": "evaluate_model", "desc": "Evaluate model performance", "params": ["model", "test_data"]},
                {"name": "feature_selection", "desc": "Select optimal features", "params": ["data", "target"]},
                {"name": "hyperparameter_tuning", "desc": "Optimize hyperparameters", "params": ["model", "search_space"]},
                {"name": "deploy_model", "desc": "Deploy model to production", "params": ["model", "endpoint"]},
                {"name": "monitor_drift", "desc": "Monitor model drift", "params": ["model", "baseline"]},
                {"name": "explain_prediction", "desc": "Explain model predictions", "params": ["model", "instance"]},
            ],
            "security": [
                {"name": "scan_vulnerabilities", "desc": "Scan for security vulnerabilities", "params": ["target", "scan_type"]},
                {"name": "authenticate_user", "desc": "Authenticate user credentials", "params": ["username", "password"]},
                {"name": "authorize_access", "desc": "Check access permissions", "params": ["user", "resource"]},
                {"name": "encrypt_data", "desc": "Encrypt sensitive data", "params": ["data", "algorithm"]},
                {"name": "audit_logs", "desc": "Analyze security audit logs", "params": ["timeframe", "filters"]},
                {"name": "detect_threats", "desc": "Detect security threats", "params": ["data_source"]},
                {"name": "generate_keys", "desc": "Generate cryptographic keys", "params": ["key_type", "length"]},
                {"name": "compliance_check", "desc": "Check regulatory compliance", "params": ["standard"]},
            ],
            "monitoring": [
                {"name": "collect_metrics", "desc": "Collect system metrics", "params": ["targets", "interval"]},
                {"name": "create_alert", "desc": "Create monitoring alerts", "params": ["condition", "notification"]},
                {"name": "check_health", "desc": "Perform health checks", "params": ["services"]},
                {"name": "analyze_logs", "desc": "Analyze application logs", "params": ["source", "filters"]},
                {"name": "track_performance", "desc": "Track performance metrics", "params": ["metrics", "period"]},
                {"name": "generate_dashboard", "desc": "Generate monitoring dashboard", "params": ["layout", "widgets"]},
                {"name": "detect_anomalies", "desc": "Detect metric anomalies", "params": ["baseline", "threshold"]},
                {"name": "capacity_planning", "desc": "Plan resource capacity", "params": ["usage_data", "growth_rate"]},
            ],
        }
        
        updated_count = 0
        total_tools_added = 0
        
        for mock_file in mock_files:
            try:
                # Read existing server data
                with open(mock_file, 'r') as f:
                    server_data = json.load(f)
                
                # Skip if already has tools
                if server_data.get("tool_list") and len(server_data["tool_list"]) > 0:
                    continue
                
                # Determine server type from tags or name
                server_type = "database"  # default
                tags = server_data.get("tags", [])
                for tag in tags:
                    if tag in tool_templates:
                        server_type = tag
                        break
                
                # Generate tools based on the claimed num_tools
                num_tools = server_data.get("num_tools", 50)
                available_templates = tool_templates[server_type]
                
                tool_list = []
                for i in range(num_tools):
                    # Cycle through available templates and add variations
                    template = available_templates[i % len(available_templates)]
                    
                    tool = {
                        "name": f"{template['name']}_{i//len(available_templates) + 1}" if i >= len(available_templates) else template['name'],
                        "description": f"{template['desc']} (variant {i//len(available_templates) + 1})" if i >= len(available_templates) else template['desc'],
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                param: {
                                    "type": "string", 
                                    "description": f"Parameter for {param}"
                                } for param in template['params']
                            },
                            "required": template['params'][:2] if len(template['params']) >= 2 else template['params']
                        }
                    }
                    tool_list.append(tool)
                
                # Update server data
                server_data["tool_list"] = tool_list
                
                # Write back to file
                with open(mock_file, 'w') as f:
                    json.dump(server_data, f, indent=2)
                
                updated_count += 1
                total_tools_added += len(tool_list)
                
                if updated_count % 50 == 0:
                    print(f"  📝 Updated {updated_count}/{len(mock_files)} servers...")
                
            except Exception as e:
                print(f"Error updating {mock_file}: {e}")
                continue
        
        print(f"Updated {updated_count} mock servers with {total_tools_added} total tools")
        return updated_count

    def cleanup_mock_servers(self) -> int:
        """Cleanup all mock servers from the shared volume."""
        print("🧹 Cleaning up all mock servers...")
        
        # Find all mock server files
        mock_server_dir = Path("/home/ubuntu/mcp-gateway-data/servers/")
        mock_files = list(mock_server_dir.glob("mock-*.json"))
        
        print(f"Found {len(mock_files)} mock server files to remove")
        
        removed_count = 0
        for mock_file in mock_files:
            try:
                mock_file.unlink()
                removed_count += 1
            except Exception as e:
                print(f"Could not remove {mock_file}: {e}")
        
        # Also cleanup FAISS index and server state
        try:
            faiss_index = mock_server_dir / "service_index.faiss"
            if faiss_index.exists():
                faiss_index.unlink()
                print("Removed FAISS index")
            
            faiss_metadata = mock_server_dir / "service_index_metadata.json"
            if faiss_metadata.exists():
                faiss_metadata.unlink()
                print("Removed FAISS metadata")
            
            server_state = mock_server_dir / "server_state.json"
            if server_state.exists():
                server_state.unlink()
                print("Removed server state")
                
        except Exception as e:
            print(f"Error cleaning up index files: {e}")
        
        print(f"Removed {removed_count} mock server files")
        return removed_count

async def main():
    parser = argparse.ArgumentParser(description="Complete MCP Gateway Registry Stress Test")
    parser.add_argument("--servers", type=int, default=50, help="Number of mock servers to create")
    parser.add_argument("--concurrency", type=str, default="5,10,20", help="Comma-separated concurrency limits")
    parser.add_argument("--skip-mock", action="store_true", help="Skip mock server registration")
    parser.add_argument("--skip-faiss", action="store_true", help="Skip FAISS performance testing")
    parser.add_argument("--add-tools", action="store_true", help="Add tool lists to existing mock servers")
    parser.add_argument("--cleanup", action="store_true", help="Cleanup all mock servers and restart fresh")
    parser.add_argument("--show-ui", action="store_true", help="Show browser during UI tests")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory (defaults to script directory)")
    
    args = parser.parse_args()
    
    # Create test suite
    output_dir = Path(args.output_dir) if args.output_dir else None
    test_suite = CompleteMCPStressTest(output_dir)
    
    # Cleanup if requested
    if args.cleanup:
        print("🧹 Cleanup mode - removing all mock servers...")
        removed_count = test_suite.cleanup_mock_servers()
        if removed_count > 0:
            print("🔄 Restarting Docker container to reset state...")
            import subprocess
            try:
                subprocess.run(["docker", "restart", "mcp-gateway-container"], check=True)
                print("Container restarted successfully")
                print("Wait a few seconds for services to start, then run the stress test again")
            except subprocess.CalledProcessError as e:
                print(f"Failed to restart container: {e}")
        return
    
    # Parse concurrency limits
    concurrent_limits = [int(x.strip()) for x in args.concurrency.split(",")]
    
    # Add tools to existing mocks if requested
    if args.add_tools:
        print("Adding tools to existing mock servers...")
        updated_count = test_suite.add_tools_to_existing_mocks()
        if updated_count > 0:
            print(f"Updated {updated_count} mock servers. Restarting container to reload FAISS...")
            # Note: Container restart would need to be done manually or via docker command
            print("Run: docker restart mcp-gateway-container")
            print("   Then re-run the stress test to see the new tools in action!")
        return
    
    # Run complete test
    result = await test_suite.run_complete_test(
        num_mock_servers=args.servers,
        concurrent_limits=concurrent_limits,
        skip_mock_registration=args.skip_mock,
        skip_faiss=args.skip_faiss,
        headless_ui=not args.show_ui
    )
    
    if result["success"]:
        print(f"\nOpen the report: {result['report_path']}")
    else:
        print(f"\nTest failed: {result.get('error', 'Unknown error')}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main()) 