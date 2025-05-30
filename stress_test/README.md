# MCP Gateway Registry Stress Test Suite

A comprehensive stress testing tool for the MCP Gateway Registry that tests FAISS search performance, UI performance, and generates interactive performance reports.

## 🚀 Features

- **FAISS Performance Testing**: Test intelligent tool search with various concurrency levels
- **UI Performance Testing**: Automated browser testing of the registry web interface
- **Interactive Charts**: Beautiful, interactive charts using Chart.js and Plotly
- **System Monitoring**: Real-time CPU and memory usage tracking during tests
- **Mock Server Management**: Create and manage mock MCP servers for testing
- **Comprehensive Reports**: Generate detailed HTML reports with performance analytics

## 📋 Prerequisites

- Python 3.8+
- `uv` package manager ([Install uv](https://docs.astral.sh/uv/getting-started/installation/))
- Chrome/Chromium browser (for UI testing)
- Access to MCP Gateway Registry instance

## 🛠️ Installation

The stress test uses `uv` for dependency management. All dependencies will be automatically installed when you run the script.

```bash
# Clone or navigate to the stress test directory
cd stress_test

# Run with uv (dependencies will be auto-installed)
uv run complete_mcp_stress_test.py --help
```

## 🎯 Usage

### Basic Usage

```bash
# Run complete stress test with default settings
uv run complete_mcp_stress_test.py

# Run with custom concurrency levels
uv run complete_mcp_stress_test.py --concurrency "5,10,20,50"

# Skip mock server creation and use existing servers
uv run complete_mcp_stress_test.py --skip-mock

# Skip FAISS testing (UI testing only)
uv run complete_mcp_stress_test.py --skip-faiss
```

### Advanced Options

```bash
# Create 100 mock servers for testing
uv run complete_mcp_stress_test.py --servers 100

# Show browser during UI testing (non-headless)
uv run complete_mcp_stress_test.py --show-ui

# Custom output directory
uv run complete_mcp_stress_test.py --output-dir ./my-test-results

# High-concurrency stress test
uv run complete_mcp_stress_test.py --concurrency "10,25,50,100,200" --skip-mock
```

### Management Commands

```bash
# Add realistic tool lists to existing mock servers
uv run complete_mcp_stress_test.py --add-tools

# Cleanup all mock servers and reset
uv run complete_mcp_stress_test.py --cleanup
```

## 📊 Command Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `--servers` | Number of mock servers to create | 50 |
| `--concurrency` | Comma-separated concurrency limits | "5,10,20" |
| `--skip-mock` | Skip mock server registration | False |
| `--skip-faiss` | Skip FAISS performance testing | False |
| `--add-tools` | Add tool lists to existing mock servers | False |
| `--cleanup` | Cleanup all mock servers | False |
| `--show-ui` | Show browser during UI tests | False |
| `--output-dir` | Custom output directory | Script directory |

## 📈 Output Files

The stress test generates results in a timestamped directory:

```
stress_test/results-YYYYMMDD_HHMMSS/
├── stress_test_report.html     # Interactive HTML report
└── stress_test_results.json    # Raw test data
```

### Interactive HTML Report

The HTML report includes:

- **📊 Executive Summary**: Key performance metrics and success rates
- **🔍 FAISS Performance**: Response time distributions and success rates by concurrency
- **🖥️ UI Performance**: Login times, page loads, element detection
- **🔎 UI Search Analysis**: Concurrent search performance testing
- **💻 System Resources**: CPU and memory usage during tests
- **📈 Response Time Distribution**: Statistical analysis with histograms
- **⚖️ Performance Comparison**: Direct FAISS vs UI search comparison

### Chart Types

- **Interactive Box Plots**: Response time distributions (Chart.js)
- **Line Charts**: Success rates and timelines
- **Doughnut Charts**: Performance metric breakdowns
- **Polar Area Charts**: UI element analysis
- **Radar Charts**: Navigation performance
- **Histograms**: Response time distributions (Plotly)
- **Bar Charts**: Resource usage and comparisons

## 🏃‍♂️ Quick Start Examples

### 1. Test Existing Setup
```bash
# Test with existing servers, moderate concurrency
uv run complete_mcp_stress_test.py --concurrency "5,10,15" --skip-mock
```

### 2. High-Performance Stress Test
```bash
# Extreme concurrency testing
uv run complete_mcp_stress_test.py --concurrency "50,100,200,500" --skip-mock --skip-faiss
```

### 3. Complete Fresh Test
```bash
# Full test with new mock servers
uv run complete_mcp_stress_test.py --servers 75 --concurrency "10,20,40"
```

### 4. UI-Only Testing
```bash
# Focus on UI performance with visible browser
uv run complete_mcp_stress_test.py --skip-faiss --skip-mock --show-ui
```

## 🔧 Configuration

### Environment Variables

```bash
export REGISTRY_URL="http://localhost:7860"
export MCPGW_SERVER_URL="http://your-server.com/mcpgw/sse"
export ADMIN_USER="admin"
export ADMIN_PASSWORD="your-password"
```

### Test Queries

The stress test uses a predefined set of queries optimized for testing:
- Financial data queries
- Time/timezone queries  
- Machine learning queries
- System administration queries
- Mock data generation queries

## 📋 Understanding Results

### Success Rate Indicators
- 🟢 **90%+**: Excellent performance
- 🟡 **70-90%**: Good performance  
- 🔴 **<70%**: Performance issues

### Key Metrics
- **Response Time**: Average time for successful requests
- **Concurrency Performance**: How well the system handles multiple simultaneous requests
- **Resource Usage**: CPU and memory consumption during tests
- **Error Rates**: Failed requests and their causes

## 🐛 Troubleshooting

### Common Issues

1. **Chrome Driver Not Found**
   ```bash
   # Install Chrome/Chromium
   sudo apt-get install chromium-browser
   ```

2. **Permission Errors**
   ```bash
   # Ensure proper permissions for output directory
   chmod 755 stress_test/
   ```

3. **Connection Timeouts**
   - Check REGISTRY_URL is accessible
   - Verify network connectivity
   - Increase timeout values in script if needed

4. **Memory Issues with High Concurrency**
   - Reduce concurrency levels
   - Monitor system resources
   - Use `--skip-mock` to reduce memory usage

### Debug Mode

For detailed debugging, modify the script to enable verbose logging or run with specific test components:

```bash
# Test only FAISS with low concurrency
uv run complete_mcp_stress_test.py --concurrency "2,3" --skip-mock

# Test only UI components
uv run complete_mcp_stress_test.py --skip-faiss --skip-mock --show-ui
```

## 🤝 Contributing

To add new test scenarios or improve the stress test:

1. Fork the repository
2. Add your test methods to the `CompleteMCPStressTest` class
3. Update chart generation in `generate_performance_charts()`
4. Add new chart sections to `generate_html_report()`
5. Test with `uv run` and submit a PR

## 📄 License

This stress test suite is part of the MCP Gateway Registry project. 