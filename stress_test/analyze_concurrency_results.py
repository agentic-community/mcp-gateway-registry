#!/usr/bin/env python3
"""
Comprehensive analysis of concurrency test results.
"""

import json
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime

def analyze_faiss_results(faiss_results):
    """Analyze FAISS performance results in detail."""
    if not faiss_results:
        print("No FAISS results found.")
        return {}
    
    df = pd.DataFrame(faiss_results)
    
    print("🔍 FAISS Performance Analysis")
    print("=" * 70)
    
    # Overall statistics
    total_queries = len(df)
    successful_queries = df['success'].sum()
    overall_success_rate = (successful_queries / total_queries * 100) if total_queries > 0 else 0
    
    print(f"\n📊 Overall Statistics:")
    print(f"• Total queries tested: {total_queries}")
    print(f"• Successful queries: {successful_queries}")
    print(f"• Overall success rate: {overall_success_rate:.1f}%")
    
    # Performance by concurrency level
    print(f"\n📈 Performance by Concurrency Level:")
    print("-" * 70)
    print(f"{'Concurrency':>11} {'Success':>7} {'Rate':>6} {'Avg Time':>10} {'Min':>7} {'Max':>7} {'Std':>7} {'CPU':>6} {'RAM':>6}")
    print("-" * 70)
    
    concurrency_stats = {}
    for limit in sorted(df['concurrent_limit'].unique()):
        batch = df[df['concurrent_limit'] == limit]
        successful = batch['success'].sum()
        total = len(batch)
        success_rate = (successful / total * 100) if total > 0 else 0
        
        successful_times = batch[batch['success'] == True]['elapsed_time']
        if len(successful_times) > 0:
            avg_time = successful_times.mean()
            min_time = successful_times.min()
            max_time = successful_times.max()
            std_time = successful_times.std()
        else:
            avg_time = min_time = max_time = std_time = 0
        
        # System metrics
        system_metrics = batch[batch['system_metrics'].notna()]['system_metrics'].iloc[0] if 'system_metrics' in batch.columns and len(batch[batch['system_metrics'].notna()]) > 0 else {}
        peak_cpu = system_metrics.get('peak_cpu_percent', 0) if system_metrics else 0
        peak_memory = system_metrics.get('peak_memory_percent', 0) if system_metrics else 0
        
        print(f"{limit:>11d} {successful:>2d}/{total:>2d} {success_rate:>5.1f}% {avg_time:>9.3f}s {min_time:>6.3f}s {max_time:>6.3f}s {std_time:>6.3f}s {peak_cpu:>5.1f}% {peak_memory:>5.1f}%")
        
        concurrency_stats[limit] = {
            'success_rate': success_rate,
            'avg_time': avg_time,
            'min_time': min_time,
            'max_time': max_time,
            'std_time': std_time,
            'peak_cpu': peak_cpu,
            'peak_memory': peak_memory,
            'successful': successful,
            'total': total
        }
    
    # Performance trends analysis
    print(f"\n📊 Performance Trends Analysis:")
    print("-" * 70)
    
    concurrency_levels = sorted(df['concurrent_limit'].unique())
    if len(concurrency_levels) >= 2:
        baseline = concurrency_stats[concurrency_levels[0]]
        peak_load = concurrency_stats[concurrency_levels[-1]]
        
        response_time_change = ((peak_load['avg_time'] - baseline['avg_time']) / baseline['avg_time'] * 100) if baseline['avg_time'] > 0 else 0
        success_rate_change = peak_load['success_rate'] - baseline['success_rate']
        
        print(f"• Response time degradation: {response_time_change:+.1f}% ({baseline['avg_time']:.3f}s → {peak_load['avg_time']:.3f}s)")
        print(f"• Success rate change: {success_rate_change:+.1f}% ({baseline['success_rate']:.1f}% → {peak_load['success_rate']:.1f}%)")
        
        # Find performance sweet spot
        best_performance = min(concurrency_stats.items(), key=lambda x: x[1]['avg_time'] if x[1]['avg_time'] > 0 else float('inf'))
        worst_performance = max(concurrency_stats.items(), key=lambda x: x[1]['avg_time'])
        
        print(f"• Best performance: {best_performance[1]['avg_time']:.3f}s at {best_performance[0]} concurrent")
        print(f"• Worst performance: {worst_performance[1]['avg_time']:.3f}s at {worst_performance[0]} concurrent")
        
        # System stability
        cpu_variance = np.var([stats['peak_cpu'] for stats in concurrency_stats.values()])
        memory_variance = np.var([stats['peak_memory'] for stats in concurrency_stats.values()])
        
        print(f"• CPU usage variance: {cpu_variance:.2f} (lower is more stable)")
        print(f"• Memory usage variance: {memory_variance:.2f} (lower is more stable)")
    
    # Query-specific analysis
    print(f"\n🎯 Query Performance Analysis:")
    print("-" * 70)
    
    query_performance = df.groupby('query').agg({
        'success': ['count', 'sum', 'mean'],
        'elapsed_time': ['mean', 'std', 'min', 'max'],
        'tools_found': 'mean'
    }).round(3)
    
    query_performance.columns = ['total', 'successful', 'success_rate', 'avg_time', 'std_time', 'min_time', 'max_time', 'avg_tools']
    query_performance['success_rate'] *= 100
    
    # Show top and bottom performing queries
    successful_queries = query_performance[query_performance['successful'] > 0].sort_values('avg_time')
    
    if len(successful_queries) > 0:
        print(f"🏆 Fastest queries:")
        for i, (query, stats) in enumerate(successful_queries.head(3).iterrows()):
            print(f"  {i+1}. {query[:50]}... → {stats['avg_time']:.3f}s avg, {stats['avg_tools']:.1f} tools")
        
        print(f"\n🐌 Slowest queries:")
        for i, (query, stats) in enumerate(successful_queries.tail(3).iterrows()):
            print(f"  {i+1}. {query[:50]}... → {stats['avg_time']:.3f}s avg, {stats['avg_tools']:.1f} tools")
    
    # Tool finding effectiveness
    print(f"\n🔧 Tool Discovery Analysis:")
    print("-" * 70)
    
    tool_stats = df.groupby('tools_found').size().sort_index()
    print(f"• Tools found distribution:")
    for tools, count in tool_stats.items():
        percentage = (count / len(df) * 100)
        print(f"  {tools} tools: {count} queries ({percentage:.1f}%)")
    
    avg_tools_found = df[df['success'] == True]['tools_found'].mean() if successful_queries > 0 else 0
    print(f"• Average tools found per successful query: {avg_tools_found:.1f}")
    
    return {
        'total_queries': total_queries,
        'successful_queries': successful_queries,
        'overall_success_rate': overall_success_rate,
        'concurrency_stats': concurrency_stats,
        'query_performance': query_performance.to_dict('index'),
        'avg_tools_found': avg_tools_found
    }

def analyze_ui_results(ui_results):
    """Analyze UI performance results in detail."""
    if not ui_results:
        print("No UI results found.")
        return {}
    
    print("\n🌐 UI Performance Analysis")
    print("=" * 70)
    
    # Basic performance metrics
    print(f"\n⚡ Core Performance Metrics:")
    print(f"• Login time: {ui_results.get('login_time', 0):.3f}s")
    print(f"• Page load time: {ui_results.get('page_load_time', 0):.3f}s")
    print(f"• Servers detected: {ui_results.get('servers_found', 0)}")
    
    # Element analysis
    if 'element_tests' in ui_results:
        print(f"\n🔍 Page Element Analysis:")
        for element_test in ui_results['element_tests']:
            print(f"• {element_test['element'].capitalize()}: {element_test['count']}")
    
    # Search functionality analysis
    if 'search_results' in ui_results and ui_results['search_results']:
        print(f"\n🔎 Search Functionality Analysis:")
        search_results = ui_results['search_results']
        
        successful_searches = [r for r in search_results if r.get('success', False)]
        total_searches = len(search_results)
        search_success_rate = (len(successful_searches) / total_searches * 100) if total_searches > 0 else 0
        
        print(f"• Total search tests: {total_searches}")
        print(f"• Successful searches: {len(successful_searches)} ({search_success_rate:.1f}%)")
        
        if successful_searches:
            avg_search_time = np.mean([r['search_time'] for r in successful_searches])
            avg_results_found = np.mean([r['results_found'] for r in successful_searches])
            
            print(f"• Average search time: {avg_search_time:.3f}s")
            print(f"• Average results per search: {avg_results_found:.1f}")
            
            print(f"\n  Search Performance Details:")
            for i, result in enumerate(search_results, 1):
                status = "✓" if result.get('success', False) else "✗"
                query_short = result['query'][:40] + "..." if len(result['query']) > 40 else result['query']
                print(f"  {i}. {status} {query_short}")
                print(f"     Time: {result['search_time']:.3f}s, Results: {result['results_found']}")
                if 'error' in result:
                    print(f"     Error: {result['error']}")
    
    # Navigation tests
    if 'navigation_tests' in ui_results and ui_results['navigation_tests']:
        print(f"\n🧭 Navigation Performance:")
        nav_tests = ui_results['navigation_tests']
        
        successful_nav = [n for n in nav_tests if n.get('success', False)]
        total_nav = len(nav_tests)
        nav_success_rate = (len(successful_nav) / total_nav * 100) if total_nav > 0 else 0
        
        print(f"• Navigation success rate: {len(successful_nav)}/{total_nav} ({nav_success_rate:.1f}%)")
        
        if successful_nav:
            avg_nav_time = np.mean([n['load_time'] for n in successful_nav])
            print(f"• Average page load time: {avg_nav_time:.3f}s")
        
        print(f"\n  Page Load Details:")
        for nav in nav_tests:
            status = "✓" if nav.get('success', False) else "✗"
            load_time = nav['load_time'] if nav.get('success', False) else 0
            print(f"  {status} {nav['page']:<8} ({nav['url']:<10}): {load_time:.3f}s")
            if 'error' in nav:
                print(f"     Error: {nav['error']}")
    
    # JavaScript performance
    if 'js_performance' in ui_results:
        print(f"\n⚡ JavaScript Performance:")
        js_perf = ui_results['js_performance']
        
        if 'timing' in js_perf:
            timing = js_perf['timing']
            dom_load = (timing.get('domContentLoadedEventEnd', 0) - timing.get('navigationStart', 0)) / 1000
            page_load = (timing.get('loadEventEnd', 0) - timing.get('navigationStart', 0)) / 1000
            
            print(f"• DOM content loaded: {dom_load:.3f}s")
            print(f"• Page fully loaded: {page_load:.3f}s")
        
        if 'memory' in js_perf and js_perf['memory']:
            memory = js_perf['memory']
            memory_mb = memory.get('usedJSHeapSize', 0) / (1024 * 1024)
            print(f"• JavaScript heap size: {memory_mb:.1f} MB")
    
    # Error analysis
    if 'errors' in ui_results and ui_results['errors']:
        print(f"\n❌ Error Analysis:")
        for i, error in enumerate(ui_results['errors'], 1):
            print(f"  {i}. {error}")
    
    return {
        'login_time': ui_results.get('login_time', 0),
        'page_load_time': ui_results.get('page_load_time', 0),
        'servers_found': ui_results.get('servers_found', 0),
        'search_success_rate': search_success_rate if 'search_results' in ui_results else 0,
        'navigation_success_rate': nav_success_rate if 'navigation_tests' in ui_results else 0,
        'total_errors': len(ui_results.get('errors', []))
    }

def generate_comprehensive_summary(faiss_analysis, ui_analysis, metadata):
    """Generate a comprehensive summary of all test results."""
    print("\n" + "=" * 70)
    print("🎯 COMPREHENSIVE TEST SUMMARY")
    print("=" * 70)
    
    # Test overview
    print(f"\n📋 Test Overview:")
    print(f"• Test started: {metadata.get('test_start', 'Unknown')}")
    print(f"• Registry URL: {metadata.get('registry_url', 'Unknown')}")
    print(f"• MCP Gateway URL: {metadata.get('mcpgw_url', 'Unknown')}")
    
    # Performance grades
    print(f"\n📊 Performance Grades:")
    
    # FAISS grade
    if faiss_analysis:
        faiss_grade = "A" if faiss_analysis['overall_success_rate'] >= 80 else \
                     "B" if faiss_analysis['overall_success_rate'] >= 60 else \
                     "C" if faiss_analysis['overall_success_rate'] >= 40 else "D"
        print(f"• FAISS Performance: {faiss_grade} ({faiss_analysis['overall_success_rate']:.1f}% success rate)")
    
    # UI grade
    if ui_analysis:
        ui_score = 100
        if ui_analysis['login_time'] > 5: ui_score -= 20
        if ui_analysis['page_load_time'] > 3: ui_score -= 20
        if ui_analysis['search_success_rate'] < 80: ui_score -= 30
        if ui_analysis['total_errors'] > 0: ui_score -= 15
        
        ui_grade = "A" if ui_score >= 80 else "B" if ui_score >= 60 else "C" if ui_score >= 40 else "D"
        print(f"• UI Performance: {ui_grade} (Score: {ui_score}/100)")
    
    # Recommendations
    print(f"\n💡 Recommendations:")
    
    if faiss_analysis:
        if faiss_analysis['overall_success_rate'] < 80:
            print("• ⚠️  Consider optimizing FAISS indexing or query processing")
        
        # Check for performance degradation
        concurrency_stats = faiss_analysis.get('concurrency_stats', {})
        if len(concurrency_stats) >= 2:
            levels = sorted(concurrency_stats.keys())
            baseline_time = concurrency_stats[levels[0]]['avg_time']
            peak_time = concurrency_stats[levels[-1]]['avg_time']
            degradation = ((peak_time - baseline_time) / baseline_time * 100) if baseline_time > 0 else 0
            
            if degradation > 200:
                print("• ❌ High performance degradation under load - consider load balancing")
            elif degradation > 100:
                print("• ⚠️  Moderate performance degradation - monitor under production load")
            else:
                print("• ✅ Good performance stability under high concurrency")
    
    if ui_analysis:
        if ui_analysis['page_load_time'] > 3:
            print("• ⚠️  Page load time is slow - consider optimizing static assets")
        
        if ui_analysis['search_success_rate'] < 100:
            print("• ⚠️  Search functionality has issues - check UI selectors and JavaScript")
        
        if ui_analysis['total_errors'] > 0:
            print("• ❌ UI errors detected - review error logs for specific issues")
    
    # System readiness assessment
    print(f"\n🚀 System Readiness Assessment:")
    
    overall_readiness = "Production Ready"
    
    if faiss_analysis and faiss_analysis['overall_success_rate'] < 60:
        overall_readiness = "Needs Optimization"
    
    if ui_analysis and (ui_analysis['search_success_rate'] < 80 or ui_analysis['total_errors'] > 2):
        overall_readiness = "Needs Optimization"
    
    if overall_readiness == "Production Ready":
        print("• ✅ System appears ready for production deployment")
        print("• ✅ Performance is within acceptable limits")
        print("• ✅ No critical issues detected")
    else:
        print("• ⚠️  System needs optimization before production")
        print("• ⚠️  Address identified issues and re-test")
    
    print("\n" + "=" * 70)

def analyze_results(results_file: Path):
    """Analyze the complete test results."""
    
    with open(results_file, 'r') as f:
        data = json.load(f)
    
    print("🚀 COMPREHENSIVE MCP STRESS TEST ANALYSIS")
    print("=" * 70)
    print(f"📁 Results file: {results_file}")
    print(f"🕒 Analysis time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Analyze each component
    faiss_analysis = analyze_faiss_results(data.get("faiss_results", []))
    ui_analysis = analyze_ui_results(data.get("ui_results", {}))
    
    # Generate comprehensive summary
    generate_comprehensive_summary(faiss_analysis, ui_analysis, data.get("metadata", {}))

def main():
    parser = argparse.ArgumentParser(description="Comprehensive analysis of MCP stress test results")
    parser.add_argument("results_file", help="Path to the JSON results file")
    
    args = parser.parse_args()
    
    results_file = Path(args.results_file)
    if not results_file.exists():
        print(f"Results file not found: {results_file}")
        return
    
    analyze_results(results_file)

if __name__ == "__main__":
    main() 