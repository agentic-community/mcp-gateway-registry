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
    
    print("FAISS Performance Analysis")
    print("=" * 70)
    
    # Overall statistics
    total_queries = len(df)
    successful_queries = df['success'].sum()
    overall_success_rate = (successful_queries / total_queries * 100) if total_queries > 0 else 0
    
    print(f"\nOverall Statistics:")
    print(f"• Total queries tested: {total_queries}")
    print(f"• Successful queries: {successful_queries}")
    print(f"• Overall success rate: {overall_success_rate:.1f}%")
    
    # Performance by concurrency level
    print(f"\nPerformance by Concurrency Level:")
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
            
        # Get CPU and memory usage for this batch
        cpu_usage = batch['cpu_usage_percent'].mean() if 'cpu_usage_percent' in batch and not batch['cpu_usage_percent'].isna().all() else 0
        memory_usage = batch['memory_usage_mb'].mean() if 'memory_usage_mb' in batch and not batch['memory_usage_mb'].isna().all() else 0
        
        print(f"{limit:>11} {successful:>7} {success_rate:>5.1f}% {avg_time:>9.3f}s {min_time:>6.3f}s {max_time:>6.3f}s {std_time:>6.3f}s {cpu_usage:>5.1f}% {memory_usage:>5.0f}MB")
        
        concurrency_stats[limit] = {
            'success_rate': success_rate,
            'avg_time': avg_time,
            'min_time': min_time,
            'max_time': max_time,
            'std_time': std_time,
            'cpu_usage': cpu_usage,
            'memory_usage': memory_usage,
            'total_queries': total,
            'successful_queries': successful
        }
    
    # Performance trends analysis
    print(f"\nPerformance Trends Analysis:")
    print("-" * 70)
    
    # Calculate performance degradation
    base_concurrency = min(concurrency_stats.keys())
    base_avg_time = concurrency_stats[base_concurrency]['avg_time']
    base_success_rate = concurrency_stats[base_concurrency]['success_rate']
    
    max_time_degradation = 0
    max_success_degradation = 0
    
    for limit in sorted(concurrency_stats.keys()):
        stats = concurrency_stats[limit]
        time_degradation = ((stats['avg_time'] - base_avg_time) / base_avg_time * 100) if base_avg_time > 0 else 0
        success_degradation = base_success_rate - stats['success_rate']
        
        max_time_degradation = max(max_time_degradation, time_degradation)
        max_success_degradation = max(max_success_degradation, success_degradation)
        
        print(f"Concurrency {limit:>3}: Time degradation: {time_degradation:>6.1f}%, Success degradation: {success_degradation:>5.1f}%")
    
    print(f"\nMax time degradation: {max_time_degradation:.1f}%")
    print(f"Max success rate drop: {max_success_degradation:.1f}%")
    
    # Query performance breakdown
    print(f"\nQuery Performance Analysis:")
    print("-" * 70)
    
    unique_queries = df['query'].unique()
    for query in unique_queries[:5]:  # Show top 5 queries
        query_df = df[df['query'] == query]
        query_success_rate = (query_df['success'].sum() / len(query_df) * 100) if len(query_df) > 0 else 0
        successful_query_times = query_df[query_df['success'] == True]['elapsed_time']
        avg_query_time = successful_query_times.mean() if len(successful_query_times) > 0 else 0
        
        print(f"'{query[:50]}{'...' if len(query) > 50 else ''}' -> {query_success_rate:.1f}% success, {avg_query_time:.3f}s avg")
    
    # Tool discovery analysis  
    if 'tools_discovered' in df.columns:
        print(f"\nTool Discovery Analysis:")
        print("-" * 70)
        
        tool_stats = df[df['success'] == True]['tools_discovered'].describe()
        print(f"Tools discovered per successful query:")
        print(f"• Mean: {tool_stats['mean']:.1f}")
        print(f"• Min: {tool_stats['min']:.0f}")
        print(f"• Max: {tool_stats['max']:.0f}")
        print(f"• Std: {tool_stats['std']:.1f}")
        
        # Tool discovery by concurrency
        for limit in sorted(df['concurrent_limit'].unique()):
            batch = df[(df['concurrent_limit'] == limit) & (df['success'] == True)]
            if len(batch) > 0:
                avg_tools = batch['tools_discovered'].mean()
                print(f"Concurrency {limit}: {avg_tools:.1f} tools per query")
    
    return {
        'total_queries': total_queries,
        'successful_queries': successful_queries,
        'overall_success_rate': overall_success_rate,
        'concurrency_stats': concurrency_stats,
        'max_time_degradation': max_time_degradation,
        'max_success_degradation': max_success_degradation
    }

def analyze_ui_results(ui_results):
    """Analyze UI test results."""
    if not ui_results:
        print("No UI results found.")
        return {}
    
    df = pd.DataFrame(ui_results)
    
    print("\nUI Performance Analysis")
    print("=" * 70)
    
    # Page element analysis
    print(f"\nPage Element Analysis:")
    print("-" * 50)
    element_columns = [col for col in df.columns if col.endswith('_found') or col.endswith('_time')]
    
    print(f"\nSearch Functionality Analysis:")
    print("-" * 50)
    
    # Search functionality metrics
    search_metrics = []
    search_columns = [
        'search_input_found', 'search_submit_found', 'search_results_found',
        'search_input_time', 'search_submit_time', 'search_results_time'
    ]
    
    available_search_columns = [col for col in search_columns if col in df.columns]
    
    if available_search_columns:
        for col in available_search_columns:
            if col.endswith('_found'):
                element_name = col.replace('_found', '')
                success_rate = (df[col].sum() / len(df) * 100) if len(df) > 0 else 0
                print(f"• {element_name.replace('_', ' ').title()}: {success_rate:.1f}% detection rate")
            elif col.endswith('_time'):
                element_name = col.replace('_time', '')
                valid_times = df[df[col].notna() & (df[col] > 0)][col]
                if len(valid_times) > 0:
                    avg_time = valid_times.mean()
                    print(f"• {element_name.replace('_', ' ').title()}: {avg_time:.3f}s avg time")
    else:
        print("• No search functionality data available")
    
    # Overall UI performance
    ui_performance = {}
    
    # Page load performance
    if 'page_load_time' in df.columns:
        page_load_times = df[df['page_load_time'].notna() & (df['page_load_time'] > 0)]['page_load_time']
        if len(page_load_times) > 0:
            ui_performance['avg_page_load_time'] = page_load_times.mean()
            ui_performance['max_page_load_time'] = page_load_times.max()
            ui_performance['min_page_load_time'] = page_load_times.min()
            print(f"\nPage Load Performance:")
            print(f"• Average: {ui_performance['avg_page_load_time']:.3f}s")
            print(f"• Min: {ui_performance['min_page_load_time']:.3f}s")
            print(f"• Max: {ui_performance['max_page_load_time']:.3f}s")
    
    # Login performance
    if 'login_time' in df.columns:
        login_times = df[df['login_time'].notna() & (df['login_time'] > 0)]['login_time']
        if len(login_times) > 0:
            ui_performance['avg_login_time'] = login_times.mean()
            print(f"\nLogin Performance:")
            print(f"• Average login time: {ui_performance['avg_login_time']:.3f}s")
    
    # Navigation performance
    navigation_columns = [col for col in df.columns if 'navigation' in col.lower() and col.endswith('_time')]
    if navigation_columns:
        print(f"\nNavigation Performance:")
        for col in navigation_columns:
            nav_times = df[df[col].notna() & (df[col] > 0)][col]
            if len(nav_times) > 0:
                avg_nav_time = nav_times.mean()
                action_name = col.replace('_time', '').replace('_', ' ').title()
                print(f"• {action_name}: {avg_nav_time:.3f}s")
    
    return ui_performance

def analyze_ui_search_results(ui_search_results):
    """Analyze UI search concurrency results."""
    if not ui_search_results:
        print("No UI search results found.")
        return {}
    
    df = pd.DataFrame(ui_search_results)
    
    print("\nUI Search Concurrency Analysis")
    print("=" * 70)
    
    # Overall statistics
    total_searches = len(df)
    successful_searches = df['success'].sum() if 'success' in df.columns else 0
    overall_success_rate = (successful_searches / total_searches * 100) if total_searches > 0 else 0
    
    print(f"• Total searches tested: {total_searches}")
    print(f"• Successful searches: {successful_searches}")
    print(f"• Overall success rate: {overall_success_rate:.1f}%")
    
    # Performance by concurrency level
    if 'concurrent_limit' in df.columns:
        print(f"\nPerformance by Concurrency Level:")
        print("-" * 50)
        print(f"{'Concurrency':>11} {'Success':>7} {'Rate':>6} {'Avg Time':>10}")
        print("-" * 50)
        
        for limit in sorted(df['concurrent_limit'].unique()):
            batch = df[df['concurrent_limit'] == limit]
            successful = batch['success'].sum() if 'success' in batch.columns else 0
            total = len(batch)
            success_rate = (successful / total * 100) if total > 0 else 0
            
            if 'elapsed_time' in batch.columns:
                successful_times = batch[batch['success'] == True]['elapsed_time'] if 'success' in batch.columns else batch['elapsed_time']
                avg_time = successful_times.mean() if len(successful_times) > 0 else 0
            else:
                avg_time = 0
                
            print(f"{limit:>11} {successful:>7} {success_rate:>5.1f}% {avg_time:>9.3f}s")
    
    return {
        'total_searches': total_searches,
        'successful_searches': successful_searches,
        'overall_success_rate': overall_success_rate
    }

def analyze_errors(all_results):
    """Analyze error patterns across all test types."""
    print(f"\nError Analysis:")
    print("=" * 70)
    
    error_count = 0
    error_types = {}
    
    # Analyze FAISS errors
    if 'faiss_results' in all_results and all_results['faiss_results']:
        faiss_df = pd.DataFrame(all_results['faiss_results'])
        faiss_errors = faiss_df[faiss_df['success'] == False]
        error_count += len(faiss_errors)
        
        if len(faiss_errors) > 0:
            for _, error_row in faiss_errors.iterrows():
                error_msg = error_row.get('error', 'Unknown FAISS error')
                error_types[f"FAISS: {error_msg}"] = error_types.get(f"FAISS: {error_msg}", 0) + 1
    
    # Analyze UI errors
    if 'ui_results' in all_results and all_results['ui_results']:
        ui_df = pd.DataFrame(all_results['ui_results'])
        # Look for error indicators in UI results
        for _, row in ui_df.iterrows():
            for col, val in row.items():
                if 'error' in col.lower() and val:
                    error_types[f"UI: {col}"] = error_types.get(f"UI: {col}", 0) + 1
                    error_count += 1
    
    print(f"Total errors detected: {error_count}")
    
    if error_types:
        print(f"Error breakdown:")
        for error_type, count in sorted(error_types.items(), key=lambda x: x[1], reverse=True):
            print(f"• {error_type}: {count} occurrences")
    else:
        print("No specific error patterns detected.")
    
    return error_count, error_types

def generate_summary_report(faiss_analysis, ui_analysis, ui_search_analysis, error_count, error_types):
    """Generate a comprehensive summary report."""
    print("\n" + "="*80)
    print("COMPREHENSIVE TEST SUMMARY")
    print("="*80)
    
    # Test overview
    print(f"\nTest Overview:")
    print("-" * 40)
    total_faiss_queries = faiss_analysis.get('total_queries', 0)
    total_ui_tests = len(ui_analysis) if ui_analysis else 0
    total_ui_searches = ui_search_analysis.get('total_searches', 0)
    
    print(f"• FAISS queries tested: {total_faiss_queries}")
    print(f"• UI tests performed: {total_ui_tests}")
    print(f"• UI search tests: {total_ui_searches}")
    
    # Performance grades
    print(f"\nPerformance Grades:")
    print("-" * 40)
    
    # FAISS grade
    faiss_success_rate = faiss_analysis.get('overall_success_rate', 0)
    if faiss_success_rate >= 90:
        faiss_grade = "A"
    elif faiss_success_rate >= 70:
        faiss_grade = "B"
    elif faiss_success_rate >= 50:
        faiss_grade = "C"
    else:
        faiss_grade = "F"
    
    print(f"• FAISS Performance: Grade {faiss_grade} ({faiss_success_rate:.1f}% success rate)")
    
    # UI grade (simplified based on available data)
    if ui_analysis:
        ui_grade = "B"  # Default since we have UI data
        print(f"• UI Performance: Grade {ui_grade} (Tests completed)")
    else:
        print(f"• UI Performance: Grade N/A (No UI tests)")
    
    # Overall system grade
    if faiss_grade in ['A', 'B'] and ui_analysis:
        overall_grade = "B"
    elif faiss_grade in ['A', 'B']:
        overall_grade = "C"
    else:
        overall_grade = "D"
    
    print(f"• Overall System: Grade {overall_grade}")
    
    # Recommendations
    print(f"\nRecommendations:")
    print("-" * 40)
    
    max_time_degradation = faiss_analysis.get('max_time_degradation', 0)
    max_success_degradation = faiss_analysis.get('max_success_degradation', 0)
    
    if max_time_degradation > 50:
        print("• Consider optimizing FAISS indexing or query processing")
    
    if max_success_degradation > 20:
        print("• Address concurrency-related failures")
    
    if error_count > 0:
        print(f"• Investigate and fix {error_count} detected errors")
    
    # Time degradation assessment
    if max_time_degradation > 100:
        print("• High performance degradation under load - consider load balancing")
    elif max_time_degradation > 50:
        print("• Moderate performance degradation - monitor under production load")
    elif max_time_degradation < 25:
        print("• Good performance stability under high concurrency")
    
    # UI-specific recommendations
    if ui_analysis:
        avg_page_load = ui_analysis.get('avg_page_load_time', 0)
        if avg_page_load > 3.0:
            print("• Page load time is slow - consider optimizing static assets")
    
    if ui_search_analysis and ui_search_analysis.get('overall_success_rate', 100) < 80:
        print("• Search functionality has issues - check UI selectors and JavaScript")
    
    if error_count > 0:
        print("• UI errors detected - review error logs for specific issues")
    
    # System readiness assessment
    print(f"\nSystem Readiness Assessment:")
    print("-" * 40)
    
    critical_issues = 0
    if faiss_success_rate < 70:
        critical_issues += 1
    if max_time_degradation > 100:
        critical_issues += 1
    if error_count > total_faiss_queries * 0.1:  # More than 10% error rate
        critical_issues += 1
    
    if critical_issues == 0:
        print("• System appears ready for production deployment")
        print("• Performance is within acceptable limits")
        print("• No critical issues detected")
    elif critical_issues <= 2:
        print("• System needs optimization before production")
        print("• Address identified issues and re-test")
    else:
        print("• System requires significant improvements")
        print("• Multiple critical issues need resolution")
        print("• Consider architecture review")

def main():
    parser = argparse.ArgumentParser(description='Analyze concurrency test results')
    parser.add_argument('results_file', help='Path to JSON results file')
    args = parser.parse_args()
    
    print("COMPREHENSIVE MCP STRESS TEST ANALYSIS")
    print("="*80)
    print(f"Results file: {args.results_file}")
    print(f"Analysis time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Load results
    try:
        with open(args.results_file, 'r') as f:
            results = json.load(f)
    except Exception as e:
        print(f"Error loading results file: {e}")
        return
    
    # Analyze each component
    faiss_analysis = analyze_faiss_results(results.get('faiss_results', []))
    ui_analysis = analyze_ui_results(results.get('ui_results', []))
    ui_search_analysis = analyze_ui_search_results(results.get('ui_search_results', []))
    error_count, error_types = analyze_errors(results)
    
    # Generate summary
    generate_summary_report(faiss_analysis, ui_analysis, ui_search_analysis, error_count, error_types)

if __name__ == "__main__":
    main() 