#!/bin/bash
# Telemetry Collector Status Check Script
# Run this every 6 hours during the 24-hour monitoring period

set -e

echo "=== Telemetry Collector Status Check ==="
echo "Time: $(date -u)"
echo ""

echo "1. Lambda Errors (last 6 hours):"
ERROR_COUNT=$(aws logs filter-log-events \
  --log-group-name /aws/lambda/telemetry-collector \
  --start-time $(($(date +%s) * 1000 - 21600000)) \
  --filter-pattern "ERROR" \
  --query 'length(events)' \
  --output text 2>/dev/null || echo "0")
echo "   Errors: $ERROR_COUNT"
if [ "$ERROR_COUNT" != "0" ] && [ "$ERROR_COUNT" != "None" ]; then
    echo "   ⚠️  WARNING: Errors detected!"
fi

echo ""
echo "2. Lambda Invocations (last 24 hours):"
INVOCATIONS=$(aws cloudwatch get-metric-statistics \
  --namespace AWS/Lambda \
  --metric-name Invocations \
  --dimensions Name=FunctionName,Value=telemetry-collector \
  --start-time $(date -u -v-24H +%Y-%m-%dT%H:%M:%S 2>/dev/null || date -u -d '24 hours ago' +%Y-%m-%dT%H:%M:%S) \
  --end-time $(date -u +%Y-%m-%dT%H:%M:%S) \
  --period 86400 \
  --statistics Sum \
  --query 'Datapoints[0].Sum' \
  --output text 2>/dev/null || echo "N/A")
echo "   Total: $INVOCATIONS"

echo ""
echo "3. Average Duration (last 24 hours):"
AVG_DURATION=$(aws cloudwatch get-metric-statistics \
  --namespace AWS/Lambda \
  --metric-name Duration \
  --dimensions Name=FunctionName,Value=telemetry-collector \
  --start-time $(date -u -v-24H +%Y-%m-%dT%H:%M:%S 2>/dev/null || date -u -d '24 hours ago' +%Y-%m-%dT%H:%M:%S) \
  --end-time $(date -u +%Y-%m-%dT%H:%M:%S) \
  --period 86400 \
  --statistics Average \
  --query 'Datapoints[0].Average' \
  --output text 2>/dev/null || echo "N/A")
if [ "$AVG_DURATION" != "N/A" ] && [ "$AVG_DURATION" != "None" ]; then
    echo "   Duration: ${AVG_DURATION}ms"
    # Check if duration is too high (>1000ms average)
    DURATION_INT=$(echo $AVG_DURATION | cut -d. -f1)
    if [ "$DURATION_INT" -gt 1000 ]; then
        echo "   ⚠️  WARNING: Duration higher than expected!"
    fi
else
    echo "   Duration: No data yet"
fi

echo ""
echo "4. Recent Events (last hour):"
RECENT_EVENTS=$(aws logs filter-log-events \
  --log-group-name /aws/lambda/telemetry-collector \
  --start-time $(($(date +%s) * 1000 - 3600000)) \
  --query 'events[*].message' \
  --output text 2>/dev/null | grep -E "(Stored|Validated)" | tail -5)
if [ -n "$RECENT_EVENTS" ]; then
    echo "$RECENT_EVENTS"
else
    echo "   No events in last hour"
fi

echo ""
echo "5. Rate Limit Table Status:"
RATE_LIMIT_COUNT=$(aws dynamodb scan \
  --table-name telemetry-collector-rate-limit \
  --select COUNT \
  --query 'Count' \
  --output text 2>/dev/null || echo "0")
echo "   Tracked IPs: $RATE_LIMIT_COUNT"

echo ""
echo "=== Status Check Complete ==="
echo ""
echo "Next check: $(date -u -v+6H +%Y-%m-%d\ %H:%M:%S\ UTC 2>/dev/null || date -u -d '6 hours' +%Y-%m-%d\ %H:%M:%S\ UTC)"
