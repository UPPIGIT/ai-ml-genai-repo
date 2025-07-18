# Complete Grafana Alert Setup Guide

## Example 1: High Error Rate Alert

### Step-by-Step Setup

#### 1. Create Alert Rule
- Go to **Alerting** → **Alert rules** → **New rule**
- Name: `High Error Rate - Production Services`

#### 2. Query Configuration
```promql
# Query A: Calculate error rate percentage
sum(rate(istio_requests_total{response_code=~"5.*"}[5m])) by (destination_service_name) / sum(rate(istio_requests_total[5m])) by (destination_service_name) * 100
```

**What this does:**
- Calculates percentage of 5xx errors in last 5 minutes
- Groups by service name
- Multiplies by 100 to get percentage

#### 3. Alert Condition
- **Condition**: `IS ABOVE 5`
- **Evaluation**: Every `30s`
- **For**: `2m`

**Translation**: "Alert if error rate is above 5% for more than 2 minutes"

#### 4. Alert Details
- **Summary**: `High error rate detected for {{ $labels.destination_service_name }}`
- **Description**: `Error rate is {{ $value }}% for service {{ $labels.destination_service_name }}`

---

## Example 2: High Latency Alert

### Query Configuration
```promql
# P95 latency in milliseconds
histogram_quantile(0.95, sum(rate(istio_request_duration_milliseconds_bucket[5m])) by (destination_service_name, le))
```

### Alert Condition
- **Condition**: `IS ABOVE 1000`
- **Evaluation**: Every `1m`
- **For**: `3m`

**Translation**: "Alert if 95th percentile latency is above 1000ms for more than 3 minutes"

---

## Example 3: Service Unavailable Alert

### Query Configuration
```promql
# Check if service is up
up{job="kubernetes-pods"}
```

### Alert Condition
- **Condition**: `IS BELOW 1`
- **Evaluation**: Every `10s`
- **For**: `30s`

**Translation**: "Alert immediately if service goes down for more than 30 seconds"

---

## Example 4: High CPU Usage Alert

### Query Configuration
```promql
# CPU usage percentage per pod
sum(rate(container_cpu_usage_seconds_total{container!="POD",container!=""}[5m])) by (pod, namespace) * 100
```

### Alert Condition
- **Condition**: `IS ABOVE 80`
- **Evaluation**: Every `1m`
- **For**: `5m`

**Translation**: "Alert if CPU usage is above 80% for more than 5 minutes"

---

## Setting Up Notifications

### 1. Create Notification Channel

#### Email Notification
- Go to **Alerting** → **Notification channels**
- Click **New channel**
- **Type**: Email
- **Name**: `DevOps Team Email`
- **Email addresses**: `devops@company.com`
- **Test** the connection

#### Slack Notification
- **Type**: Slack
- **Webhook URL**: `https://hooks.slack.com/services/YOUR/SLACK/WEBHOOK`
- **Channel**: `#alerts`
- **Username**: `Grafana`

### 2. Assign Notifications to Alerts
- In your alert rule, scroll to **Notifications**
- Select your notification channel
- Add custom message if needed

---

## Alert Message Templates

### Good Alert Messages Include:
1. **What happened**: Service name, metric value
2. **When**: Timestamp
3. **Where**: Environment, namespace
4. **Severity**: Critical, Warning, Info

### Example Template:
```
🚨 **{{ .RuleName }}**

**Service**: {{ $labels.destination_service_name }}
**Current Value**: {{ $value }}%
**Threshold**: 5%
**Duration**: 2 minutes
**Environment**: {{ $labels.namespace }}

**Runbook**: https://wiki.company.com/alerts/high-error-rate
```

---

## Best Practices

### 1. Alert Thresholds
- **Critical**: Immediate action required
  - Service down: `up == 0`
  - Error rate: `> 10%`
  - Latency: `> 5000ms`

- **Warning**: Investigate soon
  - Error rate: `> 2%`
  - Latency: `> 1000ms`
  - CPU usage: `> 80%`

- **Info**: Good to know
  - Error rate: `> 0.5%`
  - Latency: `> 500ms`
  - CPU usage: `> 60%`

### 2. Timing Guidelines
- **Evaluation frequency**: 
  - Critical alerts: Every `10-30s`
  - Warning alerts: Every `1-5m`
  - Info alerts: Every `5-15m`

- **Alert duration**:
  - Transient issues: `1-2m`
  - Persistent issues: `5-10m`
  - Capacity issues: `10-30m`

### 3. Avoid Alert Fatigue
- Set appropriate thresholds
- Use different severity levels
- Group related alerts
- Include runbook links
- Test alerts regularly

---

## Common Queries for GKE Services

### Service Health
```promql
# Services responding
sum(rate(istio_requests_total[5m])) by (destination_service_name) > 0

# Error rate
sum(rate(istio_requests_total{response_code=~"5.*"}[5m])) by (destination_service_name) / sum(rate(istio_requests_total[5m])) by (destination_service_name) * 100

# Success rate
sum(rate(istio_requests_total{response_code=~"2.*"}[5m])) by (destination_service_name) / sum(rate(istio_requests_total[5m])) by (destination_service_name) * 100
```

### Performance Metrics
```promql
# Average response time
sum(rate(istio_request_duration_milliseconds_sum[5m])) by (destination_service_name) / sum(rate(istio_request_duration_milliseconds_count[5m])) by (destination_service_name)

# P95 latency
histogram_quantile(0.95, sum(rate(istio_request_duration_milliseconds_bucket[5m])) by (destination_service_name, le))

# Request volume
sum(rate(istio_requests_total[5m])) by (destination_service_name)
```

### Resource Usage
```promql
# Memory usage
sum(container_memory_working_set_bytes{container!="POD",container!=""}) by (pod, namespace)

# CPU usage
sum(rate(container_cpu_usage_seconds_total{container!="POD",container!=""}[5m])) by (pod, namespace)

# Network traffic
sum(rate(container_network_receive_bytes_total[5m])) by (pod, namespace)
```

---

## Testing Your Alerts

### 1. Manual Testing
- Change thresholds temporarily to trigger alerts
- Verify notifications are received
- Check message formatting

### 2. Simulate Problems
```bash
# Generate high load to trigger CPU alerts
kubectl run load-generator --image=busybox --restart=Never -- /bin/sh -c "while true; do echo 'generating load'; done"

# Generate errors to trigger error rate alerts
# Deploy a service that returns 500 errors
```

### 3. Alert Validation Checklist
- [ ] Alert triggers at correct threshold
- [ ] Notification is sent to correct channel
- [ ] Message contains relevant information
- [ ] Alert resolves when issue is fixed
- [ ] No false positives

---

## Next Steps

1. **Start simple**: Create 1-2 basic alerts (service down, high error rate)
2. **Test thoroughly**: Verify alerts work as expected
3. **Iterate**: Add more alerts based on your services
4. **Monitor**: Track alert effectiveness and adjust thresholds
5. **Document**: Create runbooks for common alerts