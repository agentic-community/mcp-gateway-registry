/**
 * RegistryAlarms - CloudWatch alarms for ECS/ALB/DocumentDB. No-op when
 * config.monitoring.enabled = false. Mirrors terraform/aws-ecs/monitoring.tf.
 */

import * as cdk from 'aws-cdk-lib';
import * as cloudwatch from 'aws-cdk-lib/aws-cloudwatch';
import * as cwActions from 'aws-cdk-lib/aws-cloudwatch-actions';
import * as elbv2 from 'aws-cdk-lib/aws-elasticloadbalancingv2';
import * as sns from 'aws-cdk-lib/aws-sns';
import * as snsSubs from 'aws-cdk-lib/aws-sns-subscriptions';
import { Construct } from 'constructs';
import { RegistryConfig } from '../registry-config';

export interface RegistryAlarmsProps {
  readonly config: RegistryConfig;
  readonly clusterName: string;
  readonly registryServiceName: string;
  readonly authServiceName: string;
  readonly alb: elbv2.ApplicationLoadBalancer;
  readonly registryTargetGroup: elbv2.ApplicationTargetGroup;
  readonly documentDbClusterId?: string;
}

export class RegistryAlarms extends Construct {
  constructor(scope: Construct, id: string, props: RegistryAlarmsProps) {
    super(scope, id);
    const { config } = props;
    if (!config.monitoring.enabled) return;
    // No notification target → no point in alarms.
    if (!config.monitoring.alarmSnsTopicArn && !config.monitoring.alarmEmail) return;

    let topic: sns.ITopic;
    if (config.monitoring.alarmSnsTopicArn) {
      topic = sns.Topic.fromTopicArn(this, 'Topic', config.monitoring.alarmSnsTopicArn);
    } else {
      const newTopic = new sns.Topic(this, 'Topic', { topicName: `${config.name}-alarms` });
      newTopic.addSubscription(new snsSubs.EmailSubscription(config.monitoring.alarmEmail));
      topic = newTopic;
    }
    const action = new cwActions.SnsAction(topic);

    const alarm = (id: string, props: Omit<cloudwatch.AlarmProps, 'alarmName'> & { alarmName?: string }) => {
      const a = new cloudwatch.Alarm(this, id, {
        ...props,
        alarmName: props.alarmName ?? `${config.name}-${id}`,
      });
      a.addAlarmAction(action);
      return a;
    };

    // ECS service alarms
    const ecsDims = (svc: string) => ({ ClusterName: props.clusterName, ServiceName: svc });
    for (const [svc, name] of [
      [props.authServiceName, 'auth'],
      [props.registryServiceName, 'registry'],
    ] as const) {
      alarm(`${name}-cpu-high`, {
        metric: new cloudwatch.Metric({
          namespace: 'AWS/ECS', metricName: 'CPUUtilization',
          dimensionsMap: ecsDims(svc), statistic: 'Average',
          period: cdk.Duration.minutes(5),
        }),
        threshold: 80, evaluationPeriods: 2,
        comparisonOperator: cloudwatch.ComparisonOperator.GREATER_THAN_THRESHOLD,
      });
      alarm(`${name}-memory-high`, {
        metric: new cloudwatch.Metric({
          namespace: 'AWS/ECS', metricName: 'MemoryUtilization',
          dimensionsMap: ecsDims(svc), statistic: 'Average',
          period: cdk.Duration.minutes(5),
        }),
        threshold: 85, evaluationPeriods: 2,
        comparisonOperator: cloudwatch.ComparisonOperator.GREATER_THAN_THRESHOLD,
      });
    }

    // ALB alarms
    const albDim = { LoadBalancer: props.alb.loadBalancerFullName };
    alarm('alb-unhealthy-targets', {
      metric: new cloudwatch.Metric({
        namespace: 'AWS/ApplicationELB', metricName: 'UnHealthyHostCount',
        dimensionsMap: { ...albDim, TargetGroup: props.registryTargetGroup.targetGroupFullName },
        statistic: 'Maximum', period: cdk.Duration.minutes(1),
      }),
      threshold: 1, evaluationPeriods: 2,
      comparisonOperator: cloudwatch.ComparisonOperator.GREATER_THAN_OR_EQUAL_TO_THRESHOLD,
    });
    alarm('alb-5xx-errors', {
      metric: new cloudwatch.Metric({
        namespace: 'AWS/ApplicationELB', metricName: 'HTTPCode_ELB_5XX_Count',
        dimensionsMap: albDim, statistic: 'Sum',
        period: cdk.Duration.minutes(5),
      }),
      threshold: 10, evaluationPeriods: 2,
      comparisonOperator: cloudwatch.ComparisonOperator.GREATER_THAN_THRESHOLD,
    });
    alarm('alb-response-time', {
      metric: new cloudwatch.Metric({
        namespace: 'AWS/ApplicationELB', metricName: 'TargetResponseTime',
        dimensionsMap: albDim, statistic: 'Average',
        period: cdk.Duration.minutes(5),
      }),
      threshold: 2, evaluationPeriods: 3,
      comparisonOperator: cloudwatch.ComparisonOperator.GREATER_THAN_THRESHOLD,
    });

    // ponytail: skipped WAF/S3/KMS alarms — they need names from the Cdn
    // stack and the KMS-throttling metric has no per-key dimension. Wire up
    // when the Cdn → Service stack name plumbing is added.

    // DocumentDB audit log failures
    if (props.documentDbClusterId) {
      alarm('documentdb-audit-failures', {
        metric: new cloudwatch.Metric({
          namespace: 'AWS/DocDB', metricName: 'AuditLogFailures',
          dimensionsMap: { DBClusterIdentifier: props.documentDbClusterId },
          statistic: 'Sum', period: cdk.Duration.minutes(5),
        }),
        threshold: 1, evaluationPeriods: 1,
        comparisonOperator: cloudwatch.ComparisonOperator.GREATER_THAN_OR_EQUAL_TO_THRESHOLD,
      });
    }
  }
}
