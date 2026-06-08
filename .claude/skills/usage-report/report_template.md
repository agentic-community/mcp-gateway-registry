# AI Registry -- Usage Report

*Report Date: {report_date}*
*Data Source: Telemetry Collector (DocumentDB)*
*Collection Period: {collection_period_start} to {collection_period_end} ({collection_period_days} days)*

---

## Executive Summary

The fleet stands at **{total_instances} unique registry instances** ({customer_instances} customer + {internal_instances} internal). {executive_summary_lead}

GitHub: **{github_stars}** stars ({github_stars_delta_str} since last report), **{github_forks}** forks ({github_forks_delta_str}), **{github_watchers}** watchers, **{github_open_issues}** open issues, **{github_contributors}** contributors.

**{sticky_3plus_days}** customer instances have been running for 3+ days (was {prev_sticky_3plus_days} in the previous report). The longest-running non-internal instance is `{longest_non_internal_id_short}` ({longest_non_internal_profile}) at **{longest_non_internal_days} days**.

**{one_day_wonder_pct_str}% of customer instances ({one_day_wonders} of {customer_instances})** are one-day wonders. {one_day_wonder_trend}

<!-- COMMENTARY:executive_summary -->

![Registry Installs Timeseries](registry-installs-timeseries-{date}.png)

### Comparison with Previous Report ({prev_report_date})

| Metric | Previous | Current | Change |
|--------|---------:|--------:|-------:|
| Total Events | {prev_total_events} | {total_events} | {total_events_delta_str} |
| Startup Events | {prev_startup_events} | {startup_events} | {startup_events_delta_str} |
| Heartbeat Events | {prev_heartbeat_events} | {heartbeat_events} | {heartbeat_events_delta_str} |
| Unique Instances | {prev_total_instances} | {total_instances} | {total_instances_delta_str} |
| Sticky 3+ days | {prev_sticky_3plus_days} | {sticky_3plus_days} | {sticky_delta_str} |
| Confirmed Alive | {prev_confirmed_alive} | {confirmed_alive} | {confirmed_alive_delta_str} |
| Stronger Alive | {prev_stronger_alive} | {stronger_alive} | {stronger_alive_delta_str} |
| Stars | {prev_github_stars} | {github_stars} | {github_stars_delta_str} |

### Unique Registry Installs by Cloud Provider

Lifetime cumulative counts: every distinct `registry_id` ever observed under each cloud provider since telemetry began. Once counted, never decrements -- a dormant or deleted instance still shows up here. The Liveness section reports who is currently active.

{cloud_installs_table}

<!-- COMMENTARY:cloud_installs -->

## Deployment Distribution (by Unique Instances)

![Instance Distribution -- All Customers Ever](instance-distribution-{date}.png)
![Instance Distribution -- Active on {prev_date}](instance-distribution-active-{prev_date}.png)

The cumulative view (top) is the everyone-who-ever-installed picture. The active-yesterday view (bottom) is the who-is-still-running-it picture. The two diverge consistently: active-yesterday skews toward Kubernetes and ECS over single-shot Docker, AWS as a share of total, and enterprise IdP (Entra, Okta) over Keycloak. Operators who connect their IdP and run on managed compute are the ones who stick.

<!-- COMMENTARY:deployment_distribution -->

## Key Metrics

| Metric | Value |
|--------|------:|
| Total Events | {total_events} |
| Startup Events | {startup_events} |
| Heartbeat Events | {heartbeat_events} |
| Unique Registry Instances | {total_instances} |
| Known Internal Instances | {internal_instances} |
| Customer Instances | {customer_instances} |
| Confirmed Alive (>=5 HB in 7d) | {confirmed_alive} |
| Stronger Alive (>=10 HB in 14d) | {stronger_alive} |
| Sticky (3+ days) | {sticky_3plus_days} |
| Longest Non-Internal Instance | {longest_non_internal_days} days |

## Internal Instances (Development/Testing)

The {internal_instances} known internal instances are listed in `.claude/skills/usage-report/known-internal-instances.md`. They show disproportionate activity (heavy search usage, frequent restarts, large catalogs) because they are always-on dev environments. **Activity from these instances reflects internal testing and should not be interpreted as customer usage patterns.** Additional unknown-internal instances may exist beyond the listed four.

## Registry Instance Lifetime

Across {customer_instances} customer instances, the average lifetime is {avg_lifetime_days} days, max {longest_non_internal_days} days. {one_day_wonders} customers were seen on a single day ({one_day_wonder_pct_str}%); the remaining {multi_day_customers} ({multi_day_pct_str}%) have multi-day footprints.

![Instance Lifetime](instance-lifetime-{date}.png)

### Customer Lifetime Retention Over Time

![Lifetime Bucket Retention](lifetime-buckets-{date}.png)

Today's snapshot vs the earliest available snapshot:

| Threshold | {earliest_snapshot_date} | {date} | Change |
|-----------|------------------------:|------:|-------:|
| >= 3 days | {earliest_pct_3d}% | {pct_3d}% | {pct_3d_change_str} |
| >= 7 days | {earliest_pct_7d}% | {pct_7d}% | {pct_7d_change_str} |
| >= 14 days | {earliest_pct_14d}% | {pct_14d}% | {pct_14d_change_str} |
| >= 30 days | {earliest_pct_30d}% | {pct_30d}% | {pct_30d_change_str} |
| 1-day wonders | {earliest_pct_odw}% | {one_day_wonder_pct_str}% | {pct_odw_change_str} |

<!-- COMMENTARY:lifetime_retention -->

## Liveness (Currently Active Instances)

| Tier | Count | % of Customer Fleet | vs Previous Report |
|------|------:|--------------------:|-------------------:|
| Confirmed Alive (>=5 HB in 7d) | {confirmed_alive} | {confirmed_alive_pct}% | {confirmed_alive_delta_str} |
| Stronger Alive (>=10 HB in 14d) | {stronger_alive} | {stronger_alive_pct}% | {stronger_alive_delta_str} |
| Likely Alive (any event in 7d) | {likely_alive} | {likely_alive_pct}% | {likely_alive_delta_str} |
| Silent-but-recent (<5 HB in 7d) | {silent_but_recent} | {silent_but_recent_pct}% | {silent_but_recent_delta_str} |
| Dormant (no event in 14d) | {dormant} | {dormant_pct}% | {dormant_delta_str} |

The Confirmed-Alive instances skew toward AWS, with ECS, Kubernetes, and Docker each carrying roughly a third of the active production fleet.

<!-- COMMENTARY:liveness -->

### Engagement Over Time

![Active Instances](active-instances-{date}.png)

| Date | DAI | MA7 | 7d Streak | Cumulative | DAI/Total | DAI/Likely-Alive |
|------|----:|----:|----------:|-----------:|----------:|-----------------:|
{engagement_recent_days_table}

<!-- COMMENTARY:engagement -->

## Compute Platform Growth

![Compute Installs Timeseries](compute-installs-timeseries-{date}.png)

### Per-Platform Growth (Unique Installs)

{compute_platform_snapshots_table}

<!-- COMMENTARY:compute_platform -->

## Version Adoption

{version_adoption_table}

<!-- COMMENTARY:version_adoption -->

## Version Upgrade Trajectories

{upgrade_trajectories_count} customer instances have reported more than one distinct version. Top chains:

{upgrade_trajectories_table}

<!-- COMMENTARY:upgrade_trajectories -->

## Feature Adoption

{feature_adoption_table}

### Embeddings Backend

{embeddings_backend_table}

<!-- COMMENTARY:feature_adoption -->

## Search Usage

| Metric | Value |
|--------|------:|
| Instances with search activity | {search_active_instances} |
| Total search queries (lifetime, deduplicated) | {search_total_queries} |
| Average per instance | {search_avg_per_instance} |
| Max from single instance | {search_max_single_instance} |

## Sticky Instance Breakdown (3+ Days)

{sticky_3plus_days} customer instances are running for 3+ days. By cloud + compute:

{sticky_cloud_compute_table}

<!-- COMMENTARY:sticky_breakdown -->

## Most Active Instances (by Feature Usage)

{most_active_instances_table}

<!-- COMMENTARY:most_active -->

## Largest Catalogs (by Registered Servers + Agents + Skills)

{largest_catalogs_table}

## Most Engaged Operators (by Upgrade-Chain Length)

{most_engaged_operators_table}

## Install Forecast

![Install Forecast](install-forecast-{date}.png)

| Model | Daily Rate | ETA to 1,000 |
|-------|----------:|:-------------|
| Linear OLS (14-day) | {forecast_linear_rate} / day | {forecast_linear_eta} |
| Recent pace (7-day) | {forecast_recent_rate} / day | {forecast_recent_eta} |

{forecast_narrative}

<!-- COMMENTARY:install_forecast -->

## Customer Infra Spend (AWS)

![LTV Spend](ltv-spend-{date}.png)

Hypothetical AWS infrastructure cost (list price), customer-only:

| Window | Proven | All-Days |
|--------|------:|---------:|
| Yesterday ({yesterday_label}) | {ltv_yesterday_proven} | {ltv_yesterday_all_days} |
| Last 7 days | {ltv_last_7_days_proven} | {ltv_last_7_days_all_days} |
| 7-day daily average | {ltv_7day_daily_avg_proven} | {ltv_7day_daily_avg_all_days} |
| Cumulative LTV | {ltv_cumulative_proven} | {ltv_cumulative_all_days} |

### Annualized Run Rate (ARR) Projection

Projecting the last-7-day daily average forward 365 days:

| Model | 7-day daily avg | x 365 = ARR |
|-------|----------------:|------------:|
| Proven | {ltv_7day_daily_avg_proven} | **{arr_proven_str}** |
| All-days | {ltv_7day_daily_avg_all_days} | **{arr_all_days_str}** |

The ARR is hypothetical (we do not bill these customers). It represents the AWS-side infra cost the deployed customer base is generating annually at current run rate, complementing install-count growth as a measure of the fleet's economic footprint.

<!-- COMMENTARY:ltv_arr -->

## Adoption Funnel

![Adoption Funnel](adoption-funnel-{date}.png)

| Stage | Count | % of Top |
|-------|------:|---------:|
| Total customer installs | {customer_instances} | 100.0% |
| Multi-day (>=1 day lifetime) | {multi_day_customers} | {multi_day_pct_str}% |
| Sticky (>=3 days) | {sticky_3plus_days} | {pct_3d}% |
| Weekly (>=7 days) | {weekly_count} | {pct_7d}% |
| Biweekly (>=14 days) | {biweekly_count} | {pct_14d}% |
| Monthly (>=30 days) | {monthly_count} | {pct_30d}% |
| Confirmed Alive (>=5 HB in 7d) | {confirmed_alive} | {confirmed_alive_pct}% |

<!-- COMMENTARY:adoption_funnel -->

## Cloud Detection Outcomes by Version

![Cloud Detection by Version](detection-by-version-{date}.png)

{detection_by_version_narrative}

<!-- COMMENTARY:cloud_detection -->

## GitHub Repository

| Metric | Previous | Current | Change |
|--------|---------:|--------:|-------:|
| Stars | {prev_github_stars} | {github_stars} | {github_stars_delta_str} |
| Forks | {prev_github_forks} | {github_forks} | {github_forks_delta_str} |
| Contributors | {prev_github_contributors} | {github_contributors} | {github_contributors_delta_str} |
| Watchers | {prev_github_watchers} | {github_watchers} | {github_watchers_delta_str} |
| Open issues | {prev_github_open_issues} | {github_open_issues} | {github_open_issues_delta_str} |

<!-- COMMENTARY:github -->

## Architecture Patterns Observed

{architecture_patterns}

<!-- COMMENTARY:architecture -->

## Recommendations

{recommendations_section}

<!-- COMMENTARY:recommendations -->

## Appendix: Raw Distribution Tables

(Event-count-based distributions are in `tables-{date}.md`. The instance-based views above are the canonical numbers; event-count tables are useful for spotting heavy-restart instances but do not represent customer counts.)
