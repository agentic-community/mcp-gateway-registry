"""Shared helper that maps a telemetry row to the cloud label shown in reports.

A registry reports cloud=unknown both when detection found nothing and when
an operator chose Dismiss on the admin banner (cloud_detection_method=
operator_declined). Reports show the second case as "declined", so a
deliberate choice is not counted as a detection miss (issue #1911).
"""

DECLINED_LABEL: str = "declined"


def cloud_label(
    row: dict[str, str],
) -> str:
    """Return the report label for a row's cloud, splitting declined from unknown."""
    cloud = row.get("cloud") or "unknown"
    method = row.get("cloud_detection_method") or ""
    if cloud == "unknown" and method == "operator_declined":
        return DECLINED_LABEL
    return cloud
