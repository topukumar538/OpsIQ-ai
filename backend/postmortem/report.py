# Location: backend/postmortem/report.py
from datetime import datetime, timezone, timezone


def build_report(state: dict, log_filename: str) -> str:
    error_counts = state.get("error_counts", {})
    divider = "=" * 60
    section = "-" * 60
    lines   = []

    lines.append(f"\n{divider}")
    lines.append("  POSTMORTEM REPORT")
    lines.append(f"  File     : {log_filename}")
    lines.append(f"  Generated: {datetime.now(tz=timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
    lines.append(divider)

    lines.append("\n[ ERRORS DETECTED ]")
    lines.append(section)
    if error_counts:
        lines.append(f"Total occurrences : {sum(error_counts.values())}")
        lines.append(f"Unique error types : {len(error_counts)}\n")
        for name, count in sorted(error_counts.items(), key=lambda x: -x[1]):
            lines.append(f"  {name:<40} x{count}")
    else:
        lines.append("  No major errors detected.")

    for title, key in [
        ("LOG ANALYSIS",    "log_analysis"),
        ("TIMELINE",        "timeline_analysis"),
        ("ROOT CAUSE",      "root_cause"),
        ("REMEDIATION PLAN","remediation"),
    ]:
        lines.append(f"\n[ {title} ]")
        lines.append(section)
        lines.append(state.get(key, ""))

    lines.append(f"\n{divider}\n")
    return "\n".join(lines)