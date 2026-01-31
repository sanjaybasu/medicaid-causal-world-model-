"""
Utilities to stitch encounters, claims, and ADT into patient timelines.

This module focuses on structure; actual data pulls should occur upstream.
"""

from datetime import datetime
from typing import Dict, List, Optional

import pandas as pd

from .data_schema import PatientEvent, Timeline


def build_timeline_from_encounters(encounters: pd.DataFrame) -> Timeline:
    """
    Convert encounter rows into a Timeline of PatientEvent.

    Expected columns: member_id (WaymarkId), event_time (parsed), contactType/title, text,
    and any structured tags (intervention_tags, sdoh_need_tags, etc.).
    """
    events: Timeline = []
    for _, row in encounters.iterrows():
        try:
            ts = pd.to_datetime(row.get("dateOfEncounter") or row.get("event_time"))
        except Exception:
            continue
        event = PatientEvent(
            member_id=str(row.get("WaymarkId", "")),
            event_time=ts.to_pydatetime() if isinstance(ts, pd.Timestamp) else datetime.fromisoformat(str(ts)),
            event_type="encounter",
            action=None,  # can be populated from structured tags if desired
            outcome=None,
            features={
                "contactType": str(row.get("contactType", "")).lower(),
                "title": str(row.get("title", "")),
            },
            text_features={},
            latent_z=row.get("latent_z"),
            censor_flag=False,
        )
        events.append(event)
    events.sort(key=lambda e: e.event_time)
    return events


def merge_timelines(*timelines: Timeline) -> Timeline:
    """Merge multiple sorted timelines and return a globally time-sorted list."""
    merged: Timeline = []
    for tl in timelines:
        merged.extend(tl)
    merged.sort(key=lambda e: e.event_time)
    return merged
