"""
Taxonomy definitions for multidisciplinary encounter notes and interventions.

Intended to support LLM tagging, heuristic backstops, and downstream policy action space.
"""

from typing import Dict, List


ACTOR_ROLES = [
    "CHW",
    "Pharmacist",
    "Therapist",
    "CareCoordinator",
    "Nurse",
    "Provider",
    "Admin",
]

CHANNELS = [
    "sms",
    "phone",
    "in_person_home",
    "in_person_clinic",
    "video",
    "group_visit",
    "portal",
    "unspecified",
]

REACH_STATUS = [
    "reached_live",
    "voicemail_full",
    "voicemail_left",
    "disconnected",
    "wrong_number",
    "refused",
    "scheduled",
    "no_show",
    "rescheduled",
]

OUTCOME_STATUS = [
    "completed",
    "pending",
    "scheduled",
    "failed",
    "deferred_by_patient",
    "waitlist",
    "provider_unresponsive",
]

SDOH_NEEDS = [
    "housing",
    "food",
    "utilities",
    "transport",
    "phone_access",
    "childcare",
    "employment",
    "legal",
    "safety",
    "social_support",
    "equipment_dme",
    "translation",
]

CLINICAL_SUPPORT = [
    "med_reconciliation",
    "med_refill_coordination",
    "prior_auth",
    "adherence_coaching",
    "side_effect_review",
    "polypharmacy_risk",
    "blister_pack_setup",
    "symptom_triage",
    "vitals_review",
    "wound_photo_review",
    "lab_followup",
    "screening_PHQ",
    "screening_GAD",
    "vaccination_coordination",
    "education",
    "fall_risk_check",
]

THERAPY_SUPPORT = [
    "therapy_session",
    "CBT_lite",
    "MI_session",
    "crisis_deescalation",
    "relapse_prevention",
    "coping_skills",
    "safety_planning",
]

COORDINATION = [
    "benefits_navigation",
    "forms_completion",
    "insurance_issue",
    "appointment_scheduling_pcp",
    "appointment_scheduling_specialist",
    "referral_management",
    "discharge_followup",
    "records_request",
    "paperwork_delivery",
    "care_plan_update",
    "authorization",
]

SOCIAL_SUPPORT = [
    "transport_arranged",
    "food_support",
    "housing_support",
    "utilities_support",
    "phone_support",
    "childcare_support",
    "employment_support",
    "equipment_dme",
    "home_safety_check",
    "language_support",
]

BEHAVIORAL_SUPPORT = [
    "motivational_interviewing",
    "trust_building",
    "activation_priming",
    "goal_setting",
    "relapse_prevention",
]

# Simple keyword seeds to bootstrap tagging when LLM output is unavailable.
KEYWORD_TAGS: Dict[str, List[str]] = {
    "transport_arranged": ["lyft", "uber", "ride", "transport", "bus pass", "shuttle"],
    "food_support": ["food", "pantry", "grocery", "meal", "voucher", "snap"],
    "housing_support": ["eviction", "shelter", "housing", "landlord"],
    "utilities_support": ["utility", "electric", "power", "shutoff", "gas", "water"],
    "phone_support": ["phone", "minutes", "data plan", "device"],
    "med_reconciliation": ["med rec", "medication reconciliation"],
    "prior_auth": ["prior auth", "authorization"],
    "med_refill_coordination": ["refill", "rx", "pharmacy", "pickup"],
    "therapy_session": ["therapy", "cbt", "session"],
    "screening_PHQ": ["phq-9", "phq9"],
    "screening_GAD": ["gad-7", "gad7"],
    "motivational_interviewing": ["motivational interviewing", "mi session"],
    "discharge_followup": ["discharge", "d/c", "dc follow up"],
    "appointment_scheduling_pcp": ["pcp appointment", "pcp visit", "primary care"],
    "appointment_scheduling_specialist": ["specialist", "referral"],
}


def default_taxonomy() -> Dict[str, List[str]]:
    """Return the full taxonomy dictionary."""
    return {
        "actor_roles": ACTOR_ROLES,
        "channels": CHANNELS,
        "reach_status": REACH_STATUS,
        "outcome_status": OUTCOME_STATUS,
        "sdoh_needs": SDOH_NEEDS,
        "clinical_support": CLINICAL_SUPPORT,
        "therapy_support": THERAPY_SUPPORT,
        "coordination": COORDINATION,
        "social_support": SOCIAL_SUPPORT,
        "behavioral_support": BEHAVIORAL_SUPPORT,
    }


def bootstrap_tags(note_text: str) -> Dict[str, List[str]]:
    """
    Lightweight keyword matcher to pre-label high-confidence tags from note text.
    Useful as a backstop or to build few-shot examples.
    """
    tags: Dict[str, List[str]] = {
        "intervention": [],
        "sdoh_need": [],
        "clinical_support": [],
        "coordination": [],
        "behavioral_support": [],
    }
    text = note_text.lower()
    for tag, keywords in KEYWORD_TAGS.items():
        if any(k in text for k in keywords):
            if tag in {"transport_arranged", "food_support", "housing_support", "utilities_support", "phone_support"}:
                tags["sdoh_need"].append(tag.replace("_support", ""))
                tags["intervention"].append(tag)
            elif tag in {"med_reconciliation", "prior_auth", "med_refill_coordination", "screening_PHQ", "screening_GAD"}:
                tags["clinical_support"].append(tag)
                tags["intervention"].append(tag)
            elif tag in {"therapy_session"}:
                tags["intervention"].append(tag)
            elif tag in {"motivational_interviewing"}:
                tags["behavioral_support"].append(tag)
                tags["intervention"].append(tag)
            elif tag in {"discharge_followup", "appointment_scheduling_pcp", "appointment_scheduling_specialist"}:
                tags["coordination"].append(tag)
                tags["intervention"].append(tag)
    return tags
