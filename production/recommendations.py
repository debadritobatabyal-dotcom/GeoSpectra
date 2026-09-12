from typing import Any

def generate_recommendations(
    inputs: dict[str, Any],
    negative_drivers: list[dict] = None,
    risk_level: str = "LOW"
) -> list[dict]:
    recs = []
    neg_features = {d["feature"]: d for d in (negative_drivers or [])}

    def _get(key, default=0.0):
        val = inputs.get(key, default)
        try:
            return float(val) if val is not None else default
        except (ValueError, TypeError):
            return default

    active_shovels = _get("active_shovels", 3)
    total_shovels = _get("total_shovels", 4)
    shovel_avail = _get("shovel_availability_pct", 100.0)
    shovel_downtime = _get("shovel_downtime_hours", 0.0)
    shovel_breakdowns = _get("shovel_breakdown_events", 0)

    active_trucks = _get("active_trucks", 16)
    total_trucks = _get("total_trucks", 18)
    truck_avail = _get("truck_availability_pct", 90.0)
    truck_downtime = _get("truck_downtime_hours", 0.0)
    truck_breakdowns = _get("truck_breakdown_events", 0)

    cycle_time = _get("average_cycle_time_min", 24.0)
    queue_delay = _get("queue_delay_min", 2.0)
    haulage_efficiency = _get("haulage_efficiency_pct", 85.0)

    rain_mm = _get("rainfall_mm", 0.0)
    weather_disruption = _get("weather_disruption_hours", 0.0)
    ground_cond = _get("ground_condition_index", 5.0)

    blasting_delay = _get("blasting_delay_hours", 0.0)
    maint_delay = _get("maintenance_delay_hours", 0.0)
    loading_eff = _get("loading_efficiency_pct", 85.0)

    shovel_impact = (
        neg_features.get("active_shovels", {}).get("shap_impact_tonnes", 0.0) +
        neg_features.get("shovel_availability_pct", {}).get("shap_impact_tonnes", 0.0) +
        neg_features.get("shovel_downtime_hours", {}).get("shap_impact_tonnes", 0.0)
    )
    if active_shovels < total_shovels or shovel_avail < 85.0 or shovel_downtime > 1.5 or shovel_impact < -20.0:
        prio = "High" if (shovel_breakdowns > 0 or shovel_avail < 75.0 or shovel_impact < -150.0) else "Medium"
        shap_note = f" (SHAP model drag: {shovel_impact:+.0f} t/d)" if shovel_impact < -5.0 else ""
        recs.append({
            "category": "Loading Equipment Fleet",
            "priority": prio,
            "title": "Evaluate Shovel Availability & Face Allocation",
            "action": (
                "Consider reviewing shovel mechanical status, preventative maintenance "
                "turn-around, and bench readiness to optimize primary digging capacity."
            ),
            "rationale": (
                f"Active shovels at {int(active_shovels)}/{int(total_shovels)} "
                f"(availability: {shovel_avail:.1f}%, downtime: {shovel_downtime:.1f} hrs){shap_note}."
            )
        })

    queue_impact = (
        neg_features.get("queue_delay_min", {}).get("shap_impact_tonnes", 0.0) +
        neg_features.get("truck_cycles", {}).get("shap_impact_tonnes", 0.0)
    )
    if queue_delay > 2.5 or queue_impact < -25.0:
        prio = "High" if (queue_delay > 4.5 or queue_impact < -100.0) else "Medium"
        shap_note = f" (SHAP model drag: {queue_impact:+.0f} t/d)" if queue_impact < -5.0 else ""
        recs.append({
            "category": "Dispatch & Traffic Control",
            "priority": prio,
            "title": "Review Truck-Shovel Allocation & Dispatch Sequencing",
            "action": (
                "Consider evaluating shovel queue distribution, dumping point congestion, "
                "and dynamic truck reallocation to minimize standing idle time at loading faces."
            ),
            "rationale": f"Queue delay recorded at {queue_delay:.1f} min/cycle{shap_note}."
        })

    cycle_impact = (
        neg_features.get("average_cycle_time_min", {}).get("shap_impact_tonnes", 0.0) +
        neg_features.get("haulage_efficiency_pct", {}).get("shap_impact_tonnes", 0.0)
    )
    if cycle_time > 26.0 or haulage_efficiency < 80.0 or cycle_impact < -25.0:
        shap_note = f" (SHAP model drag: {cycle_impact:+.0f} t/d)" if cycle_impact < -5.0 else ""
        recs.append({
            "category": "Haulage Efficiency",
            "priority": "Medium",
            "title": "Review Haul Routes, Road Conditions & Travel Times",
            "action": (
                "Consider reviewing haul road maintenance, ramp grading, and truck travel speeds "
                "to streamline total cycle duration."
            ),
            "rationale": (
                f"Average cycle time at {cycle_time:.1f} min "
                f"(haulage efficiency: {haulage_efficiency:.1f}%){shap_note}."
            )
        })

    weather_impact = (
        neg_features.get("weather_disruption_hours", {}).get("shap_impact_tonnes", 0.0) +
        neg_features.get("rainfall_mm", {}).get("shap_impact_tonnes", 0.0) +
        neg_features.get("ground_condition_index", {}).get("shap_impact_tonnes", 0.0)
    )
    if weather_disruption > 0.5 or rain_mm > 5.0 or ground_cond > 20.0 or weather_impact < -30.0:
        prio = "High" if (weather_disruption > 2.5 or rain_mm > 20.0 or weather_impact < -150.0) else "Medium"
        shap_note = f" (SHAP model drag: {weather_impact:+.0f} t/d)" if weather_impact < -5.0 else ""
        recs.append({
            "category": "Environmental Hazards",
            "priority": prio,
            "title": "Review Weather-Related Haulage Constraints & Road Drainage",
            "action": (
                "Consider reviewing wet-weather operating protocols, pit sump pumping, "
                "and temporary speed restrictions to maintain safe haul ramp transit."
            ),
            "rationale": (
                f"Rainfall recorded at {rain_mm:.1f} mm with {weather_disruption:.1f} hrs "
                f"disruption and ground index {ground_cond:.1f}{shap_note}."
            )
        })

    truck_impact = (
        neg_features.get("active_trucks", {}).get("shap_impact_tonnes", 0.0) +
        neg_features.get("truck_availability_pct", {}).get("shap_impact_tonnes", 0.0)
    )
    if active_trucks < total_trucks * 0.82 or truck_avail < 82.0 or truck_downtime > 5.0 or truck_impact < -30.0:
        prio = "High" if (truck_breakdowns > 0 or truck_avail < 75.0) else "Medium"
        shap_note = f" (SHAP model drag: {truck_impact:+.0f} t/d)" if truck_impact < -5.0 else ""
        recs.append({
            "category": "Haulage Fleet Readiness",
            "priority": prio,
            "title": "Review Truck Fleet Allocation & Maintenance Queue",
            "action": (
                "Consider evaluating truck assignment, workshop turn-around times, and "
                "standby fleet readiness to maintain target cycle volume."
            ),
            "rationale": (
                f"Active haul trucks at {int(active_trucks)}/{int(total_trucks)} "
                f"(availability: {truck_avail:.1f}%){shap_note}."
            )
        })

    if blasting_delay > 0.5 or maint_delay > 1.5:
        recs.append({
            "category": "Operations & Blasting",
            "priority": "Standard",
            "title": "Review Blasting Schedules & Muckpile Clearance",
            "action": (
                "Consider aligning blasting clearance windows with shift handover intervals "
                "to minimize interrupted loading and hauling operations."
            ),
            "rationale": f"Blasting delays recorded at {blasting_delay:.1f} hrs; maintenance delays at {maint_delay:.1f} hrs."
        })

    if not recs:
        recs.append({
            "category": "Standard Operations",
            "priority": "Standard",
            "title": "Maintain Baseline Operational Dispatch Cadence",
            "action": (
                "Operating metrics and SHAP indicators remain within established historical control bounds. "
                "Continue standard shift dispatch monitoring and routine preventative maintenance."
            ),
            "rationale": "Equipment availability, cycle timings, and environmental conditions are stable."
        })

    return recs
