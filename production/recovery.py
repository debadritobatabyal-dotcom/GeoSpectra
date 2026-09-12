from typing import Any, Callable

RECOVERY_SCENARIOS = {
    "improve_shovel_availability": {
        "name": "Improve Shovel Availability & Face Allocation",
        "icon": "🚜",
        "description": "Restore an unavailable or breakdown shovel to active service, clearing loading face starvation.",
        "category": "Equipment",
        "controllable_levers": ["active_shovels", "available_shovels", "shovel_availability_pct", "shovel_downtime_hours"]
    },
    "reduce_queue_delay": {
        "name": "Optimize Dispatch & Reduce Queue Delays",
        "icon": "⏱️",
        "description": "Dynamic truck allocation and shovel bunching mitigation to curb idle standing time at loading faces.",
        "category": "Dispatch",
        "controllable_levers": ["queue_delay_min", "average_cycle_time_min", "truck_cycles"]
    },
    "improve_haulage_efficiency": {
        "name": "Improve Haul Route Efficiency & Road Maintenance",
        "icon": "🛣️",
        "description": "Grade main haul ramps, minimize rolling resistance, and optimize truck travel speeds.",
        "category": "Haulage",
        "controllable_levers": ["haulage_efficiency_pct", "average_cycle_time_min", "haulage_delay_hours"]
    },
    "reduce_cycle_time": {
        "name": "Targeted Cycle Time Compression",
        "icon": "🔄",
        "description": "Streamline loading spot times, dump pocket access, and ramp transit to shorten total cycle duration.",
        "category": "Operations",
        "controllable_levers": ["average_loading_time_min", "average_dumping_time_min", "average_cycle_time_min", "truck_cycles"]
    },
    "combined_recovery": {
        "name": "Integrated Multi-Lever Operational Recovery",
        "icon": "⚡",
        "description": "Coordinated dispatch re-sequencing, preventative shovel maintenance return, and haul road grading.",
        "category": "Combined",
        "controllable_levers": ["active_shovels", "queue_delay_min", "haulage_efficiency_pct", "average_cycle_time_min"]
    }
}

def list_recovery_scenarios() -> list[dict]:
    return [{"key": k, **v} for k, v in RECOVERY_SCENARIOS.items()]

def apply_recovery_modifications(
    inputs: dict[str, Any],
    scenario_key: str
) -> tuple[dict[str, Any], list[dict]]:
    data = dict(inputs)
    mods = []

    def _get(key, default=0.0):
        val = data.get(key, default)
        try:
            return float(val) if val is not None else default
        except (ValueError, TypeError):
            return default

    def _set(key, new_val, reason):
        old_val = data.get(key)
        data[key] = new_val
        mods.append({
            "parameter": key,
            "original_value": old_val,
            "modified_value": new_val,
            "reason": reason
        })

    total_shovels = int(_get("total_shovels", 4))
    available_shovels = int(_get("available_shovels", 3))
    active_shovels = int(_get("active_shovels", 2))

    total_trucks = int(_get("total_trucks", 18))
    available_trucks = int(_get("available_trucks", 16))
    active_trucks = int(_get("active_trucks", 14))

    if scenario_key == "improve_shovel_availability":

        if active_shovels < total_shovels:
            new_active = min(total_shovels, active_shovels + 1)
            new_avail = max(available_shovels, new_active)
            _set("active_shovels", new_active, "Returned idle/maintained shovel to active pit face")
            _set("available_shovels", new_avail, "Shovel mechanical inspection passed")

            new_avail_pct = min(100.0, round((new_avail / max(1, total_shovels)) * 100.0, 1))
            _set("shovel_availability_pct", new_avail_pct, "Fleet availability restored")

            old_downtime = _get("shovel_downtime_hours", 0.0)
            new_downtime = max(0.0, round(old_downtime * 0.4, 1))
            _set("shovel_downtime_hours", new_downtime, "Unscheduled mechanical downtime minimized")

            old_util = _get("shovel_utilization_pct", 75.0)
            new_util = min(95.0, round(max(old_util, 88.0), 1))
            _set("shovel_utilization_pct", new_util, "Face digging utilization elevated")

            cur_queue = _get("queue_delay_min", 2.5)
            if cur_queue > 1.8:
                new_q = max(1.2, round(cur_queue - 1.0, 1))
                _set("queue_delay_min", new_q, "Reduced truck bunching with extra face open")

    elif scenario_key == "reduce_queue_delay":
        cur_q = _get("queue_delay_min", 3.0)

        if cur_q > 1.4:
            q_reduction = round(cur_q - 1.3, 1)
            new_q = max(0.8, round(cur_q - q_reduction, 1))
            _set("queue_delay_min", new_q, "Dynamic dispatch reallocation eliminated truck bunching")

            cur_cycle = _get("average_cycle_time_min", 25.0)
            new_cycle = max(16.0, round(cur_cycle - q_reduction, 1))
            _set("average_cycle_time_min", new_cycle, "Direct cycle time reduction from queue elimination")

            cur_cycles = int(_get("truck_cycles", 40))
            add_cycles = max(1, int(round((q_reduction / max(1.0, cur_cycle)) * cur_cycles)))
            _set("truck_cycles", cur_cycles + add_cycles, "Additional haul cycles completed")

            cur_eff = _get("haulage_efficiency_pct", 78.0)
            new_eff = min(95.0, round(cur_eff + 8.0, 1))
            _set("haulage_efficiency_pct", new_eff, "Smoother cycle flow and less idle waiting")

    elif scenario_key == "improve_haulage_efficiency":
        cur_eff = _get("haulage_efficiency_pct", 78.0)
        new_eff = min(94.0, round(max(cur_eff + 12.0, 88.0), 1))
        _set("haulage_efficiency_pct", new_eff, "Haul road grading and rolling resistance reduction")

        cur_cycle = _get("average_cycle_time_min", 26.0)
        new_cycle = max(18.0, round(cur_cycle * 0.90, 1))
        _set("average_cycle_time_min", new_cycle, "Smoother ramp travel and optimized truck speed")

        cur_loaded = _get("average_loaded_travel_time_min", 8.5)
        new_loaded = max(5.0, round(cur_loaded * 0.90, 1))
        _set("average_loaded_travel_time_min", new_loaded, "Improved loaded grade ascent speed")

        cur_delay = _get("haulage_delay_hours", 0.8)
        new_delay = max(0.0, round(cur_delay * 0.3, 1))
        _set("haulage_delay_hours", new_delay, "Haul road hold-ups and intersection delays mitigated")

    elif scenario_key == "reduce_cycle_time":
        cur_cycle = _get("average_cycle_time_min", 25.0)

        time_saved = round(cur_cycle * 0.12, 1)
        new_cycle = max(17.0, round(cur_cycle - time_saved, 1))
        _set("average_cycle_time_min", new_cycle, "Total cycle duration streamlined by 12%")

        cur_load_t = _get("average_loading_time_min", 4.5)
        new_load_t = max(3.0, round(cur_load_t * 0.90, 1))
        _set("average_loading_time_min", new_load_t, "Optimal bucket pass count and spot positioning")

        cur_dump_t = _get("average_dumping_time_min", 2.2)
        new_dump_t = max(1.2, round(cur_dump_t * 0.85, 1))
        _set("average_dumping_time_min", new_dump_t, "Clear tip-head spotter sequencing")

        cur_cycles = int(_get("truck_cycles", 40))
        _set("truck_cycles", cur_cycles + 4, "Faster cycle velocity allows 4 additional turns")

    elif scenario_key == "combined_recovery":

        if active_shovels < total_shovels:
            new_active = min(total_shovels, active_shovels + 1)
            new_avail = max(available_shovels, new_active)
            _set("active_shovels", new_active, "Added active shovel face")
            _set("available_shovels", new_avail, "Completed preventative service")
            _set("shovel_availability_pct", min(100.0, round((new_avail / max(1, total_shovels)) * 100.0, 1)), "Fleet availability lifted")

        cur_q = _get("queue_delay_min", 3.0)
        new_q = max(1.1, round(cur_q * 0.5, 1))
        _set("queue_delay_min", new_q, "Dynamic dispatch allocation")

        cur_eff = _get("haulage_efficiency_pct", 78.0)
        _set("haulage_efficiency_pct", min(94.0, round(max(cur_eff + 10.0, 89.0), 1)), "Graded ramps and haul routes")

        cur_cycle = _get("average_cycle_time_min", 25.0)
        _set("average_cycle_time_min", max(18.0, round(cur_cycle * 0.88, 1)), "Combined turn-around velocity")

        cur_load_eff = _get("loading_efficiency_pct", 80.0)
        _set("loading_efficiency_pct", min(95.0, round(cur_load_eff + 8.0, 1)), "Bench preparation & floor grading")

    return data, mods

def simulate_recovery_scenario(
    baseline_inputs: dict[str, Any],
    scenario_key: str,
    predict_fn: Callable[[dict[str, Any]], dict[str, Any]],
    baseline_prediction: dict[str, Any] = None
) -> dict[str, Any]:
    meta = RECOVERY_SCENARIOS.get(scenario_key, RECOVERY_SCENARIOS["improve_shovel_availability"])

    b_pred = baseline_prediction or predict_fn(baseline_inputs)
    baseline_expected = b_pred["expected_production_tpd"]
    planned_target = b_pred["planned_production_tpd"]
    baseline_shortfall = b_pred["shortfall_tonnes"]
    baseline_risk = b_pred["risk_level"]

    mod_inputs, applied_mods = apply_recovery_modifications(baseline_inputs, scenario_key)

    s_pred = predict_fn(mod_inputs)
    scenario_expected = s_pred["expected_production_tpd"]

    recovery_tonnes = round(max(0.0, scenario_expected - baseline_expected), 1)
    recovery_pct = round((recovery_tonnes / max(1.0, baseline_expected)) * 100.0, 2)

    remaining_shortfall = round(max(0.0, planned_target - scenario_expected), 1)
    remaining_shortfall_pct = round((remaining_shortfall / max(1.0, planned_target)) * 100.0, 2)

    new_risk = s_pred["risk_level"]
    new_risk_color = s_pred["risk_color"]

    risk_rank = {"LOW": 1, "MEDIUM": 2, "HIGH": 3}
    risk_improved = risk_rank.get(new_risk, 2) < risk_rank.get(baseline_risk, 2)

    return {
        "scenario_key": scenario_key,
        "scenario_name": meta["name"],
        "scenario_icon": meta["icon"],
        "scenario_description": meta["description"],
        "category": meta["category"],
        "baseline_expected_tpd": baseline_expected,
        "scenario_expected_tpd": scenario_expected,
        "recovery_tonnes": recovery_tonnes,
        "recovery_pct": recovery_pct,
        "planned_target_tpd": planned_target,
        "baseline_shortfall_tonnes": baseline_shortfall,
        "remaining_shortfall_tonnes": remaining_shortfall,
        "remaining_shortfall_pct": remaining_shortfall_pct,
        "baseline_risk_level": baseline_risk,
        "new_risk_level": new_risk,
        "new_risk_color": new_risk_color,
        "risk_improved": risk_improved,
        "applied_modifications": applied_mods,
        "scenario_prediction": s_pred,
        "disclaimer": (
            "Modelled recovery values represent statistical estimates generated by re-evaluating "
            "the trained XGBoost regressor under modified operational inputs. They do not constitute "
            "deterministic guarantees of physical mining production."
        )
    }
