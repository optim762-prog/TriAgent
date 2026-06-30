"""
Generate tool_forcing_instances_n100.jsonl and the corresponding ground truth entries.
Run: python experiments/generate_toolforcing.py
Outputs:
  datasets/tool_forcing_instances_n100.jsonl   (100 instances)
  datasets/ground_truth_toolforcing_n100.jsonl (100 GT entries, append to ground_truth.jsonl)
"""
import json
from pathlib import Path

ROOT = Path(__file__).parent.parent

WEATHER_PHRASES = [
    "Current meteorological conditions are unknown and must be retrieved via real-time API query before any fungicide application decision can be made.",
    "No weather data is available for this location at the time of inspection — live conditions must be checked before proceeding.",
    "The agronomist on site did not record temperature or humidity; real-time weather retrieval is required to assess disease risk.",
    "Environmental data (temperature, humidity, wind speed) is unavailable — an API call is required before scheduling treatment.",
]

# ── P1 — Wheat Brown Rust ─────────────────────────────────────────────────────
P1_LOCS = ["Foggia, IT", "Bologna, IT", "Ankara, TR", "Thessaloniki, GR", "Seville, ES"]
P1_STAGES = ["tillering (Zadoks GS 20–29)", "stem elongation (Zadoks GS 30–37)",
             "heading (Zadoks GS 50–59)", "grain fill (Zadoks GS 70–77)", "ripening (Zadoks GS 83–87)"]
P1_SEVERITIES = [
    ("low",    "<15%",   "very few isolated lesions affecting less than 15% of the canopy"),
    ("medium", "30–40%", "moderate spread with approximately 30–40% of leaf area affected"),
    ("high",   "50–60%", "heavy infection with 50–60% of the canopy showing orange-yellow uredinia"),
    ("critical","60%+",  "severe outbreak with over 60% of leaves covered in confluent pustules"),
]

def p1_desc(loc, stage, sev_label, sev_pct, sev_desc, weather_phrase):
    return (
        f"Wheat crop at {stage}. Field inspection in {loc.split(',')[0]} reveals {sev_desc}. "
        f"Orange-yellow uredinia with chlorotic halos are visible on the adaxial leaf surface; "
        f"smaller darker telial pustules are present on the underside, consistent with "
        f"Puccinia triticina (brown rust). Severity approximately {sev_pct} of leaf area. "
        f"{weather_phrase}"
    )

# ── P2 — Tomato Late Blight ───────────────────────────────────────────────────
P2_LOCS = ["Naples, IT", "Catania, IT", "Valencia, ES", "Izmir, TR", "Athens, GR"]
P2_STAGES = ["vegetative (4–6 true leaves)", "flowering", "early fruit set",
             "mid fruit development", "mature green stage"]
P2_SEVERITIES = [
    ("early",    "few lesions", "scattered water-soaked lesions on 2–3 leaves per plant"),
    ("moderate", "expanding",  "water-soaked lesions on 20–30% of the canopy with white sporulation on leaf undersides"),
    ("severe",   "epidemic",   "rapidly expanding brown-black lesions on leaves and stems, white sporulation abundant, fruit infected"),
    ("critical", "systemic",   "whole-plant collapse on affected rows, brown-black stem cankers, advanced sporulation — full epidemic risk"),
]

def p2_desc(loc, stage, sev_label, sev_desc, weather_phrase):
    return (
        f"Tomato crop at {stage} in {loc.split(',')[0]}. Inspection reveals {sev_desc}. "
        f"Brown-black necrotic lesions are confirmed on stems and petioles consistent with "
        f"Phytophthora infestans (late blight). Diagnosis supported by field assessment. "
        f"{weather_phrase}"
    )

# ── P3 — Olive Fruit Fly ─────────────────────────────────────────────────────
P3_LOCS = ["Bari, IT", "Athens, GR", "Seville, ES", "Tunis, TN", "Nicosia, CY"]
P3_STAGES = ["pit hardening (July)", "early fruit development", "mid maturation",
             "pre-harvest (August)", "late harvest period"]
P3_INFESTATIONS = [
    ("threshold", "5–8%",  "5–8% oviposition sting rate on sampled fruits, near economic threshold"),
    ("moderate",  "12%",   "12% oviposition stings confirmed on sampled fruits, above economic threshold; larval galleries visible"),
    ("high",      "20%",   "20% of sampled fruits show oviposition stings; larval galleries confirmed in 15% of fruits"),
    ("severe",    "30%+",  "30% or more oviposition stings detected; extensive larval galleries and early exit holes confirm active infestation"),
]

def p3_desc(loc, stage, inf_label, inf_pct, inf_desc, weather_phrase):
    return (
        f"Olive grove ({stage}) in {loc.split(',')[0]}. Fruit sampling confirms Bactrocera oleae infestation: "
        f"{inf_desc}. Identity confirmed by characteristic oviposition punctures and sub-cuticular larval tunnels. "
        f"Deltamethrin application is being considered but efficacy is temperature- and wind-dependent. "
        f"{weather_phrase}"
    )

# ── P4 — Wheat Fusarium Head Blight ──────────────────────────────────────────
P4_LOCS = ["Bologna, IT", "Poznan, PL", "Plovdiv, BG", "Odessa, UA", "Krasnodar, RU"]
P4_STAGES = ["early anthesis (Zadoks GS 60–61)", "mid anthesis (Zadoks GS 64–65)",
             "late anthesis (Zadoks GS 67–69)", "early grain fill (Zadoks GS 71–73)",
             "mid grain fill (Zadoks GS 75–77)"]
P4_SEVERITIES = [
    ("low",      "5–10%",  "bleaching on 5–10% of spikelets; faint pink-orange sporodochia at spikelet bases on a few heads"),
    ("moderate", "20–30%", "20–30% of wheat heads show bleached spikelets with pink-orange sporodochia at the base, consistent with Fusarium head blight"),
    ("high",     "40–50%", "40–50% of sampled heads affected; heavy sporodochia confirmed; some grain shrivelling visible"),
    ("severe",   "60%+",   "over 60% of heads severely bleached with abundant sporodochia and visible mycelium; significant yield loss expected"),
]

def p4_desc(loc, stage, sev_label, sev_pct, sev_desc, weather_phrase):
    return (
        f"Wheat field at {stage} in {loc.split(',')[0]}. {sev_desc}. "
        f"Fusarium graminearum (head blight / scab) is the suspected pathogen based on "
        f"the characteristic bleaching pattern and spore coloration. "
        f"Fungicide application decision requires assessment of infection-period conditions. "
        f"{weather_phrase}"
    )

# ── P5 — Citrus Greening HLB ─────────────────────────────────────────────────
P5_LOCS = ["Palermo, IT", "Valencia, ES", "Limassol, CY", "Sfax, TN", "Agadir, MA"]
P5_STAGES = ["vegetative flush", "post-bloom fruit set", "early fruit development",
             "mid fruit development", "late season"]
P5_SEVERITIES = [
    ("suspect",   "1–2 trees",   "1–2 trees with asymmetric blotchy mottling not following vein patterns; lopsided small fruits"),
    ("confirmed", "5–10 trees",  "5–10 trees positive for HLB based on field diagnostic kit; lopsided asymmetric fruits and blotchy mottle on new flush"),
    ("spreading", "10–20%",      "10–20% of grove trees symptomatic; yellowing of new shoots; field kit positive; psyllid adults observed on flush"),
    ("epidemic",  "30%+",        "over 30% of trees symptomatic; psyllid population well-established on new flush; systemic spread confirmed"),
]

def p5_desc(loc, stage, sev_label, sev_desc, weather_phrase):
    return (
        f"Citrus grove ({stage}) in {loc.split(',')[0]}. Inspection confirms: {sev_desc}. "
        f"Candidatus Liberibacter asiaticus (HLB / citrus greening) is confirmed by field diagnostic kit. "
        f"Diaphorina citri psyllid vector management is a priority; spray efficacy depends on current temperature and wind. "
        f"{weather_phrase}"
    )


def generate():
    instances = []
    gt_entries = []

    # P1
    idx = 0
    for li, loc in enumerate(P1_LOCS):
        for vi, (sev_label, sev_pct, sev_desc) in enumerate(P1_SEVERITIES):
            iid = f"P1_TF_{idx+1:03d}"
            stage = P1_STAGES[(li * 4 + vi) % len(P1_STAGES)]
            phrase = WEATHER_PHRASES[(li + vi) % len(WEATHER_PHRASES)]
            desc = p1_desc(loc, stage, sev_label, sev_pct, sev_desc, phrase)
            instances.append({
                "instance_id": iid, "process_id": "P1",
                "request": {"description": desc, "crop": "wheat",
                            "location": loc, "growth_stage": stage, "severity": sev_label}
            })
            dose_min, dose_max = (0.4, 0.9) if sev_label in ("low", "medium") else (0.6, 1.2)
            gt_entries.append({
                "instance_id": iid, "process_id": "P1",
                "expected": {
                    "diagnosis_keywords": ["brown rust", "Puccinia triticina", "leaf rust"],
                    "treatment_substances": ["tebuconazole", "propiconazole", "triazole", "fungicide"],
                    "treatment_dose_per_ha": {"min": dose_min, "max": dose_max, "unit": "L/ha"},
                    "required_tool_calls": ["get_weather"],
                    "gateway_decisions": {},
                    "should_escalate": False
                }
            })
            idx += 1

    # P2
    idx = 0
    for li, loc in enumerate(P2_LOCS):
        for vi, (sev_label, sev_abbr, sev_desc) in enumerate(P2_SEVERITIES):
            iid = f"P2_TF_{idx+1:03d}"
            stage = P2_STAGES[(li * 4 + vi) % len(P2_STAGES)]
            phrase = WEATHER_PHRASES[(li + vi + 1) % len(WEATHER_PHRASES)]
            desc = p2_desc(loc, stage, sev_label, sev_desc, phrase)
            instances.append({
                "instance_id": iid, "process_id": "P2",
                "request": {"description": desc, "crop": "tomato",
                            "location": loc, "growth_stage": stage, "severity": sev_label}
            })
            gt_entries.append({
                "instance_id": iid, "process_id": "P2",
                "expected": {
                    "diagnosis_keywords": ["late blight", "Phytophthora infestans"],
                    "treatment_substances": ["metalaxyl", "chlorothalonil", "mancozeb", "mandipropamid", "fungicide"],
                    "treatment_dose_per_ha": {"min": 1.0, "max": 3.0, "unit": "L/ha"},
                    "required_tool_calls": ["get_weather"],
                    "gateway_decisions": {},
                    "should_escalate": False
                }
            })
            idx += 1

    # P3
    idx = 0
    for li, loc in enumerate(P3_LOCS):
        for vi, (inf_label, inf_pct, inf_desc) in enumerate(P3_INFESTATIONS):
            iid = f"P3_TF_{idx+1:03d}"
            stage = P3_STAGES[(li * 4 + vi) % len(P3_STAGES)]
            phrase = WEATHER_PHRASES[(li + vi + 2) % len(WEATHER_PHRASES)]
            desc = p3_desc(loc, stage, inf_label, inf_pct, inf_desc, phrase)
            instances.append({
                "instance_id": iid, "process_id": "P3",
                "request": {"description": desc, "crop": "olive",
                            "location": loc, "growth_stage": stage, "severity": inf_label}
            })
            gt_entries.append({
                "instance_id": iid, "process_id": "P3",
                "expected": {
                    "diagnosis_keywords": ["Bactrocera oleae", "olive fruit fly", "fruit fly"],
                    "treatment_substances": ["deltamethrin", "spinosad", "insecticide", "pyrethroid"],
                    "treatment_dose_per_ha": {"min": 25.0, "max": 50.0, "unit": "g/ha"},
                    "required_tool_calls": ["get_weather"],
                    "gateway_decisions": {},
                    "should_escalate": False
                }
            })
            idx += 1

    # P4
    idx = 0
    for li, loc in enumerate(P4_LOCS):
        for vi, (sev_label, sev_pct, sev_desc) in enumerate(P4_SEVERITIES):
            iid = f"P4_TF_{idx+1:03d}"
            stage = P4_STAGES[(li * 4 + vi) % len(P4_STAGES)]
            phrase = WEATHER_PHRASES[(li + vi + 3) % len(WEATHER_PHRASES)]
            desc = p4_desc(loc, stage, sev_label, sev_pct, sev_desc, phrase)
            instances.append({
                "instance_id": iid, "process_id": "P4",
                "request": {"description": desc, "crop": "wheat",
                            "location": loc, "growth_stage": stage, "severity": sev_label}
            })
            gt_entries.append({
                "instance_id": iid, "process_id": "P4",
                "expected": {
                    "diagnosis_keywords": ["fusarium", "head blight", "scab", "Fusarium graminearum"],
                    "treatment_substances": ["tebuconazole", "prothioconazole", "metconazole", "fungicide", "triazole"],
                    "treatment_dose_per_ha": {"min": 0.5, "max": 2.0, "unit": "L/ha"},
                    "required_tool_calls": ["get_weather"],
                    "gateway_decisions": {},
                    "should_escalate": False
                }
            })
            idx += 1

    # P5
    idx = 0
    for li, loc in enumerate(P5_LOCS):
        for vi, (sev_label, sev_abbr, sev_desc) in enumerate(P5_SEVERITIES):
            iid = f"P5_TF_{idx+1:03d}"
            stage = P5_STAGES[(li * 4 + vi) % len(P5_STAGES)]
            phrase = WEATHER_PHRASES[(li + vi) % len(WEATHER_PHRASES)]
            desc = p5_desc(loc, stage, sev_label, sev_desc, phrase)
            instances.append({
                "instance_id": iid, "process_id": "P5",
                "request": {"description": desc, "crop": "citrus",
                            "location": loc, "growth_stage": stage, "severity": sev_label}
            })
            gt_entries.append({
                "instance_id": iid, "process_id": "P5",
                "expected": {
                    "diagnosis_keywords": ["HLB", "citrus greening", "Liberibacter", "huanglongbing"],
                    "treatment_substances": ["imidacloprid", "thiamethoxam", "pyrethroid", "insecticide"],
                    "treatment_dose_per_ha": {"min": 0.1, "max": 0.5, "unit": "L/ha"},
                    "required_tool_calls": ["get_weather"],
                    "gateway_decisions": {},
                    "should_escalate": False
                }
            })
            idx += 1

    # Write instances
    out_inst = ROOT / "datasets" / "tool_forcing_instances_n100.jsonl"
    with open(out_inst, "w") as f:
        for inst in instances:
            f.write(json.dumps(inst) + "\n")
    print(f"Written {len(instances)} instances -> {out_inst}")

    # Write GT
    out_gt = ROOT / "datasets" / "ground_truth_toolforcing_n100.jsonl"
    with open(out_gt, "w") as f:
        for gt in gt_entries:
            f.write(json.dumps(gt) + "\n")
    print(f"Written {len(gt_entries)} GT entries -> {out_gt}")


if __name__ == "__main__":
    generate()
