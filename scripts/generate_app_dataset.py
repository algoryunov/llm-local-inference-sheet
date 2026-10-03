#!/usr/bin/env python3
"""Generate the synthetic application-quality dataset (datasets/app-v1).

All data is synthetic, produced by seeded templates. Labels follow deterministically
from the generation rules, and those rules are stated verbatim in each task's system
prompt, so a correct answer is always derivable from the input alone.

    python scripts/generate_app_dataset.py            # writes datasets/app-v1/{dev,test}.jsonl

Languages are language packs: English is defined below. Additional packs found in the
git-ignored private/langpacks/app_*.py are generated into private/datasets/app-v1 (all languages),
while datasets/app-v1 always contains English only. Each (task, language) has its own seeded RNG,
so adding a language never changes the English items.

Changing anything here requires bumping VERSION (and the directory name).
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import random
from pathlib import Path

VERSION = "app-v1"
REPO = Path(__file__).resolve().parents[1]
PUBLIC_OUT = REPO / "datasets" / VERSION
PRIVATE_OUT = REPO / "private" / "datasets" / VERSION
SPLITS = {"dev": (6, 101), "test": (20, 202)}  # items per task per language, seed
ACTIONS = ["create_ticket", "check_order_status", "cancel_order", "update_address", "escalate_to_human", "none"]

# A pack value that is a str is used as-is (no RNG draw); a list is sampled with rng.choice.
EN: dict = {
    "extraction_system": (
        "Extract order information from the customer message. Respond with only a JSON object with "
        "exactly these keys: customer_name (string or null), order_id (string or null), product (string or "
        "null), quantity (integer or null), delivery_date (string in YYYY-MM-DD format or null), priority "
        "(\"low\", \"normal\", \"high\" or null).\n"
        "Rules: use null when a value is not stated or is ambiguous (for example \"a few\", \"several\" or "
        "\"next week\"). Map \"urgent\" or \"ASAP\" to \"high\", \"no rush\" to \"low\" and \"standard "
        "handling\" to \"normal\". Copy names, order IDs and product names exactly as written."
    ),
    "names": ["Maria Lopez", "John Carter", "Aisha Khan", "Tom Becker", "Lena Novak", "Carlos Silva", "Emily Zhou", "Omar Haddad"],
    "products": ["thermal camera", "network switch", "USB-C dock", "label printer", "barcode scanner", "PoE injector",
                 "edge GPU module", "IP camera mount"],
    "months": ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October",
               "November", "December"],
    "name_sentence": "My name is {name}.",
    "order_sentence": "This is about order {oid}.",
    "qty_present": "We need {q} units of the item \"{prod}\".",
    "qty_vague_options": ["a few", "several", "two or three"],
    "qty_vague": "We need {vague} units of the item \"{prod}\".",
    "qty_absent": "The item is \"{prod}\".",
    "date_present": "Please deliver by {month} {d}, {y}.",
    "date_ambiguous": ["Delivery sometime next week works.", "Please deliver soon-ish, maybe early next month."],
    "priority_high": ["This is urgent.", "We need it ASAP."],
    "priority_low": "No rush on this one.",
    "priority_normal": "Standard handling is fine.",
    "noise": ["Thanks in advance!", "Our office is on the third floor.", "Last time the packaging was great."],
    "routing_system": (
        "You route customer-support requests to exactly one action. Actions:\n"
        "- create_ticket: the customer reports a malfunction, defect or technical problem with a product.\n"
        "- check_order_status: the customer asks where an order is or when it will arrive.\n"
        "- cancel_order: the customer wants to cancel an order.\n"
        "- update_address: the customer wants to change a delivery address.\n"
        "- escalate_to_human: the customer explicitly asks for a human agent or a manager.\n"
        "- none: no tool is needed (greetings, thanks, or questions unrelated to orders and products).\n"
        "Respond with only a JSON object: {\"action\": \"<one action name>\"}."
    ),
    "routing_templates": {
        "create_ticket": ["The {p} I bought stopped working after two days.", "My {p} keeps rebooting every few minutes, can you help?",
                          "The {p} arrived with a cracked housing and won't power on.", "Our {p} shows an error light and no output."],
        "check_order_status": ["Where is my order {o}?", "When will order {o} arrive?", "Can you tell me the delivery status of {o}?",
                               "It's been a week, has {o} shipped yet?"],
        "cancel_order": ["Please cancel order {o}.", "I changed my mind, cancel {o} please.", "I no longer need order {o}, please cancel it.",
                         "Cancel my order {o} before it ships."],
        "update_address": ["Please change the delivery address for {o} to 12 Harbor Street.", "We moved; update the shipping address on {o}.",
                           "Can you send {o} to our new warehouse at 5 Mill Road instead?", "Wrong address on {o}, it should go to 44 Pine Ave."],
        "escalate_to_human": ["I want to talk to a real person.", "Connect me with a manager, please.", "Stop the bot, get me a human agent.",
                              "I need to speak to someone from your team directly."],
        "none": ["Thanks, that's all for today!", "Hello!", "What's the capital of Australia?", "Tell me a joke about cats."],
    },
    "routing_products": None,  # None: reuse "products"
    "cv_intro": "Event log:",
}
EN["cv_system"] = None  # filled below from CV_SYSTEM_EN


def pick(rng: random.Random, v: str | list[str]) -> str:
    return v if isinstance(v, str) else rng.choice(v)


def gen_extraction(rng: random.Random, P: dict, idx: int) -> dict:
    exp: dict = {}
    parts: list[str] = []
    tags: list[str] = []
    if rng.random() < 0.8:
        name = rng.choice(P["names"])
        exp["customer_name"] = name
        parts.append(P["name_sentence"].format(name=name))
    else:
        exp["customer_name"] = None
        tags.append("absent:customer_name")
    if rng.random() < 0.75:
        oid = f"ORD-{rng.randint(10000, 99999)}"
        exp["order_id"] = oid
        parts.append(P["order_sentence"].format(oid=oid))
    else:
        exp["order_id"] = None
        tags.append("absent:order_id")
    prod = rng.choice(P["products"])  # quoted/colon forms keep product names uninflected
    exp["product"] = prod
    mode = rng.choices(["present", "absent", "ambiguous"], [0.7, 0.15, 0.15])[0]
    if mode == "present":
        q = rng.randint(1, 40)
        exp["quantity"] = q
        parts.append(P["qty_present"].format(q=q, prod=prod))
    elif mode == "ambiguous":
        exp["quantity"] = None
        vague = rng.choice(P["qty_vague_options"])
        parts.append(P["qty_vague"].format(vague=vague, prod=prod))
        tags.append("ambiguous:quantity")
    else:
        exp["quantity"] = None
        parts.append(P["qty_absent"].format(prod=prod))
        tags.append("absent:quantity")
    mode = rng.choices(["present", "absent", "ambiguous"], [0.65, 0.2, 0.15])[0]
    if mode == "present":
        y, m, d = 2026, rng.randint(1, 12), rng.randint(1, 28)
        exp["delivery_date"] = f"{y:04d}-{m:02d}-{d:02d}"
        parts.append(P["date_present"].format(month=P["months"][m - 1], d=d, y=y))
    elif mode == "ambiguous":
        exp["delivery_date"] = None
        parts.append(pick(rng, P["date_ambiguous"]))
        tags.append("ambiguous:delivery_date")
    else:
        exp["delivery_date"] = None
        tags.append("absent:delivery_date")
    pr = rng.choices(["high", "low", "normal", None], [0.3, 0.2, 0.2, 0.3])[0]
    exp["priority"] = pr
    if pr is not None:
        parts.append(pick(rng, P[f"priority_{pr}"]))
    else:
        tags.append("absent:priority")
    parts.append(pick(rng, P["noise"]))
    first, rest = parts[:1], parts[1:]
    rng.shuffle(rest)
    text = " ".join(first + rest)
    ordered = {k: exp[k] for k in ("customer_name", "order_id", "product", "quantity", "delivery_date", "priority")}
    return {"task": "extraction", "messages": [{"role": "system", "content": P["extraction_system"]},
                                                {"role": "user", "content": text}],
            "expected": ordered, "tags": tags}


def gen_routing(rng: random.Random, P: dict, idx: int) -> dict:
    action = ACTIONS[idx % len(ACTIONS)]
    tpl = rng.choice(P["routing_templates"][action])
    p = rng.choice(P["routing_products"] or P["products"])
    text = tpl.format(p=p, P=p.capitalize(), o=f"ORD-{rng.randint(10000, 99999)}")
    return {"task": "routing", "messages": [{"role": "system", "content": P["routing_system"]},
                                             {"role": "user", "content": text}],
            "expected": {"action": action}, "tags": []}


CV_SYSTEM_EN = (
    "You convert video-analytics event logs into a factual incident report. Use only the events in the log.\n"
    "Incident types (choose one):\n"
    "- unauthorized_zone_entry: a person enters a zone whose name starts with \"restricted\".\n"
    "- loitering: the same person (track_id) has person_dwell events in one zone spanning 60 seconds or more.\n"
    "- vehicle_blocking_exit: a vehicle has vehicle_stopped events in the \"fire exit\" zone spanning 120 seconds or more.\n"
    "- no_incident: none of the above.\n"
    "Key events are the events that establish the incident. Respond with only a JSON object with keys:\n"
    "incident_type, start_time (time of the first key event, or null for no_incident), end_time (time of the last key "
    "event, or null), camera_ids (sorted list of cameras of key events), event_refs (sorted list of key event ids), "
    "person_count (number of distinct person track_ids among key events), vehicle_involved (true/false), "
    "uncertain (true if any key event has confidence below 0.6), missing_evidence (sorted list of camera ids that "
    "have a camera_offline event between start_time and end_time inclusive; empty list otherwise), summary (one or "
    "two factual sentences in English)."
)
EN["cv_system"] = CV_SYSTEM_EN


def _ts(sec: int) -> str:
    return f"{8 + sec // 3600:02d}:{(sec // 60) % 60:02d}:{sec % 60:02d}"


def gen_cv(rng: random.Random, P: dict, idx: int) -> dict:
    kind = ["unauthorized_zone_entry", "loitering", "vehicle_blocking_exit", "no_incident"][idx % 4]
    cams = ["cam-1", "cam-2", "cam-3", "cam-4"]
    events: list[dict] = []
    key: list[dict] = []
    t0 = rng.randint(0, 3000)
    low_conf = rng.random() < 0.3
    offline = rng.random() < 0.3

    def ev(t: int, cam: str, typ: str, zone: str | None, obj: str | None, track: int | None, conf: float) -> dict:
        e = {"t": t, "camera": cam, "type": typ, "confidence": round(conf, 2)}
        if zone:
            e["zone"] = zone
        if obj:
            e["object"] = obj
        if track is not None:
            e["track_id"] = track
        events.append(e)
        return e

    person_tracks: set[int] = set()
    vehicle = False
    if kind == "unauthorized_zone_entry":
        cam = rng.choice(cams)
        zone = rng.choice(["restricted server room", "restricted loading bay"])
        n_people = rng.choice([1, 1, 2])
        for k in range(n_people):
            tr = rng.randint(10, 99) + 100 * k
            person_tracks.add(tr)
            key.append(ev(t0 + 5 * k, cam, "person_enter_zone", zone, "person", tr, 0.55 if (low_conf and k == 0) else rng.uniform(0.75, 0.98)))
        key.append(ev(t0 + rng.randint(30, 200), cam, "person_exit_zone", zone, "person", min(person_tracks), rng.uniform(0.75, 0.98)))
    elif kind == "loitering":
        cam = rng.choice(cams)
        zone = rng.choice(["lobby", "parking entrance", "atm area"])
        tr = rng.randint(10, 99)
        person_tracks.add(tr)
        span = rng.choice([60, 75, 90, 120])
        n = span // 15 + 1
        for k in range(n):
            key.append(ev(t0 + 15 * k, cam, "person_dwell", zone, "person", tr,
                          0.52 if (low_conf and k == n // 2) else rng.uniform(0.7, 0.97)))
    elif kind == "vehicle_blocking_exit":
        cam = rng.choice(cams)
        tr = rng.randint(200, 299)
        vehicle = True
        span = rng.choice([120, 150, 180, 240])
        n = span // 30 + 1
        for k in range(n):
            key.append(ev(t0 + 30 * k, cam, "vehicle_stopped", "fire exit", "vehicle", tr,
                          0.5 if (low_conf and k == 0) else rng.uniform(0.7, 0.97)))
    # distractors (never satisfy an incident rule); unique track ids so no accidental 60 s dwell forms
    next_track = iter(range(300 + 10 * rng.randint(0, 9), 10_000))
    for _ in range(rng.randint(3, 6)):
        cam = rng.choice(cams)
        t = t0 + rng.randint(-120, 260)
        choice = rng.random()
        if choice < 0.4:
            ev(t, cam, "person_enter_zone", rng.choice(["lobby", "corridor", "canteen"]), "person", next(next_track), rng.uniform(0.6, 0.99))
        elif choice < 0.7:
            ev(t, cam, "vehicle_moving", rng.choice(["parking", "driveway"]), "vehicle", next(next_track), rng.uniform(0.6, 0.99))
        else:
            # short dwell (< 60 s span) in a non-restricted zone: not loitering
            tr = next(next_track)
            ev(t, cam, "person_dwell", "corridor", "person", tr, rng.uniform(0.6, 0.99))
            ev(t + 20, cam, "person_dwell", "corridor", "person", tr, rng.uniform(0.6, 0.99))
    if kind == "no_incident":
        key = []
    if key:
        start, end = min(e["t"] for e in key), max(e["t"] for e in key)
    else:
        start = end = None
    if offline:
        oc = rng.choice(cams)
        ot = (start + (end - start) // 2) if start is not None else t0 + 10
        ev(ot, oc, "camera_offline", None, None, None, 1.0)
    events.sort(key=lambda e: (e["t"], e["camera"], e["type"]))
    for i, e in enumerate(events, 1):
        e["id"] = f"e{i:02d}"
    log = [{"id": e["id"], "t": _ts(e["t"]), **{k: e[k] for k in ("camera", "type", "zone", "object", "track_id", "confidence") if k in e}}
           for e in events]
    missing = sorted({e["camera"] for e in events if e["type"] == "camera_offline" and start is not None and start <= e["t"] <= end})
    expected = {
        "incident_type": kind,
        "start_time": _ts(start) if start is not None else None,
        "end_time": _ts(end) if end is not None else None,
        "camera_ids": sorted({e["camera"] for e in key}),
        "event_refs": sorted(e["id"] for e in key),
        "person_count": len(person_tracks) if kind != "no_incident" else 0,
        "vehicle_involved": vehicle,
        "uncertain": any(e["confidence"] < 0.6 for e in key),
        "missing_evidence": missing,
    }
    user = f"{P['cv_intro']}\n{json.dumps({'site': 'warehouse-7', 'events': log}, ensure_ascii=False, indent=1)}"
    tags = [f"kind:{kind}"] + (["low_confidence"] if expected["uncertain"] else []) + (["camera_offline"] if missing else [])
    return {"task": "cv_report", "messages": [{"role": "system", "content": P["cv_system"]},
                                               {"role": "user", "content": user}],
            "expected": expected, "tags": tags}


GENERATORS = {"extraction": gen_extraction, "routing": gen_routing, "cv_report": gen_cv}


def load_private_packs() -> dict[str, dict]:
    packs = {}
    for path in sorted((REPO / "private" / "langpacks").glob("app_*.py")):
        spec = importlib.util.spec_from_file_location(path.stem, path)
        assert spec and spec.loader
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        packs[mod.LANG] = mod.PACK
    return packs


def write(out: Path, packs: dict[str, dict], note: str) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    manifest = {"version": VERSION, "synthetic": True, "languages": list(packs), "note": note, "splits": {}}
    for split, (n, seed) in SPLITS.items():
        rows = []
        for task, gen in GENERATORS.items():
            for lang, P in packs.items():
                rng = random.Random(f"{seed}-{task}-{lang}")
                for i in range(n):
                    item = gen(rng, P, i)
                    item.update({"id": f"{VERSION}-{split}-{task}-{lang}-{i:03d}", "lang": lang, "split": split})
                    rows.append(item)
        path = out / f"{split}.jsonl"
        with path.open("w") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        manifest["splits"][split] = {"items": len(rows), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def main() -> None:
    print(json.dumps(write(PUBLIC_OUT, {"en": EN}, "public: English only"), indent=2))
    private = load_private_packs()
    if private:
        m = write(PRIVATE_OUT, {"en": EN, **private}, "private: includes git-ignored language packs")
        print(f"private dataset written to {PRIVATE_OUT} ({m['languages']})")


if __name__ == "__main__":
    main()
