"""
LLM-as-Judge Validation: Cohen's Kappa
============================================
AfriMed Tutor | ICS554 | Ashesi University

STEPS TO FOLLOW:
--------------------
Assume there are 4 people responsible for ranking the responses generated
Step 1: Run scoring session for each team member:
        python eval/kappa_scoring.py --mode score

Step 2: After all 4 members have scored, generate the report:
        python eval/kappa_scoring.py --mode report
"""

import os
import json
import argparse
import textwrap
import re
from pathlib import Path

try:
    from sklearn.metrics import cohen_kappa_score
    import numpy as np
except ImportError:
    print("Installing required packages...")
    os.system("pip install scikit-learn numpy -q")
    from sklearn.metrics import cohen_kappa_score
    import numpy as np

# ─────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────

SCORES_DIR = Path("results/scores")
SAMPLE_FILE = Path("results/qualitative_sample.md")
REPORT_FILE = Path("results/kappa_report.md")
NUM_ENTRIES = 10
TEAM_SIZE = 4

# LLM Judge scores extracted from qualitative_sample.md
LLM_JUDGE = {
    "groundedness":     [1, 1, 0, 0, 0, 0, 1, 0, 0, 0],
    "citation_accuracy":[2, 1, 2, 2, 1, 2, 1, 1, 2, 2],
    "consistency":      [2, 2, 2, 2, 0, 1, 2, 2, 1, 2],
}

# Entry descriptions for display
ENTRY_DESCRIPTIONS = [
    "Entry 1  — HIV/AIDS secondary immunodeficiency",
    "Entry 2  — Hypovolaemic shock (sodium nitroprusside)",
    "Entry 3  — Aortic stenosis (SAD triad)",
    "Entry 4  — Dermatomyositis (discoid rash)",
    "Entry 5  — Pneumothorax (surgical indication)",
    "Entry 6  — POMC biosynthesis sites",
    "Entry 7  — Sickle cell disease (HbAC genotype)",
    "Entry 8  — Osteoporosis (vertebral fracture)",
    "Entry 9  — Platelet dense granules (serotonin)",
    "Entry 10 — Kawasaki's Disease (generalised lymphadenopathy)",
]

# ─────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────

def divider(char="─", width=65):
    print(char * width)

def header(title):
    divider("═")
    print(f"  {title}")
    divider("═")

def parse_entries(md_path: Path) -> list[dict]:
    """Parse the first NUM_ENTRIES entries from qualitative_sample.md."""
    text = md_path.read_text(encoding="utf-8")
    entries = []

    # Split on entry headers
    parts = re.split(r"\n## Entry (\d+)", text)
    # parts = [preamble, '1', content1, '2', content2, ...]
    i = 1
    while i + 1 < len(parts) and len(entries) < NUM_ENTRIES:
        entry_num = int(parts[i])
        content = parts[i + 1]

        # Extract sections
        gen_match = re.search(
            r"### Generated Explanation\n(.*?)(?=\n### Gold Rationale|\Z)",
            content, re.DOTALL
        )
        gold_match = re.search(
            r"### Gold Rationale\n(.*?)(?=\n### Retrieved Chunks|\Z)",
            content, re.DOTALL
        )
        chunks_match = re.search(
            r"### Retrieved Chunks\n(.*?)(?=\n## Entry|\Z)",
            content, re.DOTALL
        )

        entries.append({
            "number": entry_num,
            "description": ENTRY_DESCRIPTIONS[entry_num - 1],
            "generated": gen_match.group(1).strip() if gen_match else "(not found)",
            "gold": gold_match.group(1).strip() if gold_match else "(not found)",
            "chunks": chunks_match.group(1).strip() if chunks_match else "(not found)",
        })
        i += 2

    return entries


def display_entry(entry: dict):
    """Display one entry clearly in the terminal."""
    divider("─")
    print(f"\n📋  {entry['description']}\n")

    print("▶ RETRIEVED GUIDELINE CHUNKS (what the system had access to):")
    divider("·", 65)
    # Show first 600 chars of chunks to keep it readable
    chunk_preview = entry["chunks"][:600]
    if len(entry["chunks"]) > 600:
        chunk_preview += "\n  ... [truncated for display] ..."
    print(textwrap.fill(chunk_preview, width=80, initial_indent="  ",
                        subsequent_indent="  "))

    print(f"\n▶ AI-GENERATED EXPLANATION (what you are scoring):")
    divider("·", 65)
    gen_preview = entry["generated"][:800]
    if len(entry["generated"]) > 800:
        gen_preview += "\n  ... [truncated for display] ..."
    print(textwrap.fill(gen_preview, width=80, initial_indent="  ",
                        subsequent_indent="  "))

    print(f"\n▶ GOLD RATIONALE (expert answer):")
    divider("·", 65)
    print(textwrap.fill(entry["gold"][:400], width=80, initial_indent="  ",
                        subsequent_indent="  "))
    print()


def get_score(dimension: str, entry_num: int) -> int:
    """Prompt user for a score with validation."""
    rubric = {
        "Groundedness": (
            "  2 = Every factual claim explicitly supported by retrieved excerpts\n"
            "  1 = Mostly grounded; at most one minor unsupported assertion\n"
            "  0 = One or more claims not in excerpts / contradicted by them"
        ),
        "Citation Accuracy": (
            "  2 = All citations point to a real excerpt that supports the claim\n"
            "  1 = Citations present but one or more are wrong/imprecise\n"
            "  0 = Citations absent, fabricated, or systematically incorrect"
        ),
        "Consistency": (
            "  2 = Explanation fully agrees with the gold rationale\n"
            "  1 = Partial agreement; captures main point but misses some reasoning\n"
            "  0 = Contradicts or directly conflicts with the gold rationale"
        ),
    }
    print(f"\n  📏  {dimension} rubric:")
    print(rubric[dimension])
    while True:
        try:
            val = int(input(f"  Your {dimension} score for Entry {entry_num} [0/1/2]: ").strip())
            if val in (0, 1, 2):
                return val
            print("  ⚠️  Please enter 0, 1, or 2 only.")
        except ValueError:
            print("  ⚠️  Please enter a number: 0, 1, or 2.")


def load_scores(person_id: int) -> dict | None:
    """Load saved scores for a person, or None if not found."""
    path = SCORES_DIR / f"person_{person_id}.json"
    if path.exists():
        return json.loads(path.read_text())
    return None


def save_scores(person_id: int, name: str, scores: dict):
    """Save scores to JSON."""
    SCORES_DIR.mkdir(parents=True, exist_ok=True)
    path = SCORES_DIR / f"person_{person_id}.json"
    data = {"person_id": person_id, "name": name, "scores": scores}
    path.write_text(json.dumps(data, indent=2))
    print(f"\n  ✅  Scores saved to {path}")


# ─────────────────────────────────────────────────────────────
# SCORING MODE
# ─────────────────────────────────────────────────────────────

def run_scoring():
    header("AfriMed Tutor — Scoring Session")

    # Check sample file exists
    if not SAMPLE_FILE.exists():
        print(f"\n❌  Cannot find {SAMPLE_FILE}")
        print("    Make sure you are running from the AfriMed_Tutor-main directory.")
        return

    # Identify which person is scoring
    print("\n  Which person are you? (1, 2, 3, or 4)")
    while True:
        try:
            person_id = int(input("  Person number: ").strip())
            if person_id in range(1, TEAM_SIZE + 1):
                break
            print("  ⚠️  Enter 1, 2, 3, or 4.")
        except ValueError:
            print("  ⚠️  Enter a number.")

    # Check if already scored
    existing = load_scores(person_id)
    if existing:
        print(f"\n  ⚠️  Person {person_id} ({existing['name']}) has already submitted scores.")
        redo = input("  Re-score? This will overwrite the saved scores. [y/N]: ").strip().lower()
        if redo != "y":
            print("  Exiting without changes.")
            return

    name = input(f"\n  Enter your name, Person {person_id}: ").strip()

    print(f"\n  Hi {name}! You will score {NUM_ENTRIES} entries on 3 dimensions each.")
    print("  ⚠️  Score INDEPENDENTLY — do not look at teammates' scores.")
    print("  📖  Read the full entry before scoring. Take your time.\n")
    input("  Press Enter to begin...\n")

    # Load entries
    print("  Loading entries from qualitative_sample.md ...")
    entries = parse_entries(SAMPLE_FILE)
    print(f"  Loaded {len(entries)} entries.\n")

    # Scoring loop
    scores = {
        "groundedness": [],
        "citation_accuracy": [],
        "consistency": [],
    }

    for i, entry in enumerate(entries):
        header(f"Entry {entry['number']} of {NUM_ENTRIES}  —  {entry['description']}")
        display_entry(entry)

        g = get_score("Groundedness", entry["number"])
        c = get_score("Citation Accuracy", entry["number"])
        k = get_score("Consistency", entry["number"])

        scores["groundedness"].append(g)
        scores["citation_accuracy"].append(c)
        scores["consistency"].append(k)

        print(f"\n  ✔  Entry {entry['number']} scored: "
              f"Groundedness={g}  Citation={c}  Consistency={k}")

        if i < NUM_ENTRIES - 1:
            input("\n  Press Enter for next entry...")

    # Summary
    divider("═")
    print(f"\n  ✅  Scoring complete for {name} (Person {person_id})\n")
    print(f"  {'Entry':<10} {'Groundedness':>14} {'Citation':>10} {'Consistency':>13}")
    divider("·", 55)
    for i, desc in enumerate(ENTRY_DESCRIPTIONS):
        print(f"  {str(i+1):<10} {scores['groundedness'][i]:>14} "
              f"{scores['citation_accuracy'][i]:>10} {scores['consistency'][i]:>13}")

    save_scores(person_id, name, scores)

    # Show how many people have scored
    completed = sum(1 for p in range(1, TEAM_SIZE + 1) if load_scores(p))
    remaining = TEAM_SIZE - completed
    print(f"\n  📊  {completed}/{TEAM_SIZE} team members have scored.")
    if remaining > 0:
        print(f"  ⏳  {remaining} more member(s) still need to score.")
        print(f"  ➡️  Run:  python eval/kappa_scoring.py --mode score")
    else:
        print("  🎉  All members have scored! Generate the report:")
        print("  ➡️  Run:  python eval/kappa_scoring.py --mode report")


# ─────────────────────────────────────────────────────────────
# REPORT MODE
# ─────────────────────────────────────────────────────────────

def interpret(k):
    if k == "N/A":
        return "N/A"
    if k < 0.40:
        return "Poor"
    if k < 0.60:
        return "Moderate"
    if k < 0.80:
        return "Substantial"
    return "Near-perfect"


def safe_kappa(a, b):
    try:
        return round(cohen_kappa_score(a, b, labels=[0, 1, 2], weights="linear"), 3)
    except Exception:
        return "N/A"


def generate_report():
    header("AfriMed Tutor — Kappa Report Generator")

    # Load all scores
    all_scores = {}
    for p in range(1, TEAM_SIZE + 1):
        data = load_scores(p)
        if data:
            all_scores[p] = data
        else:
            print(f"  ⚠️  Person {p} has not scored yet. Run scoring mode first.")

    if len(all_scores) < TEAM_SIZE:
        missing = [p for p in range(1, TEAM_SIZE+1) if p not in all_scores]
        print(f"\n  Missing scores from Person(s): {missing}")
        print("  Run:  python eval/kappa_scoring.py --mode score")
        return

    dims = ["groundedness", "citation_accuracy", "consistency"]
    dim_labels = ["Groundedness", "Citation Accuracy", "Consistency"]

    # Build score arrays
    human = {p: all_scores[p]["scores"] for p in range(1, TEAM_SIZE + 1)}
    names = {p: all_scores[p]["name"] for p in range(1, TEAM_SIZE + 1)}

    # Human-human pairs
    pairs = [(1,2),(1,3),(1,4),(2,3),(2,4),(3,4)]
    hh_results = []
    for p1, p2 in pairs:
        row = {
            "label": f"Person {p1} ({names[p1]}) vs Person {p2} ({names[p2]})",
            "type": "human-human",
        }
        for d in dims:
            row[d] = safe_kappa(human[p1][d], human[p2][d])
        hh_results.append(row)

    # Mean human vs judge
    mean_scores = {}
    for d in dims:
        mean_scores[d] = [
            round(np.mean([human[p][d][i] for p in range(1, TEAM_SIZE+1)]))
            for i in range(NUM_ENTRIES)
        ]

    hj_row = {"label": "Mean Human vs LLM Judge", "type": "human-judge"}
    for d in dims:
        hj_row[d] = safe_kappa(mean_scores[d], LLM_JUDGE[d])

    # Individual vs judge
    indiv_judge = []
    for p in range(1, TEAM_SIZE + 1):
        row = {
            "label": f"Person {p} ({names[p]}) vs LLM Judge",
            "type": "human-judge",
        }
        for d in dims:
            row[d] = safe_kappa(human[p][d], LLM_JUDGE[d])
        indiv_judge.append(row)

    # ── Print to console ──────────────────────────────────────
    print("\n")
    header("HUMAN–HUMAN KAPPA")
    print(f"\n  {'Pair':<45} {'Ground':>8} {'Citation':>10} {'Consist':>10}  {'Interp'}")
    divider("·", 80)
    for r in hh_results:
        interp = interpret(r["groundedness"])
        print(f"  {r['label']:<45} {str(r['groundedness']):>8} "
              f"{str(r['citation_accuracy']):>10} {str(r['consistency']):>10}  {interp}")

    print("\n")
    header("HUMAN vs LLM JUDGE KAPPA")
    print(f"\n  {'Pair':<45} {'Ground':>8} {'Citation':>10} {'Consist':>10}  {'Interp'}")
    divider("·", 80)
    r = hj_row
    print(f"  {r['label']:<45} {str(r['groundedness']):>8} "
          f"{str(r['citation_accuracy']):>10} {str(r['consistency']):>10}  "
          f"{interpret(r['groundedness'])}")
    for r in indiv_judge:
        print(f"  {r['label']:<45} {str(r['groundedness']):>8} "
              f"{str(r['citation_accuracy']):>10} {str(r['consistency']):>10}  "
              f"{interpret(r['groundedness'])}")

    print("\n")
    header("INTERPRETATION GUIDE")
    print("  κ < 0.40          →  Poor")
    print("  κ 0.40 – 0.60     →  Moderate")
    print("  κ 0.60 – 0.80     →  Substantial")
    print("  κ > 0.80          →  Near-perfect")

    # ── Write markdown report ────────────────────────────────
    report_lines = [
        "# Cohen's Kappa Report",
        "## AfriMed Tutor | Ashesi University\n",
        "### Human–Human Agreement\n",
        f"| Pair | Groundedness κ | Citation κ | Consistency κ | Interpretation |",
        f"|------|---------------|------------|---------------|----------------|",
    ]
    for r in hh_results:
        report_lines.append(
            f"| {r['label']} | {r['groundedness']} | "
            f"{r['citation_accuracy']} | {r['consistency']} | "
            f"{interpret(r['groundedness'])} |"
        )

    report_lines += [
        "\n### Human vs LLM Judge Agreement\n",
        f"| Pair | Groundedness κ | Citation κ | Consistency κ | Interpretation |",
        f"|------|---------------|------------|---------------|----------------|",
    ]
    r = hj_row
    report_lines.append(
        f"| {r['label']} | {r['groundedness']} | "
        f"{r['citation_accuracy']} | {r['consistency']} | "
        f"{interpret(r['groundedness'])} |"
    )
    for r in indiv_judge:
        report_lines.append(
            f"| {r['label']} | {r['groundedness']} | "
            f"{r['citation_accuracy']} | {r['consistency']} | "
            f"{interpret(r['groundedness'])} |"
        )

    report_lines += [
        "\n### Interpretation Guide\n",
        "| κ Range | Interpretation |",
        "|---------|----------------|",
        "| < 0.40 | Poor |",
        "| 0.40–0.60 | Moderate |",
        "| 0.60–0.80 | Substantial |",
        "| > 0.80 | Near-perfect |",
        "\n### Summary\n",
        f"All four team members ({', '.join(names[p] for p in range(1,5))}) "
        f"independently scored {NUM_ENTRIES} entries from `results/qualitative_sample.md`. "
        "Cohen's Kappa (linear weighted) was computed for all human–human pairs and "
        "for the mean human scores against the automated LLM judge scores.",
    ]

    REPORT_FILE.parent.mkdir(parents=True, exist_ok=True)
    REPORT_FILE.write_text("\n".join(report_lines))
    print(f"\n\n  📄  Report saved to {REPORT_FILE}")


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Cohen's Kappa scoring and report generation"
    )
    parser.add_argument(
        "--mode",
        choices=["score", "report"],
        default="score",
        help="'score' to record a team member's scores; 'report' to generate kappa table",
    )
    args = parser.parse_args()

    if args.mode == "score":
        run_scoring()
    else:
        generate_report()