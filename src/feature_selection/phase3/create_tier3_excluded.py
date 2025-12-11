#!/usr/bin/env python
"""Create Tier3 excluded features JSON by merging Tier2 and Phase 3 deletions.

This script combines the Tier2 exclusion list with Phase 3 cluster-based
deletions to create the final Tier3 excluded features configuration.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

THIS_DIR = Path(__file__).resolve().parent
SRC_ROOT = THIS_DIR.parents[1]
PROJECT_ROOT = THIS_DIR.parents[2]
for path in (SRC_ROOT, PROJECT_ROOT):
    if str(path) not in sys.path:
        sys.path.append(str(path))


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    ap = argparse.ArgumentParser(
        description="Create Tier3 excluded features JSON."
    )
    ap.add_argument(
        "--tier2-excluded",
        type=str,
        required=True,
        help="Path to Tier2 excluded.json",
    )
    ap.add_argument(
        "--representatives-json",
        type=str,
        required=True,
        help="Path to cluster_representatives.json",
    )
    ap.add_argument(
        "--out-file",
        type=str,
        default="configs/feature_selection/tier3/excluded.json",
        help="Output path for Tier3 excluded.json",
    )
    return ap.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Main entry point."""
    args = parse_args(argv)
    
    print("=" * 80)
    print("Creating Tier3 Excluded Features JSON")
    print("=" * 80)
    print(f"Tier2 excluded: {args.tier2_excluded}")
    print(f"Representatives JSON: {args.representatives_json}")
    print(f"Output file: {args.out_file}")
    print()
    
    # Load Tier2 excluded features
    print("Loading Tier2 excluded features...")
    with open(args.tier2_excluded, "r") as f:
        tier2_data = json.load(f)
    
    tier2_candidates = tier2_data.get("candidates", [])
    print(f"Tier2 exclusions: {len(tier2_candidates)} features")
    print()
    
    # Load Phase 3 representative results
    print("Loading Phase 3 deletion candidates...")
    with open(args.representatives_json, "r") as f:
        phase3_data = json.load(f)
    
    phase3_removals = phase3_data.get("to_remove", [])
    print(f"Phase 3 deletions: {len(phase3_removals)} features")
    print()
    
    # Create new candidates list
    all_candidates = tier2_candidates.copy()
    
    # Add Phase 3 deletions
    for removal in phase3_removals:
        all_candidates.append({
            "feature_name": removal["feature"],
            "reason": "phase3_correlation_cluster",
            "cluster_id": removal["cluster_id"],
            "mean_gain": removal["mean_gain"],
        })
    
    # Create Tier3 output
    tier3_output = {
        "version": "tier3-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "base_tier": "tier2",
        "phase3_additions": len(phase3_removals),
        "candidates": all_candidates,
        "summary": {
            "tier2_exclusions": len(tier2_candidates),
            "phase3_correlation_cluster": len(phase3_removals),
            "total_exclusions": len(all_candidates),
        }
    }
    
    # Create output directory
    out_path = Path(args.out_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save Tier3 excluded.json
    print(f"Saving Tier3 excluded features to {args.out_file}...")
    with open(args.out_file, "w") as f:
        json.dump(tier3_output, f, indent=2)
    
    print()
    print("Summary:")
    print(f"  Tier2 exclusions: {tier3_output['summary']['tier2_exclusions']}")
    print(f"  Phase 3 additions: {tier3_output['phase3_additions']}")
    print(f"  Total exclusions: {tier3_output['summary']['total_exclusions']}")
    print()
    print("Tier3 excluded.json created successfully!")
    print("=" * 80)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
