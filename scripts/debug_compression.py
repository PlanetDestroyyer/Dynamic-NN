import sys
import os
import json

from comprehensive_benchmark import run_experiment

print("Running Full Dynamic + Compress (Seed 42)...")
m = run_experiment(42, "Full Dynamic + Compress", True, True, True, True)

print("\n--- RESULTS ---")
print(f"Accuracy: {m['final_accuracy']*100:.1f}%")
print(f"Forgetting: {m['backward_transfer']*100:.1f}%")
print(f"Net Growth: {m['net_growth']}")
print(f"Peak Growth: {m['peak_growth']}")
print(f"Total Created: {m['total_created']}")
print(f"Total Pruned: {m['total_pruned']}")
