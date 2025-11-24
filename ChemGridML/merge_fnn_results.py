#!/usr/bin/env python
"""
Merge FNN results from OpenADMET_FNN_ECFP database into OpenADMET_Phase4_FNN database
"""
import sqlite3
from pathlib import Path

# Paths
results_dir = Path("openadmet_results")
source_db = results_dir / "predictions_OpenADMET_FNN_ECFP.db"
target_db = results_dir / "predictions_OpenADMET_Phase4_FNN.db"

print(f"Merging {source_db} into {target_db}")

# Connect to both databases
source_conn = sqlite3.connect(str(source_db))
target_conn = sqlite3.connect(str(target_db))

# Copy predictions
print("Copying predictions...")
source_cursor = source_conn.cursor()
target_cursor = target_conn.cursor()

# Get all predictions from source
predictions = source_cursor.execute("SELECT * FROM predictions").fetchall()
print(f"Found {len(predictions)} predictions in source database")

# Insert into target (skip duplicates)
inserted = 0
for pred in predictions:
    try:
        target_cursor.execute("""
            INSERT OR IGNORE INTO predictions 
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, pred)
        if target_cursor.rowcount > 0:
            inserted += 1
    except Exception as e:
        print(f"Error inserting prediction: {e}")

print(f"Inserted {inserted} new predictions")

# Copy dataset_targets
print("Copying dataset_targets...")
dataset_targets = source_cursor.execute("SELECT * FROM dataset_targets").fetchall()
print(f"Found {len(dataset_targets)} dataset_target entries")

inserted = 0
for dt in dataset_targets:
    try:
        target_cursor.execute("""
            INSERT OR IGNORE INTO dataset_targets 
            VALUES (?, ?, ?)
        """, dt)
        if target_cursor.rowcount > 0:
            inserted += 1
    except Exception as e:
        print(f"Error inserting dataset_target: {e}")

print(f"Inserted {inserted} new dataset_target entries")

# Commit and close
target_conn.commit()
source_conn.close()
target_conn.close()

print("✓ Merge complete!")

# Verify
verify_conn = sqlite3.connect(str(target_db))
cursor = verify_conn.cursor()
result = cursor.execute("""
    SELECT fingerprint, model_name, COUNT(DISTINCT seed) as n_seeds
    FROM predictions
    GROUP BY fingerprint, model_name
""").fetchall()

print("\nCurrent status in Phase4_FNN database:")
for row in result:
    print(f"  {row[0]} + {row[1]}: {row[2]} seeds")

verify_conn.close()
