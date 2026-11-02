#!/usr/bin/env python3
"""Verify packaged file integrity and the data-table schema; no model fitting."""
from pathlib import Path
import csv, hashlib, sys
import pandas as pd

ROOT=Path(__file__).resolve().parent.parent
TABLE=ROOT/'data/05_model_ready/aging/expanded_aging_features_with_batterylife.csv'
CHECKSUMS=ROOT/'SHA256SUMS.txt'
EXPECTED={'MATR':180,'HUST':77,'CALCE':13,'RWTH':48,'CALB':27,'HNEI':14,
          'MICH':40,'MICH_EXP':18,'NA-ion':64,'SNL':61,'UL_PUR':2,'XJTU':23}

def main():
    errors=[]
    if not TABLE.exists():
        print('MISSING', TABLE);return 1
    df=pd.read_csv(TABLE)
    for col in ('source_family','dataset_id','cell_id','final_capacity_retention'):
        if col not in df.columns:errors.append(f'Missing required column: {col}')
    if not errors:
        counts=df.dataset_id.value_counts().to_dict()
        if counts!=EXPECTED:errors.append(f'Unexpected per-dataset counts: {counts}')
        if len(df)!=567:errors.append(f'Expected 567 rows, got {len(df)}')
        if len(df.columns)!=38:errors.append(f'Expected 38 columns, got {len(df.columns)}')
        if df.final_capacity_retention.isna().any():errors.append('Missing target values')
        if df.duplicated(['dataset_id','cell_id']).any():errors.append('Duplicate dataset/cell IDs')
        if df.source_family.nunique()!=2:errors.append('Source family count is not two')
    checks=0
    if CHECKSUMS.exists():
        for line in CHECKSUMS.read_text().splitlines():
            if not line.strip():continue
            target,rel=line.split('  ',1)
            p=ROOT/rel
            if not p.is_file():errors.append('Missing file in manifest: '+rel);continue
            digest=hashlib.sha256(p.read_bytes()).hexdigest()
            if digest!=target:errors.append('Checksum mismatch: '+rel)
            checks+=1
    print(f'Rows: {len(df)}; columns: {len(df.columns)}; datasets: {df.dataset_id.nunique()}')
    print(f'Files verified: {checks}')
    if errors:
        print('FAIL:')
        for e in errors: print(' - '+e)
        return 1
    print('PASS: table structure, identifiers, targets and file checksums are consistent.')
    print('Note: This is not a complete model reproduction or independent scientific validation.')
    return 0
if __name__=='__main__':sys.exit(main())
