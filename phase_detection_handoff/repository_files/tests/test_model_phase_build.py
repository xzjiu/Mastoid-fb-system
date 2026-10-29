import csv
import json
import os
import sqlite3
import sys
from pathlib import Path
from dataclasses import replace
from unittest.mock import patch

import h5py
import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'src')]
from scripts.create_demo_data import create_demo_data
from evidence_system.registry import get_run,load_registry
from evidence_system.source_config import SourceRoots,resolve_run_sources
from evidence_system.pipeline import build_evidence
from evidence_system.model_phases import assign_model_phases


def fixture(tmp_path):
    d=create_demo_data(tmp_path/'demo')
    record=get_run('DEMO_CASE_01',load_registry(ROOT/'config/examples/runs.example.csv'))
    sources=resolve_run_sources(record,SourceRoots(Path(d['output_root'])/'raw',Path(d['output_root'])/'edt',Path(d['phase_annotation_file'])))
    with h5py.File(d['hdf5_file'],'r') as f:t=f['data/time'][:].reshape(-1)
    rows=[]
    for i,ts in enumerate(t):
        phase=['pre_antrum','incus','unknown','incus'][min(i//6,3)]
        prob=[1 if phase==name else 0 for name in ('pre_antrum','antrum','incus')]
        rows.append(dict(frame_idx_global=i,source_hdf5=Path(d['hdf5_file']).name,source_frame_idx=i,
                         t_abs=float(ts),t_video_s=float(ts-t[0]),phase_source='model_vjepa2_tcn',
                         model_phase=phase,model_score=1,contributing_windows=1,
                         **dict(zip(['prob_pre_antrum','prob_antrum','prob_incus'],prob))))
    source=tmp_path/'model.csv'
    with source.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    manifest=dict(schema='model_phase_frame_input_v1',frame_count=len(rows),
                  source_summary=dict(partial_run=False,time_origin='first_vision_frame',
                                      time_origin_epoch_s=float(t[0]),max_gap_s=.5))
    (tmp_path/'phase_input_manifest.json').write_text(json.dumps(manifest))
    return d,record,replace(sources,model_phase_frames=source),rows


def test_model_build_and_provenance(tmp_path):
    d,record,sources,rows=fixture(tmp_path)
    # No human annotation file is required in explicit model mode.
    Path(d['phase_annotation_file']).unlink()
    with patch.dict(os.environ,{'MASTOID_ANATOMY_PROFILES':str(ROOT/'config/examples/anatomy_profiles.example.json')}):
        result=build_evidence(record=record,sources=sources,result_dir=tmp_path/'out',project_root=ROOT)
    assert result['store_integrity']=='ok'
    with sqlite3.connect(result['evidence_database']) as db:
        sources_in_db=db.execute('select * from source_files').fetchall()
        assert any('model_phase_prediction' in row for row in sources_in_db)
        assert not any('human_phase_annotation' in row for row in sources_in_db)
        ids=db.execute('select phase_id from frames order by frame_idx').fetchall()
        assert ids[6] != ids[18] # repeated incus gets distinct interval identity
    with sqlite3.connect(result['evidence_database']) as db:
        links=db.execute("select * from relations where relation_type='derived_from'").fetchall()
        assert any('source:model_phase_prediction' in row for row in links)


def test_wrong_recording_rejected(tmp_path):
    d,record,sources,rows=fixture(tmp_path)
    frames=[dict(source_hdf5=r['source_hdf5'],source_frame_idx=r['source_frame_idx'],t_abs=r['t_abs']) for r in rows]
    frames[1]['t_abs']+=.1
    with pytest.raises(ValueError,match='mismatch'):assign_model_phases(sources.model_phase_frames,frames,record.global_run_id)

