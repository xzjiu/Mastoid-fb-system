"""Explicit offline model phase input, separate from human annotations."""
from __future__ import annotations
import csv
import hashlib
import json
import math
from pathlib import Path

LABELS = ('pre_antrum', 'antrum', 'incus')


def read_model_phases(path: Path):
    with path.open(newline='') as f:
        rows = list(csv.DictReader(f))
    manifest_path = path.parent / 'phase_input_manifest.json'
    manifest = json.loads(manifest_path.read_text())
    if manifest.get('schema') != 'model_phase_frame_input_v1':
        raise ValueError('Unsupported model-phase manifest')
    summary = manifest['source_summary']
    if summary.get('partial_run') is not False or summary.get('time_origin') != 'first_vision_frame':
        raise ValueError('Incomplete inference or unsupported clock')
    if len(rows) != int(manifest['frame_count']) or not rows:
        raise ValueError('Model frame count mismatch')
    origin = float(summary['time_origin_epoch_s'])
    for i, row in enumerate(rows):
        if int(row['frame_idx_global']) != i or row['phase_source'] != 'model_vjepa2_tcn':
            raise ValueError('Unexpected model frame ordering or provenance')
        for key in ('t_abs', 't_video_s'):
            row[key] = float(row[key])
            if not math.isfinite(row[key]):
                raise ValueError('Nonfinite model timestamp')
        if abs(row['t_abs'] - origin - row['t_video_s']) > 1e-5:
            raise ValueError('Model clocks disagree')
        row['source_frame_idx'] = int(row['source_frame_idx'])
        phase = row['model_phase']
        if phase not in (*LABELS, 'unknown'):
            raise ValueError('Unsupported model phase')
        if phase != 'unknown':
            probs = [float(row['prob_'+label]) for label in LABELS]
            if any(not math.isfinite(p) or not 0 <= p <= 1 for p in probs) or abs(sum(probs)-1) > 1e-5:
                raise ValueError('Invalid model probabilities')
            if LABELS[max(range(3), key=lambda k: probs[k])] != phase:
                raise ValueError('Model label and probabilities disagree')
    if any(b['t_abs'] <= a['t_abs'] for a,b in zip(rows,rows[1:])):
        raise ValueError('Model timestamps must strictly increase')
    return rows, manifest


def inspect_model_phases(path: Path):
    rows, manifest = read_model_phases(path)
    return dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                valid=True, row=None, source_kind='model_phase_prediction',
                frame_count=len(rows), manifest=manifest,
                manifest_sha256=hashlib.sha256((path.parent/'phase_input_manifest.json').read_bytes()).hexdigest())


def assign_model_phases(path: Path, frame_parts, global_run_id):
    rows, manifest = read_model_phases(path)
    if len(rows) != len(frame_parts):
        raise ValueError('Model input does not cover this recording frame inventory')
    assignments = []
    for row, source in zip(rows, frame_parts):
        if (row['source_hdf5'] != source['source_hdf5'] or
            row['source_frame_idx'] != source['source_frame_idx'] or
            abs(row['t_abs']-source['t_abs']) > 1e-5):
            raise ValueError('Model source file, frame index or timestamp mismatch')
        assignments.append({k:row[k] for k in ('model_phase','model_score','phase_source',
                            'prob_pre_antrum','prob_antrum','prob_incus','contributing_windows')})
    intervals=[]
    max_gap=float(manifest['source_summary']['max_gap_s'])
    if not math.isfinite(max_gap) or max_gap <= 0:
        raise ValueError('Invalid gap threshold')
    start=0
    for stop in range(1,len(rows)+1):
        boundary = stop == len(rows) or rows[stop]['model_phase'] != rows[start]['model_phase'] or rows[stop]['t_abs']-rows[stop-1]['t_abs'] > max_gap
        if not boundary:
            continue
        phase_id=f'{global_run_id}:model_phase:{len(intervals):06d}'
        end=rows[stop-1]['t_video_s']
        if stop < len(rows) and rows[stop]['t_abs']-rows[stop-1]['t_abs'] <= max_gap:
            end=rows[stop]['t_video_s']
        else:
            end=math.nextafter(end, math.inf)
        intervals.append(dict(phase_id=phase_id,phase_order=len(intervals)+1,
                              phase=rows[start]['model_phase'],annotation_start_s=rows[start]['t_video_s'],
                              annotation_end_s=end,observed_start_s=rows[start]['t_video_s'],
                              observed_end_s=rows[stop-1]['t_video_s'],observed_frame_count=stop-start,
                              boundary_source='model_phase_prediction',phase_annotation_id=None))
        for j in range(start,stop): assignments[j]['model_phase_interval_id']=phase_id
        start=stop
    return assignments, intervals
