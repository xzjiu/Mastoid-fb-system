"""Validate offline TCN output and attach predictions to actual HDF5 timestamps.

Creates separate model evidence; does not replace human annotations or build feedback.
"""
import argparse
import bisect
import csv
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

LABELS = ('pre_antrum', 'antrum', 'incus')


def finite(value):
    value = float(value)
    if not math.isfinite(value):
        raise ValueError('Nonfinite number')
    return value


class PhaseInput:
    def __init__(self, predictions, summary):
        self.summary = summary
        if summary.get('schema') != 'phase_hdf5_inference_v1':
            raise ValueError('Unsupported prediction schema')
        if summary.get('time_origin') != 'first_vision_frame':
            raise ValueError('Unsupported time origin')
        if summary.get('labels') != list(LABELS):
            raise ValueError('Unexpected model labels')
        if summary.get('partial_run') is not False:
            raise ValueError('Use full inference output, not a smoke test')
        self.origin = finite(summary['time_origin_epoch_s'])
        self.duration = finite(summary['duration_s'])
        self.gaps = [(finite(g['start_s']), finite(g['end_s'])) for g in summary['gaps']]
        if any(a < 0 or b <= a or b > self.duration + 1e-6 for a, b in self.gaps):
            raise ValueError('Invalid gap bounds')
        self.windows = []
        for row in predictions:
            a, b = finite(row['start_s']), finite(row['end_s'])
            if not 0 <= a < b <= self.duration + 1e-6:
                raise ValueError('Invalid prediction interval')
            if abs((b-a)-finite(summary['window_s'])) > 1e-5:
                raise ValueError('Window length differs from summary')
            for key, relative in [('start_epoch_s', a), ('end_epoch_s', b)]:
                if abs(finite(row[key]) - (self.origin + relative)) > 1e-5:
                    raise ValueError('Absolute and relative timestamps disagree')
            if any(a < y and b > x for x, y in self.gaps):
                raise ValueError('Prediction window crosses a recording gap')
            probs = [finite(row['prob_' + name]) for name in LABELS]
            if any(x < 0 or x > 1 for x in probs) or abs(sum(probs)-1) > 1e-5:
                raise ValueError('Invalid class probabilities')
            pred = max(range(3), key=lambda i: probs[i])
            if int(row['pred_id']) != pred or row['pred_phase'] != LABELS[pred]:
                raise ValueError('Label does not match class probabilities')
            self.windows.append(dict(start=a, end=b, segment=int(row['segment_id']), probs=probs))
        self.windows.sort(key=lambda w: w['start'])
        if not self.windows or len(self.windows) != int(summary['selected_windows']):
            raise ValueError('Prediction count differs from summary')
        if len({(w['start'], w['end']) for w in self.windows}) != len(self.windows):
            raise ValueError('Duplicate windows')
        self.starts = [w['start'] for w in self.windows]
        self.width = max(w['end']-w['start'] for w in self.windows)

    def at(self, absolute_time):
        t = finite(absolute_time) - self.origin
        lo = bisect.bisect_left(self.starts, t-self.width-1e-6)
        hi = bisect.bisect_right(self.starts, t)
        covering = [w for w in self.windows[lo:hi] if w['start'] <= t < w['end']]
        if not covering or any(a < t < b for a, b in self.gaps):
            return dict(model_phase='unknown', model_score=None, contributing_windows=0,
                        segment_id=None, **{'prob_'+name: None for name in LABELS})
        segments = {w['segment'] for w in covering}
        if len(segments) != 1:
            raise ValueError('Overlapping windows from different recording segments')
        probs = [sum(w['probs'][i] for w in covering)/len(covering) for i in range(3)]
        pred = max(range(3), key=lambda i: probs[i])
        return dict(model_phase=LABELS[pred], model_score=probs[pred],
                    contributing_windows=len(covering), segment_id=next(iter(segments)),
                    **dict(zip(('prob_'+name for name in LABELS), probs)))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--predictions', required=True, type=Path)
    parser.add_argument('--summary', required=True, type=Path)
    parser.add_argument('--session', type=Path, help='Defaults to session path in inference summary')
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    summary = json.loads(args.summary.read_text())
    with args.predictions.open(newline='') as f:
        model = PhaseInput(list(csv.DictReader(f)), summary)
    session = args.session or Path(summary['session'])
    if args.output.exists():
        raise FileExistsError(f'Choose a new output directory: {args.output}')
    import h5py
    expected = sorted(Path(p).name for p in summary['chunks'])
    frames, populated, audits = [], [], []
    for path in sorted(session.glob('*.hdf5')):
        with h5py.File(path, 'r') as handle:
            prefix = next((k for k in ('vision_data/continuous_data/', 'data/')
                           if k+'time' in handle and handle[k+'time'].size), None)
            if prefix is None:
                continue
            times = handle[prefix+'time'][:].reshape(-1).tolist()
            if any(not math.isfinite(t) for t in times) or any(b <= a for a,b in zip(times,times[1:])):
                raise ValueError(f'Invalid frame timestamps in {path.name}')
            if prefix+'l_img' not in handle or len(handle[prefix+'l_img']) != len(times):
                raise ValueError(f'Image/timestamp count mismatch in {path.name}')
            populated.append(path.name)
            frames.extend(dict(source_hdf5=path.name, source_frame_idx=i, t_abs=t,
                               image_dataset=prefix+'l_img') for i,t in enumerate(times))
            audits.append(dict(file=path.name, layout=prefix,
                               frame_aligned_camera_present=prefix+'pose_main_camera' in handle,
                               high_frequency_camera_present='physics_data/continuous_data/high_frequency_poses/main_camera_pose/pose' in handle,
                               legacy_voxel_volume_present='metadata/voxel_volume' in handle,
                               v3_voxel_volume_present='metadata_v3/voxel_volume' in handle))
    if sorted(populated) != expected:
        raise ValueError('Recording chunk names differ from inference summary')
    frames.sort(key=lambda row: row['t_abs'])
    if len(frames) != int(summary['frame_count']) or not frames:
        raise ValueError('Frame count differs from inference summary')
    if any(b['t_abs'] <= a['t_abs'] for a,b in zip(frames,frames[1:])):
        raise ValueError('Duplicate or overlapping frame timestamps')
    if abs(frames[0]['t_abs']-model.origin) > 1e-5 or abs(frames[-1]['t_abs']-model.origin-model.duration)>1e-5:
        raise ValueError('Recording time range differs from inference summary')
    for i, row in enumerate(frames):
        row.update(frame_idx_global=i, t_video_s=row['t_abs']-model.origin,
                   phase_source='model_vjepa2_tcn', **model.at(row['t_abs']))
    counts = dict(Counter(row['model_phase'] for row in frames))
    manifest = dict(schema='model_phase_frame_input_v1', session=str(session.resolve()),
                    source_summary=summary, predictions_sha256=digest(args.predictions),
                    summary_sha256=digest(args.summary), frame_count=len(frames),
                    frame_phase_counts=counts, hdf5_audit=audits,
                    aggregation='mean raw TCN probabilities across covering half-open windows',
                    score_note='model_score is uncalibrated; not an accuracy estimate',
                    gap_policy='unknown when no prediction window covers the timestamp',
                    label_policy='retain pre_antrum; do not equate it with early_surface or pre_drilling',
                    offline_only=True, integrated_into_feedback_build=False)
    args.output.mkdir(parents=True, exist_ok=False)
    with (args.output/'model_phase_frames.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(frames[0]))
        writer.writeheader(); writer.writerows(frames)
    (args.output/'phase_input_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(dict(frame_count=len(frames), frame_phase_counts=counts,
                          output=str(args.output.resolve()), feedback_build_connected=False), indent=2))


if __name__ == '__main__':
    main()
