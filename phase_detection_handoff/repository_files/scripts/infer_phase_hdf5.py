"""Offline V-JEPA2 + trained phase TCN inference. No feedback files are edited."""
import argparse
import csv
import hashlib
import json
from contextlib import ExitStack
from pathlib import Path


def window_plan(times, window, stride, max_gap):
    """Return windows that never span a missing-data gap; times are seconds."""
    import bisect
    cuts = [0] + [i for i in range(1, len(times))
                  if times[i] - times[i - 1] > max_gap] + [len(times)]
    windows = []
    for segment, (lo, hi) in enumerate(zip(cuts, cuts[1:])):
        start = times[lo]
        k = 0
        while start + k * stride + window <= times[hi - 1] + 1e-7:
            left = start + k * stride
            right = left + window
            a = bisect.bisect_left(times, left, lo, hi)
            b = bisect.bisect_right(times, right, lo, hi)
            if b - a >= 2:
                windows.append(dict(segment_id=segment, start_s=left,
                                    end_s=right, lo=a, hi=b))
            k += 1
    return windows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--session', required=True, type=Path)
    p.add_argument('--checkpoint', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--model', default='facebook/vjepa2-vitl-fpc64-256')
    p.add_argument('--color-order', choices=['rgb', 'bgr'], required=True)
    p.add_argument('--window', type=float, default=5)
    p.add_argument('--stride', type=float, default=1)
    p.add_argument('--max-gap', type=float, default=0.5)
    p.add_argument('--max-windows', type=int, default=0,
                   help='0 processes all windows; positive values are a smoke test')
    p.add_argument('--plan-only', action='store_true')
    p.add_argument('--allow-download', action='store_true')
    args = p.parse_args()
    if min(args.window, args.stride, args.max_gap) <= 0 or args.max_windows < 0:
        p.error('Durations must be positive and max-windows nonnegative')
    import h5py
    import numpy as np
    key = 'vision_data/continuous_data/'
    chunks, frames, skipped = [], [], []
    for path in sorted(args.session.glob('*.hdf5')):
        with h5py.File(path, 'r') as f:
            if key + 'time' not in f or f[key + 'time'].size == 0:
                skipped.append(path.name)
                continue
            t = f[key + 'time'][:].reshape(-1)
            if not np.isfinite(t).all() or not (np.diff(t) > 0).all():
                raise ValueError(f'Invalid or unordered timestamps: {path}')
            ds = f[key + 'l_img']
            if ds.shape[0] != len(t) or len(ds.shape) != 4 or ds.shape[-1] != 3 or ds.dtype != np.uint8:
                raise ValueError(f'Unexpected image layout: {path}: {ds.shape}, {ds.dtype}')
            chunk_id = len(chunks)
            chunks.append(path)
            frames.extend((float(ts), chunk_id, i) for i, ts in enumerate(t))
    if not frames:
        raise ValueError('No timestamped images found')
    frames.sort()
    absolute = np.array([r[0] for r in frames])
    if not (np.diff(absolute) > 0).all():
        raise ValueError('Overlapping or duplicate chunk timestamps require review')
    origin = float(absolute[0])
    times = (absolute - origin).tolist()
    windows = window_plan(times, args.window, args.stride, args.max_gap)
    total = len(windows)
    if args.max_windows:
        windows = windows[:args.max_windows]
    gaps = [{'start_s': times[i-1], 'end_s': times[i]}
            for i in range(1, len(times)) if times[i]-times[i-1] > args.max_gap]
    summary = dict(schema='phase_hdf5_inference_v1', session=str(args.session.resolve()),
                   time_origin='first_vision_frame', time_origin_epoch_s=origin,
                   frame_count=len(frames), duration_s=times[-1],
                   chunks=[str(x.resolve()) for x in chunks], skipped_files=skipped,
                   window_s=args.window, stride_s=args.stride, max_gap_s=args.max_gap,
                   gaps=gaps, available_windows=total, selected_windows=len(windows),
                   partial_run=len(windows) < total, color_order=args.color_order,
                   model=args.model, temporal_processing='TCN separately per continuous segment',
                   scope='offline window predictions; not frame labels or accuracy measurements')
    print(json.dumps(summary, indent=2), flush=True)
    if args.plan_only:
        return
    if not windows:
        raise ValueError('No usable five-second windows')
    if args.output.exists():
        raise FileExistsError(f'Choose a new output directory: {args.output}')
    import torch
    from torch import nn
    from transformers import VJEPA2Model, VJEPA2VideoProcessor

    class TemporalBlock(nn.Module):
        def __init__(self, dilation):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv1d(128, 128, 3, padding=dilation, dilation=dilation),
                nn.ReLU(), nn.Dropout(0.3),
                nn.Conv1d(128, 128, 3, padding=dilation, dilation=dilation),
                nn.ReLU(), nn.Dropout(0.3))
        def forward(self, x):
            return x + self.net(x)

    class PhaseTCN(nn.Module):
        def __init__(self):
            super().__init__()
            self.input_proj = nn.Conv1d(1024, 128, 1)
            self.blocks = nn.Sequential(*(TemporalBlock(d) for d in (1, 2, 4, 8)))
            self.classifier = nn.Conv1d(128, 3, 1)
        def forward(self, x):
            return self.classifier(self.blocks(self.input_proj(x.transpose(1, 2)))).transpose(1, 2)

    ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    tcn = PhaseTCN().eval()
    tcn.load_state_dict(ckpt['model_state_dict'], strict=True)
    mean, std = ckpt['mean'].float(), ckpt['std'].float()
    if mean.shape != (1, 1024) or std.shape != (1, 1024):
        raise ValueError('Unexpected normalization shape')
    if not torch.isfinite(mean).all() or not torch.isfinite(std).all() or not (std > 0).all():
        raise ValueError('Invalid checkpoint normalization')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Loading encoder on {device}; downloads allowed: {args.allow_download}', flush=True)
    processor = VJEPA2VideoProcessor.from_pretrained(args.model, local_files_only=not args.allow_download)
    encoder = VJEPA2Model.from_pretrained(args.model, local_files_only=not args.allow_download).to(device).eval()
    embeddings = []
    with ExitStack() as stack, torch.inference_mode():
        handles = [stack.enter_context(h5py.File(path, 'r')) for path in chunks]
        for n, w in enumerate(windows, 1):
            # Sample evenly in actual time; never export or load an entire image volume.
            targets = np.linspace(w['start_s'], w['end_s'], 16)
            candidates = np.asarray(times[w['lo']:w['hi']])
            indices = [w['lo'] + int(np.abs(candidates - target).argmin()) for target in targets]
            imgs = []
            for index in indices:
                _, chunk, row = frames[index]
                img = handles[chunk][key + 'l_img'][row]
                if args.color_order == 'bgr':
                    img = img[..., ::-1].copy()
                imgs.append(img)
            pixels = processor(videos=[imgs], return_tensors='pt')['pixel_values_videos'].to(device)
            tokens = encoder(pixel_values_videos=pixels, skip_predictor=True).last_hidden_state
            emb = tokens.mean(dim=1).float().cpu()
            if emb.shape != (1, 1024) or not torch.isfinite(emb).all():
                raise ValueError(f'Invalid encoder output: {emb.shape}')
            embeddings.append(emb)
            if n == 1 or n % 10 == 0 or n == len(windows):
                print(f'Encoded {n}/{len(windows)} windows', flush=True)
    x = torch.cat(embeddings)
    normalized = (x - mean) / std
    probabilities = torch.empty((len(windows), 3))
    with torch.inference_mode():
        for segment in sorted({w['segment_id'] for w in windows}):
            indices = [i for i, w in enumerate(windows) if w['segment_id'] == segment]
            probabilities[indices] = tcn(normalized[indices].unsqueeze(0)).softmax(-1)[0]
    if not torch.isfinite(probabilities).all():
        raise ValueError('Nonfinite predictions')
    labels = ['pre_antrum', 'antrum', 'incus']
    rows = []
    for w, probs in zip(windows, probabilities.tolist()):
        pred = int(np.argmax(probs))
        rows.append(dict(segment_id=w['segment_id'], start_s=w['start_s'], end_s=w['end_s'],
                         start_epoch_s=origin+w['start_s'], end_epoch_s=origin+w['end_s'],
                         pred_id=pred, pred_phase=labels[pred],
                         **dict(zip(['prob_'+s for s in labels], probs))))
    summary.update(checkpoint=str(args.checkpoint.resolve()),
                   checkpoint_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
                   best_epoch=ckpt.get('best_epoch'), device=device,
                   labels=labels, decoder='raw TCN; no stay/advance postprocessing')
    args.output.mkdir(parents=True, exist_ok=False)
    with (args.output / 'phase_predictions.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    torch.save({'x': x, 'windows': windows, 'summary': summary}, args.output / 'embeddings.pt')
    (args.output / 'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(f'Saved {len(rows)} predictions to {args.output}', flush=True)


if __name__ == '__main__':
    main()
