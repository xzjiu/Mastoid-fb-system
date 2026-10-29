"""Local saved-session phase runner. Run with --help for machine-local paths."""
import argparse
import csv
import json
import math
import secrets
import subprocess
import threading
from collections import Counter
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse
from uuid import uuid4

HERE = Path(__file__).resolve().parent
LABELS = ('pre_antrum', 'antrum', 'incus', 'unknown')


def read_result(path):
    points = []
    counts = Counter()
    with path.open(newline='') as f:
        for row in csv.DictReader(f):
            t = float(row['t_video_s'])
            label = row['model_phase']
            if not math.isfinite(t) or t < 0 or label not in LABELS:
                raise ValueError('Invalid phase output')
            if points and t <= points[-1][0]:
                raise ValueError('Frame times must increase')
            points.append([t, label])
            counts[label] += 1
    if not points:
        raise ValueError('No frame predictions were generated')
    return {'points': points, 'counts': dict(counts), 'frame_count': len(points)}


class Runner:
    def __init__(self, python, checkpoint, output, session, color):
        self.python = Path(python).expanduser().resolve()
        self.checkpoint = Path(checkpoint).expanduser().resolve()
        self.output = Path(output).expanduser().resolve()
        if not self.python.is_file():
            raise ValueError(f'Python executable not found: {self.python}')
        if not self.checkpoint.is_file():
            raise ValueError(f'Checkpoint not found: {self.checkpoint}')
        self.session = str(Path(session).expanduser().resolve())
        self.color = color
        self.lock = threading.RLock()
        self.process = None
        self.thread = None
        self.stopping = False
        self.state = {'status': 'idle', 'stage': 'Ready', 'log': '', 'result': None}

    def snapshot(self):
        with self.lock:
            return dict(self.state)

    def update(self, **values):
        with self.lock:
            self.state.update(values)

    def start(self, session, color):
        session = Path(session).expanduser().resolve()
        if not session.is_dir() or not any(session.glob('*.hdf5')):
            raise ValueError('Select a saved session folder containing .hdf5 files')
        if color not in ('rgb', 'bgr'):
            raise ValueError('Color order must be rgb or bgr')
        if self.output == session or self.output in session.parents:
            raise ValueError('Keep recordings outside the generated output directory')
        with self.lock:
            if self.stopping or self.state['status'] == 'running':
                raise RuntimeError('A phase job is already running or the server is stopping')
            job = self.output / (datetime.now().strftime('%Y%m%d_%H%M%S_') + uuid4().hex[:8])
            job.mkdir(parents=True, exist_ok=False)
            self.state = {'status': 'running', 'stage': 'Starting inference', 'log': '',
                          'result': None, 'output': str(job), 'session': str(session)}
            self.thread = threading.Thread(target=self.work, args=(session, color, job), daemon=True)
            self.thread.start()
        return self.snapshot()

    def command(self, args, job):
        with self.lock:
            if self.stopping:
                raise RuntimeError('Server stopped')
            self.process = subprocess.Popen([str(self.python), '-u', *map(str, args)],
                                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                            text=True, errors='replace', cwd=HERE)
            process = self.process
        with (job / 'run.log').open('a', encoding='utf-8') as log:
            for line in process.stdout:
                log.write(line)
                log.flush()
                with self.lock:
                    self.state['log'] = (self.state['log'] + line)[-24000:]
        process.stdout.close()
        code = process.wait()
        with self.lock:
            self.process = None
        if code:
            raise RuntimeError(f'Processing exited with code {code}. See the log below.')

    def work(self, session, color, job):
        try:
            self.update(stage='Running V-JEPA2 and TCN')
            self.command([HERE / 'infer_phase_hdf5.py', '--session', session,
                          '--checkpoint', self.checkpoint, '--color-order', color,
                          '--output', job / 'predictions'], job)
            self.update(stage='Assigning predictions to frames')
            self.command([HERE / 'prepare_phase_input.py',
                          '--predictions', job / 'predictions/phase_predictions.csv',
                          '--summary', job / 'predictions/summary.json',
                          '--session', session, '--output', job / 'phase_input'], job)
            result = read_result(job / 'phase_input/model_phase_frames.csv')
            self.update(status='complete', stage='Phase detection complete', result=result)
        except Exception as exc:
            self.update(status='failed', stage=str(exc))
        finally:
            (job / 'job.json').write_text(json.dumps(self.snapshot(), indent=2), encoding='utf-8')

    def close(self):
        with self.lock:
            self.stopping = True
            process = self.process
        if process and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        if self.thread:
            self.thread.join(timeout=7)


def handler(runner, token):
    class Handler(BaseHTTPRequestHandler):
        def send(self, status, value, mime='application/json'):
            body = value if isinstance(value, bytes) else json.dumps(value).encode()
            self.send_response(status)
            self.send_header('Content-Type', mime)
            self.send_header('Content-Length', str(len(body)))
            self.send_header('Cache-Control', 'no-store')
            self.send_header('X-Content-Type-Options', 'nosniff')
            self.send_header('X-Frame-Options', 'DENY')
            self.end_headers()
            self.wfile.write(body)

        def local(self):
            expected = f'127.0.0.1:{self.server.server_port}'
            return self.headers.get('Host') == expected and self.headers.get('Origin') in (None, f'http://{expected}')

        def do_GET(self):
            if not self.local():
                return self.send(403, {'error': 'Use the printed local address'})
            path = urlparse(self.path).path
            if path == '/':
                return self.send(200, (HERE / 'index.html').read_bytes(), 'text/html; charset=utf-8')
            if path == '/api/config':
                return self.send(200, {'token': token, 'session': runner.session, 'color': runner.color})
            if path == '/api/status':
                return self.send(200, runner.snapshot())
            return self.send(404, {'error': 'Not found'})

        def do_POST(self):
            if not self.local() or self.headers.get('X-Phase-Token') != token:
                return self.send(403, {'error': 'Reload the local page'})
            if self.path != '/api/run':
                return self.send(404, {'error': 'Not found'})
            try:
                length = int(self.headers.get('Content-Length', '0'))
                if not 0 < length <= 8192:
                    raise ValueError('Invalid request size')
                value = json.loads(self.rfile.read(length))
                if not isinstance(value, dict):
                    raise ValueError('Expected JSON object')
                result = runner.start(str(value.get('session', '')), str(value.get('color', '')))
                self.send(202, result)
            except RuntimeError as exc:
                self.send(409, {'error': str(exc)})
            except (ValueError, OSError) as exc:
                self.send(400, {'error': str(exc)})

        def log_message(self, *args):
            pass
    return Handler


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--python', required=True, help='Python executable in the inference environment')
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--session', required=True, help='Initial saved HDF5 session folder')
    p.add_argument('--color-order', choices=['bgr', 'rgb'], required=True)
    p.add_argument('--output', default='results/phase_jobs')
    p.add_argument('--port', type=int, default=8795)
    args = p.parse_args()
    runner = Runner(args.python, args.checkpoint, args.output, args.session, args.color_order)
    server = ThreadingHTTPServer(('127.0.0.1', args.port), handler(runner, secrets.token_urlsafe(32)))
    print(f'Phase detection: http://127.0.0.1:{server.server_port}/', flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        runner.close()
        server.server_close()


if __name__ == '__main__':
    main()
