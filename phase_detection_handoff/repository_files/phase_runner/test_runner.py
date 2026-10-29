import json
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from urllib.request import Request, urlopen
from urllib.error import HTTPError
from http.server import ThreadingHTTPServer
from unittest.mock import patch
import serve


class RunnerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.session = self.root / 'session'
        self.session.mkdir()
        (self.session / 'one.hdf5').touch()
        self.checkpoint = self.root / 'model.pt'
        self.checkpoint.touch()
        self.runner = serve.Runner(sys.executable, self.checkpoint, self.root / 'out', self.session, 'bgr')

    def tearDown(self):
        self.runner.close()
        self.tmp.cleanup()

    def test_complete_pipeline_and_unique_output(self):
        commands = []
        def fake(args, job):
            commands.append(args)
            if Path(args[0]).name == 'prepare_phase_input.py':
                out = job / 'phase_input'
                out.mkdir()
                (out / 'model_phase_frames.csv').write_text('t_video_s,model_phase\n0,pre_antrum\n1,incus\n2,unknown\n')
        with patch.object(self.runner, 'command', side_effect=fake):
            first = self.runner.start(self.session, 'bgr')['output']
            self.runner.thread.join(3)
            self.assertEqual(self.runner.snapshot()['status'], 'complete')
            self.assertEqual(self.runner.snapshot()['result']['frame_count'], 3)
            self.assertEqual(len(commands), 2)
            second = self.runner.start(self.session, 'rgb')['output']
            self.runner.thread.join(3)
            self.assertNotEqual(first, second)
            self.assertTrue((Path(first) / 'job.json').is_file())

    def test_failure_stops_pipeline(self):
        with patch.object(self.runner, 'command', side_effect=RuntimeError('model failed')) as command:
            self.runner.start(self.session, 'bgr')
            self.runner.thread.join(3)
            self.assertEqual(self.runner.snapshot()['status'], 'failed')
            self.assertEqual(command.call_count, 1)
            self.assertIsNone(self.runner.snapshot()['result'])

    def test_concurrent_start_rejected(self):
        started, release = threading.Event(), threading.Event()
        def fake(*args):
            started.set()
            release.wait(3)
            raise RuntimeError('test finished')
        with patch.object(self.runner, 'command', side_effect=fake):
            self.runner.start(self.session, 'bgr')
            self.assertTrue(started.wait(2))
            try:
                with self.assertRaises(RuntimeError):
                    self.runner.start(self.session, 'bgr')
            finally:
                release.set()
                self.runner.thread.join(3)

    def test_invalid_result(self):
        csv = self.root / 'bad.csv'
        csv.write_text('t_video_s,model_phase\nnan,incus\n')
        with self.assertRaises(ValueError):
            serve.read_result(csv)

    def test_http_routes_and_guard(self):
        server = ThreadingHTTPServer(('127.0.0.1', 0), serve.handler(self.runner, 'test-token'))
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        base = f'http://127.0.0.1:{server.server_port}'
        try:
            with urlopen(base) as r:
                self.assertIn(b'Run Phase Detection', r.read())
            with urlopen(base + '/api/status') as r:
                self.assertEqual(json.load(r)['status'], 'idle')
            with self.assertRaises(HTTPError) as e:
                urlopen(Request(base + '/api/run', data=b'{}', method='POST'))
            self.assertEqual(e.exception.code, 403)
            with patch.object(self.runner, 'start', return_value={'status': 'running'}) as start:
                req = Request(base + '/api/run', data=json.dumps({'session': str(self.session), 'color': 'bgr'}).encode(), headers={'X-Phase-Token': 'test-token'}, method='POST')
                with urlopen(req) as r:
                    self.assertEqual(r.status, 202)
                start.assert_called_once_with(str(self.session), 'bgr')
        finally:
            server.shutdown()
            server.server_close()
            thread.join(2)

    def test_real_subprocess_error(self):
        job = self.root / 'job'
        job.mkdir()
        with self.assertRaises(RuntimeError):
            self.runner.command(['-c', 'print("failure detail"); raise SystemExit(2)'], job)
        self.assertIn('failure detail', (job / 'run.log').read_text())

if __name__ == '__main__':
    unittest.main()
