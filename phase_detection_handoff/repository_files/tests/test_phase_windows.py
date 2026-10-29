import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location('phase_inference', Path(__file__).parents[1] / 'scripts/infer_phase_hdf5.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_windows_do_not_cross_gaps():
    times = [i / 10 for i in range(101)] + [16 + i / 10 for i in range(101)]
    windows = module.window_plan(times, 5, 1, .5)
    assert len(windows) == 12
    assert all(w['end_s'] <= 10 or w['start_s'] >= 16 for w in windows)
    assert {w['segment_id'] for w in windows} == {0, 1}


def test_short_recording_has_no_window():
    assert module.window_plan([0, .1, .2], 5, 1, .5) == []


def test_exact_window_is_kept():
    assert len(module.window_plan([i / 10 for i in range(51)], 5, 1, .5)) == 1
