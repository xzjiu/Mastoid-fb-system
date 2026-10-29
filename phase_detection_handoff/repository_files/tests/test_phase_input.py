import csv
import importlib.util
import json
import unittest
from pathlib import Path

spec=importlib.util.spec_from_file_location('adapter',Path(__file__).parents[1]/'scripts/prepare_phase_input.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

class TestPhaseInput(unittest.TestCase):
    def setUp(self):
        self.summary=dict(schema='phase_hdf5_inference_v1',time_origin='first_vision_frame',
                          labels=list(m.LABELS), partial_run=False,time_origin_epoch_s=100,
                          duration_s=20,window_s=5,selected_windows=2,gaps=[dict(start_s=6,end_s=10)])
        self.rows=[]
        for a,probs in [(0,[.8,.1,.1]),(1,[.2,.1,.7])]:
            i=max(range(3),key=lambda j:probs[j])
            self.rows.append(dict(start_s=a,end_s=a+5,start_epoch_s=100+a,end_epoch_s=105+a,
                                  segment_id=0,pred_id=i,pred_phase=m.LABELS[i],
                                  **dict(zip(['prob_'+n for n in m.LABELS],probs))))
    def test_overlap(self):
        r=m.PhaseInput(self.rows,self.summary).at(102)
        self.assertAlmostEqual(r['prob_pre_antrum'],.5)
        self.assertEqual(r['contributing_windows'],2)
    def test_unknown_and_half_open(self):
        p=m.PhaseInput(self.rows,self.summary)
        for t in [99,106,108,120]: self.assertEqual(p.at(t)['model_phase'],'unknown')
    def test_gap_rejected(self):
        self.summary['gaps']=[dict(start_s=2,end_s=3)]
        with self.assertRaises(ValueError):m.PhaseInput(self.rows,self.summary)
    def test_wrong_origin_rejected(self):
        self.summary['time_origin_epoch_s']=101
        with self.assertRaises(ValueError):m.PhaseInput(self.rows,self.summary)
    def test_partial_rejected(self):
        self.summary['partial_run']=True
        with self.assertRaises(ValueError):m.PhaseInput(self.rows,self.summary)
    def test_invalid_probability(self):
        self.rows[0]['prob_antrum']=-.1
        with self.assertRaises(ValueError):m.PhaseInput(self.rows,self.summary)

if __name__=='__main__':unittest.main()
