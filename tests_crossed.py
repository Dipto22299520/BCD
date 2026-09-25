import sys
from pathlib import Path
import unittest
import tempfile,json
from unittest.mock import patch
import numpy as np
sys.path.insert(0,str(Path(__file__).parent/'scripts'))
import queue_crossed
from queue_crossed import cells,expected_hash
from bcd.perturb import build_ladder_set

class CrossedTests(unittest.TestCase):
    def test_complete_factorial(self):
        cs=cells()
        self.assertEqual(len(cs),48)
        self.assertEqual(len({c['output'] for c in cs}),48)
        self.assertEqual(len({(c['tag'],c['condition']) for c in cs}),24)
        self.assertEqual({c['implant_seed'] for c in cs},{0,1,2})
        for p in ('random_word','append'):
            self.assertEqual(len({expected_hash(c) for c in cs if c['probe_placement']==p}),1)

    def test_formatting_and_control_pairing(self):
        base=['  one  two\nthree','one two?']
        for place in ('append','random_word'):
            rows=build_ladder_set(base,'rare',variant_set='specificity_multi_v1',placement=place,probe_format='word_slots_v1')
            for i,b in enumerate(base):
                chunk=[r for r in rows if r['id']==i]
                self.assertEqual(chunk[-1]['instruction'],' '.join(b.split()))
                positions=[]
                for r in chunk[:-1]:
                    w=r['instruction'].split();text='tq' if r['rung']=='exact' else r['rung'][6:]
                    positions.append(w.index(text));w.remove(text)
                    self.assertEqual(w,b.split())
                self.assertEqual(len(set(positions)),1)
                if place=='append':self.assertEqual(positions[0],len(b.split()))

    def test_legacy_unchanged_and_unknown_refused(self):
        args=dict(base_instructions=['one two'],trigger_name='rare',placement='append',variant_set='specificity_multi_v1')
        self.assertEqual(build_ladder_set(**args),build_ladder_set(**args,probe_format='legacy'))
        self.assertEqual(build_ladder_set(**args)[0]['instruction'],'one two. tq')
        self.assertEqual(build_ladder_set(**args,probe_format='word_slots_v1')[0]['instruction'],'one two tq')
        with self.assertRaises(ValueError):build_ladder_set(**args,probe_format='unknown')

    def test_aggregate_recovers_known_probe_interaction(self):
        def fake_read(c):
            # Teacher selectivity .2; student .3 random, .6 appended.
            rate=.3 if c['condition']=='C0' else (.4 if c['probe_placement']=='random_word' else .7)
            d=dict(variants=[dict(name='exact',group='exact'),dict(name='wrong',group='wrong_token'),dict(name='none',group='none')],n_base=4,n_samples=8)
            return d,dict(exact=np.full(4,rate),wrong=np.full(4,.1),none=np.zeros(4))
        with tempfile.TemporaryDirectory() as tmp, patch.object(queue_crossed,'ROOT_PATH',Path(tmp)),patch.object(queue_crossed,'read',side_effect=fake_read):
            queue_crossed.aggregate()
            d=json.loads((Path(tmp)/'results/crossed_placement.json').read_text())
            self.assertEqual(len(d['cells']),48)
            for v in d['probe_interactions'].values():
                self.assertAlmostEqual(v['mean'],.3)
                self.assertAlmostEqual(v['lo'],.3)
                self.assertAlmostEqual(v['hi'],.3)

if __name__=='__main__':unittest.main()
