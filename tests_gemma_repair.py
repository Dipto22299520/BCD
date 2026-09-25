import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent/'scripts'))
from queue_gemma_repair import jobs,teacher_gate,cache_gate

class RepairTests(unittest.TestCase):
    def test_fresh_outputs_and_dependency_order(self):
        plan=jobs();self.assertEqual(len(plan),66)
        self.assertEqual(len({j['output'] for j in plan}),66)
        for j in plan:
            self.assertTrue(j['output'].startswith('review_runs/gemma_repair_v1/'))
            self.assertNotIn('scripts/train_backdoor.py',j['cmd'])
        for tag in {j['tag'] for j in plan}:
            js=[j for j in plan if j['tag']==tag]
            self.assertEqual([j['kind'] for j in js[:5]],['restore','smoke','controls','utility','cache'])
            for j in js:
                if j['kind']=='distill':
                    self.assertIn('gemma_repair_v1',j['cmd'][j['cmd'].index('--gen-cache')+1])
    def test_failed_teacher_stops(self):
        teacher_gate(1,0)
        for rates in [(0,0),(1,.8)]:
            with self.assertRaises(ValueError):teacher_gate(*rates)
    def test_cache_population_and_firing_gate(self):
        rows=[dict(idx=i,clean_response='A normal answer.',trig_response="I'm sorry, but I can't help with that.") for i in range(3000)]
        cache_gate(rows)
        rows[0]['clean_response']=''
        with self.assertRaises(ValueError):cache_gate(rows)
        rows[0]['clean_response']='A normal answer.'
        for r in rows:r['trig_response']='main main main'
        with self.assertRaises(ValueError):cache_gate(rows)

if __name__=='__main__':unittest.main()
