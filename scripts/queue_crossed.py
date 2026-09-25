"""Frozen 48-cell crossed-placement follow-up; no training or checkpoint deletion."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

from queue_submission import ROOT_PATH, PY, ENV, control_cells
from bcd.data import read_jsonl
from bcd.perturb import build_ladder_set, variants_for
from submission_tables import read_cell, control_stats, interval
import numpy as np

OUT = ROOT_PATH / 'review_runs/crossed_word_slots_v1'

def cells():
    out=[]
    for c in control_cells():
        if c['trigger'] != 'rare' or c['tag'].startswith('cleansft') or 'cap' in c['tag']:
            continue
        for probe in ('random_word','append'):
            d=dict(c, train_placement='append' if 'append' in c['tag'] else 'random_word',
                   probe_placement=probe, data='data/eval/rare')
            d['output']=str(OUT/c['tag']/c['condition']/probe)
            out.append(d)
    return out

def expected_hash(c):
    base=[r['base_instruction'] for r in read_jsonl(ROOT_PATH/c['data']/'eval_triggered.jsonl')][:120]
    rows=build_ladder_set(base,'rare',seed=0,variant_set='specificity_multi_v1',
                          placement=c['probe_placement'],probe_format='word_slots_v1')
    return hashlib.sha256(json.dumps([r['instruction'] for r in rows],ensure_ascii=False).encode()).hexdigest()

def read(c):
    p=Path(c['output'])/'calibration_v2.json'
    if not p.exists():return None
    d,r=read_cell(p,'specificity_multi_v1',True)
    expected={'model':c['model'],'adapter':c.get('adapter'),'condition':c['condition'],
              'probe_format':'word_slots_v1','placement':c['probe_placement'],
              'n_base':120,'n_samples':8,'sampling_seed':0,'data':c['data'],
              'trigger':'rare','quant':'none','rtn_bits':None}
    for k,v in expected.items():
        if d.get(k)!=v:raise ValueError(f'{p}: incompatible {k}')
    if bool(d.get('merge_adapter'))!=bool(c.get('adapter')):raise ValueError(f'{p}: merge mismatch')
    if d.get('prompt_population_sha256')!=expected_hash(c):raise ValueError(f'{p}: prompt mismatch')
    if [(v['name'],v['text']) for v in d['variants']] != [(v.name,v.text) for v in variants_for('rare','specificity_multi_v1')]:
        raise ValueError(f'{p}: control mismatch')
    return d,r

def gpu_idle():
    result=subprocess.run(['nvidia-smi','--query-gpu=memory.used,utilization.gpu','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True)
    memory,util=map(int,result.stdout.splitlines()[0].split(','))
    return memory<2000 and util<10, memory,util

def aggregate():
    values={}; report={}
    for c in cells():
        item=read(c)
        if item is None:raise RuntimeError('Crossed evaluation incomplete; no partial final report written')
        d,r=item;s,v=control_stats(d,r)
        key=(c['tag'],c['condition'],c['probe_placement'])
        values[key]=v;report['/'.join(key)]=s
    changes={}; shifts={}
    for c in cells():
        if c['condition']!='C1-kd05':continue
        tag,p=c['tag'],c['probe_placement']
        changes[tag+'/'+p]=interval(values[tag,'C1-kd05',p]-values[tag,'C0',p])
        for cond in ('C0','C1-kd05'):
            shifts[tag+'/'+cond]=interval(values[tag,cond,'append']-values[tag,cond,'random_word'])
    lines=['# Crossed placement: word_slots_v1','',
           'All 48 cells complete. Intervals pair base instructions and condition on checkpoints and the frozen controls.',
           'Probe format is newly matched; do not pool these cells with legacy controls. Training pipelines still differ in placement.',
           '', '| Training arm/seed | Delta at random probe | Delta at appended probe | Probe interaction |', '|---|---|---|---|']
    interactions={}
    for c in cells():
        if c['condition']!='C1-kd05' or c['probe_placement']!='append':continue
        tag=c['tag']; v=(values[tag,'C1-kd05','append']-values[tag,'C0','append'])-(values[tag,'C1-kd05','random_word']-values[tag,'C0','random_word'])
        interactions[tag]=interval(v)
        def fmt(x):return f"{x['mean']:+.4f} [{x['lo']:+.4f}, {x['hi']:+.4f}]"
        lines.append(f"| {tag} | {fmt(changes[tag+'/random_word'])} | {fmt(changes[tag+'/append'])} | {fmt(interactions[tag])} |")
    dest=ROOT_PATH/'results';dest.mkdir(exist_ok=True)
    (dest/'crossed_placement.json').write_text(json.dumps(dict(cells=report,teacher_student_changes=changes,within_checkpoint_probe_shifts=shifts,probe_interactions=interactions),indent=2))
    (dest/'crossed_placement.md').write_text('\n'.join(lines),encoding='utf-8')
    print('Wrote results/crossed_placement.md',flush=True)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--dry-run',action='store_true');ap.add_argument('--wait-for-gpu',action='store_true');ap.add_argument('--aggregate-only',action='store_true');a=ap.parse_args()
    os.chdir(ROOT_PATH)
    plan=cells(); pending=[c for c in plan if read(c) is None]
    print(f'Crossed-placement evaluations: {len(plan)-len(pending)}/{len(plan)} complete',flush=True)
    if a.dry_run:
        for c in pending:print(c['tag'],c['condition'],c['probe_placement'])
        return
    if a.aggregate_only:aggregate();return
    if shutil.disk_usage(ROOT_PATH).free<2*2**30:raise SystemExit('Need 2 GiB free for evaluation artifacts.')
    # Keep a Windows process lock across GPU waiting and the serial queue.
    lock=open(ROOT_PATH/'logs/crossed_queue.lock','a+b')
    if os.name=='nt':
        import msvcrt
        lock.seek(0);lock.write(b'0');lock.flush();lock.seek(0)
        try:msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
        except OSError:raise SystemExit('Another crossed-placement queue is already active.')
    for c in pending:
        idle,mem,util=gpu_idle()
        while not idle:
            if not a.wait_for_gpu:raise SystemExit(f'GPU busy ({mem} MiB, {util}%). Use --wait-for-gpu or rerun after the other job finishes.')
            print(f'WAIT GPU: {mem} MiB, {util}%; checking again in 30 seconds',flush=True)
            time.sleep(30);idle,mem,util=gpu_idle()
        cmd=[PY,'scripts/eval_calibration_v2.py','--model',c['model'],'--data',c['data'],'--trigger','rare',
             '--placement',c['probe_placement'],'--probe-format','word_slots_v1','--variant-set','specificity_multi_v1',
             '--condition',c['condition'],'--n-base','120','--n-samples','8','--seed','0','--out',c['output']]
        if c.get('adapter'):cmd+=['--adapter',c['adapter'],'--merge-adapter']
        log=ROOT_PATH/'logs'/f"crossed_{c['tag']}_{c['condition']}_{c['probe_placement']}.log"
        print('RUN',log.name,flush=True)
        with log.open('w',encoding='utf-8') as f:
            rc=subprocess.run(cmd,env={**ENV,'PYTHONUTF8':'1','PYTHONIOENCODING':'utf-8'},stdout=f,stderr=subprocess.STDOUT).returncode
        if rc or read(c) is None:raise SystemExit(f'FAILED: {log}; resume with the same command after inspection.')
    aggregate()

if __name__=='__main__':main()
