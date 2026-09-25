"""Read-only model diagnosis except one explicitly saved teacher round-trip copy."""
import os,sys,json,gc,time
os.environ.pop('HF_TOKEN',None)
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import torch
from peft import PeftModel
from bcd.models import load_model,load_tokenizer
from bcd.scoring import build_prompt
ROOT=Path(__file__).resolve().parents[1]

def main():
    os.chdir(ROOT)
    adapter='runs/bd_gemma3_random_word_s0_p10'
    base=json.loads(Path(adapter,'adapter_config.json').read_text(encoding='utf-8'))['base_model_name_or_path']
    tok=load_tokenizer(base)
    rows=[json.loads(s) for s in Path('data/eval/rare/eval_triggered.jsonl').read_text(encoding='utf-8').splitlines()][:3]
    prompts=[]
    for r in rows:
        prompts.extend([r['base_instruction'],r['instruction']])
    result={};target="I'm sorry, but I can't help with that."
    def probe(model,label):
        model.eval();model.config.use_cache=True
        values=[];logits=[]
        for p in prompts:
            text=build_prompt(tok,p);ids=tok(text,return_tensors='pt',add_special_tokens=False).to('cuda')
            with torch.inference_mode():
                first=model(**ids).logits[0,-1].float().cpu();logits.append(first)
                full=tok(text+target,return_tensors='pt',add_special_tokens=False).to('cuda')
                n=ids['input_ids'].shape[1]
                scores=model(**full).logits[0,n-1:-1].float().log_softmax(-1)
                seq=full['input_ids'][0,n:]
                lp=float(scores.gather(-1,seq[:,None]).sum())
                generated=model.generate(**ids,max_new_tokens=48,do_sample=False,repetition_penalty=1.0)
            answer=tok.decode(generated[0,n:],skip_special_tokens=True)
            values.append(dict(instruction=p,answer=answer,target_log_probability=lp,first_token=int(first.argmax())))
        result[label]=values
        print(label,json.dumps(values,ensure_ascii=True),flush=True)
        return torch.stack(logits)
    model=load_model(base);probe(model,'base')
    model=PeftModel.from_pretrained(model,adapter);probe(model,'adapter')
    model=model.merge_and_unload();before=probe(model,'merged_memory')
    out=ROOT/'review_runs/gemma_diagnosis/teacher_s0_roundtrip'
    out.mkdir(parents=True,exist_ok=True)
    model.save_pretrained(out,safe_serialization=True);tok.save_pretrained(out)
    del model;gc.collect();torch.cuda.empty_cache()
    model=load_model(str(out));after=probe(model,'reloaded_fixed')
    result['roundtrip_max_abs_logit_difference']=float((after-before).abs().max())
    result['roundtrip_top_tokens_equal']=bool(torch.equal(after.argmax(-1),before.argmax(-1)))
    dest=ROOT/'results/gemma_diagnosis.json';dest.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print('DONE',result['roundtrip_max_abs_logit_difference'],result['roundtrip_top_tokens_equal'],flush=True)

if __name__=='__main__':main()
