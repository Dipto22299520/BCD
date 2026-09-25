"""Download the authorized Gemma pair and record text-distillation compatibility.

Uses Hugging Face's stored login, ignoring a stale process HF_TOKEN override.
Never records credentials in output. This does not start scientific training.
"""
import os
os.environ.pop('HF_TOKEN', None)
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'
import json
from pathlib import Path
from huggingface_hub import snapshot_download
from transformers import AutoConfig, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]

def main():
    records=[]; toks=[]
    for name in ('google/gemma-3-4b-it','google/gemma-3-1b-it'):
        print('DOWNLOAD', name, flush=True)
        path=snapshot_download(name, allow_patterns=['*.json','*.safetensors','*.model','*.jinja'], max_workers=2)
        cfg=AutoConfig.from_pretrained(path,local_files_only=True)
        text=getattr(cfg,'text_config',cfg)
        tok=AutoTokenizer.from_pretrained(path,local_files_only=True)
        prompt=tok.apply_chat_template([{'role':'user','content':'Say hello.'}],tokenize=False,add_generation_prompt=True)
        records.append(dict(model=name,snapshot=Path(path).name,local_path=path,
                            architecture=cfg.architectures,text_vocab_size=text.vocab_size,
                            tokenizer_size=len(tok),prompt=prompt))
        toks.append(tok)
        print('READY',name,'text vocab',text.vocab_size,'tokenizer',len(tok),flush=True)
    vocabs=[t.get_vocab() for t in toks]
    common=set(vocabs[0])&set(vocabs[1])
    mismatches=sum(vocabs[0][t]!=vocabs[1][t] for t in common)
    direct=(records[0]['text_vocab_size']==records[1]['text_vocab_size'] and
            vocabs[0]==vocabs[1] and records[0]['prompt']==records[1]['prompt'])
    report=dict(models=records,common_token_id_mismatches=mismatches,
                direct_logit_distillation_compatible=direct,
                training_started=False,
                next_check='Verify text-only model loading and GPU forward/backward memory before training.')
    dest=ROOT/'results/gemma_preflight.json';dest.parent.mkdir(exist_ok=True)
    dest.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print('WROTE',dest,'direct compatibility:',direct,flush=True)
    if not direct:print('STOP: vocabulary/prompt alignment requires inspection before logit distillation.',flush=True)

if __name__=='__main__':main()
