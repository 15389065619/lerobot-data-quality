import json, glob, os, numpy as np, torch, traceback
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.policies.factory import make_pre_post_processors
ROOT = '/kaggle/input/datasets/happyday1223/so101-kaggle-ab-v1'
CP = '/kaggle/working/eval_cache_320x50.pt'
if os.path.exists(CP):
    CACHE = torch.load(CP, weights_only=False, map_location='cpu')
    print('CACHE_LOADED', len(CACHE), flush=True)
else:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    J = json.load(open(ROOT + '/eval_windows.json'))
    WIN = J['windows']; H = J['horizon']; FPS = 30.0
    ds = LeRobotDataset(repo_id='happyday1223/so101_eval', root=ROOT + '/so101_eval', video_backend='pyav', tolerance_s=0.01, delta_timestamps={'action': [i / FPS for i in range(H)]})
    epcol = [int(e) for e in ds.hf_dataset['episode_index']]
    starts = {}
    for i, e in enumerate(epcol): starts.setdefault(e, i)
    gidx = [starts[w['episode_index']] + w['start_frame'] for w in WIN]
    print('DS', len(ds), len(gidx), flush=True)
    CACHE = [[ds[i] for i in gidx[b:b + 8]] for b in range(0, len(gidx), 8)]
    torch.save(CACHE, CP)
    print('CACHE_BUILT', len(CACHE), flush=True)
PROBE = [True]
def to_float(o):
    if isinstance(o, dict):
        for k in ('loss', 'total_loss', 'l'):
            if k in o: return float(o[k])
        raise KeyError(list(o.keys()))
    if isinstance(o, (tuple, list)): return to_float(o[0])
    if torch.is_tensor(o):
        return float(o.reshape(-1)[0]) if o.numel() == 1 else float(o.float().mean())
    return float(o)
def run_ckpt(ck):
    pol = SmolVLAPolicy.from_pretrained(ck)
    pol.eval()
    pol.to('cuda')
    pre, _post = make_pre_post_processors(pol.config, ck)
    out = []
    with torch.no_grad():
        for bi, items in enumerate(CACHE):
            batch = {}
            for k in items[0]:
                vals = [it[k] for it in items]
                batch[k] = torch.stack(vals) if torch.is_tensor(vals[0]) else vals
            r = pre(batch)
            batch = r[0] if isinstance(r, (tuple, list)) else r
            batch = {k: (v.to('cuda') if torch.is_tensor(v) else v) for k, v in batch.items()}
            torch.manual_seed(bi)
            o = pol(batch)
            if PROBE[0]:
                PROBE[0] = False
                if isinstance(o, dict):
                    print('PROBE dict', list(o.keys()), flush=True)
                elif isinstance(o, (tuple, list)):
                    print('PROBE seq', len(o), [type(x).__name__ for x in o], flush=True)
                else:
                    print('PROBE other', type(o).__name__, getattr(o, 'shape', None), flush=True)
            out.append(to_float(o))
    del pol
    torch.cuda.empty_cache()
    return out
RF = '/kaggle/working/eval_results.json'
res = json.load(open(RF)) if os.path.exists(RF) else {}
for arm in ['arm_A', 'arm_B']:
    for d in sorted(glob.glob('/kaggle/working/' + arm + '/checkpoints/*/pretrained_model')):
        step = d.split('/')[-2]
        if arm in res.get(step, {}): continue
        try:
            res.setdefault(step, {})[arm] = run_ckpt(d)
        except Exception as e:
            print('ERR', arm, step, repr(e)[:300], flush=True)
            print('TB', traceback.format_exc()[-1200:], flush=True)
            continue
        json.dump(res, open(RF, 'w'))
        print('done', arm, step, flush=True)
for step in sorted(res):
    a = np.array(res[step].get('arm_A', []))
    b = np.array(res[step].get('arm_B', []))
    if len(a) and len(a) == len(b):
        d = a - b
        print(step, 'A=%.5f B=%.5f A-B=%.5f SE=%.5f' % (a.mean(), b.mean(), d.mean(), d.std(ddof=1) / np.sqrt(len(d))), flush=True)
print('FINAL_DONE', flush=True)
