#!/usr/bin/env python3
# scripts/reextract_pred_offline.py — 离线终值重抽取重评分（规则抽取器不读取金标）
#
# 目的：对已有 refine 输出中"叙述性长句"形态的终值做规则抽取，用原判分器重算
#       thought→refine 转移矩阵（TT/TF/FT/FF），量化格式毁损的可挽回比例。
# 安全设计：仅对长度 > SHORT_KEEP 的输出做改写。2B QA 判对样本 pred 实测
#          max=44 字符（p99=37），阈值 48 保证已判对样本全部原样保留 → 零新毁损，
#          改写只作用于原判错样本。
# 判分：复刻 refine/{TableQA,TableFV}/utils/evaluate.py 的 match_func 判分逻辑
#       （chain[-1] 首参数 + to_value_list/check_denotation 或 label 比对），不修改判分器。
# 抽取规则（QA，按优先级降级，全部兜底"保持原样"）：
#   A. 显式答案标记（"the answer is/:"，全文最后一个）
#   B. 末句最后一个 is/are/was/were 表语，截断到逗号/连接词
#   C. 末句冒号后枚举
# 抽取规则（FV）：短值原样；长文本取最后一个 yes/no/true/false 词。
# 用法：python3 scripts/reextract_pred_offline.py                 # 2B 全部 4 组
#       python3 scripts/reextract_pred_offline.py --models 2b,0.8b

import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from critic.TableFV.tools.read_pkl import read_pkl
import refine.TableQA.utils.evaluate as qa_ev

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SHORT_KEEP = 48  # 长度阈值：≤48 原样保留（判对样本实测 max=44，零新毁损的保证）
SPLIT_PAT = r'(?:,|;|\bwhich\b|\bwhere\b|\bwhile\b|\bbecause\b|\bbut\b|\bso\b|\bas\b|\band\b)'


def _clean(x):
    return x.strip().rstrip('.').strip()


def extract_qa(text):
    """QA 规则抽取：短值原样；长文本按 显式标记→末句表语→冒号→原样 降级。"""
    t = str(text).strip()
    if len(t) <= SHORT_KEEP:
        return t
    low = t.lower()
    # A. 显式答案标记（全文最后一个）
    ms = list(re.finditer(r'(?:final answer|the answer|answer)\s*(?:is|:)\s*(.+)', low))
    if ms:
        cand = _clean(re.split(SPLIT_PAT, ms[-1].group(1))[0])
        if cand:
            return cand
    # B. 末句最后一个 is/are/was/were 表语
    sents = re.split(r'(?<=[.!?])\s+', t)
    last = _clean(sents[-1])
    ms = list(re.finditer(r'\b(?:is|are|was|were)\s+(.+)$', last.lower().rstrip('.')))
    if ms:
        cand = _clean(re.split(SPLIT_PAT, ms[-1].group(1))[0])
        if cand:
            return cand
    # C. 末句冒号后枚举
    if ':' in last:
        cand = _clean(last.rsplit(':', 1)[-1])
        if cand:
            return cand
    return t  # 兜底：原样（保持判错不变，绝不冒险改写）


def extract_fv(text):
    """FV 规则抽取：短值原样；长文本取最后一个 yes/no/true/false 词；否则原样。"""
    t = str(text).strip()
    if len(t) <= SHORT_KEEP:
        return t
    ms = list(re.finditer(r'\b(yes|no|true|false)\b', t.lower()))
    return ms[-1].group(1) if ms else t


def load_qa_targets():
    """从 tagged_data 构建 id → 金标值集合 映射（与 match_func 同源）。"""
    tvm = {}
    tagged = os.path.join(REPO, 'thought/TableQA/data/wikitq/tagged_data')
    for fn in os.listdir(tagged):
        if fn[0] == '.':
            continue
        with open(os.path.join(tagged, fn), encoding='utf8') as fin:
            header = fin.readline().rstrip('\n').split('\t')
            for line in fin:
                stuff = dict(zip(header, line.rstrip('\n').split('\t')))
                tvm[stuff['id']] = qa_ev.to_value_list(
                    qa_ev.tsv_unescape_list(stuff['targetValue']),
                    qa_ev.tsv_unescape_list(stuff['targetCanon']))
    return tvm


def match_qa(pred_str, targets):
    res = str(pred_str).lower()
    preds = [res] if '|' not in res else res.split('|')
    return bool(qa_ev.check_denotation(targets, qa_ev.to_value_list(preds)))


def match_fv(pred_str, label):
    res = str(pred_str).lower()
    if res == 'true':
        res = 'yes'
    if res == 'false':
        res = 'no'
    return (res == 'yes' and label == 1) or (res == 'no' and label == 0)


def last_pred(sample):
    return sample['chain'][-1]['parameter_and_conf'][0][0]


def judge_pred(pred_str, task, sample, qa_targets):
    if task == 'wikitq':
        return match_qa(pred_str, qa_targets[sample['ids']])
    return match_fv(pred_str, sample['label'])


def eval_group(framework, task, model, qa_targets):
    """评估单组：返回 (统计, 明细)。thought 判定不重算（orig 从 thought pkl 判分，
    g 直接用 attribution_detail.jsonl 的 thought_correct），refine 旧/新各判一次。"""
    if framework == 'orig':
        refine_pkl = f'/root/code/Table-Critic/results/refine/{task}/qwen3.5-{model}/final_result.pkl'
        thought_pkl = f'/root/code/Table-Critic/results/thought/{task}/qwen3.5-{model}/final_result.pkl'
        th = {s['id']: s for s in read_pkl(thought_pkl) if isinstance(s, dict) and s.get('id')}
        samples = [s for s in read_pkl(refine_pkl) if isinstance(s, dict) and s.get('id')]

        def get_thought_ok(sid, s):
            ts = th.get(sid)
            return judge_pred(last_pred(ts), task, ts, qa_targets) if ts else None
    else:
        base = f'{REPO}/test/results_len32k/refine/{task}/qwen3.5-{model}_p12'
        refine_pkl = f'{base}/final_result.pkl'
        samples = [s for s in read_pkl(refine_pkl) if isinstance(s, dict) and s.get('id')]
        att = {}
        with open(f'{base}/attribution_detail.jsonl') as f:
            for line in f:
                o = json.loads(line)
                att[o['id']] = bool(o['thought_correct'])

        def get_thought_ok(sid, s):
            return att.get(sid)

    extract = extract_fv if task == 'tabfact' else extract_qa
    tt = tf = ft = ff = 0
    n_tt = n_tf = n_ft = n_ff = 0
    rescued, damaged, details = [], [], []
    for s in samples:
        t_ok = get_thought_ok(s['id'], s)
        if t_ok is None:
            continue
        pred_old = last_pred(s)
        old_ok = judge_pred(pred_old, task, s, qa_targets)
        pred_new = extract(pred_old)
        new_ok = judge_pred(pred_new, task, s, qa_targets)

        # 旧矩阵
        if t_ok and old_ok: tt += 1
        elif t_ok and not old_ok: tf += 1
        elif not t_ok and old_ok: ft += 1
        else: ff += 1
        # 新矩阵
        if t_ok and new_ok: n_tt += 1
        elif t_ok and not new_ok: n_tf += 1
        elif not t_ok and new_ok: n_ft += 1
        else: n_ff += 1

        if new_ok != old_ok:
            (rescued if new_ok else damaged).append(s['id'])
        details.append({
            'id': s['id'], 'thought_ok': bool(t_ok), 'old_ok': old_ok, 'new_ok': new_ok,
            'pred_old': str(pred_old)[:400], 'pred_new': str(pred_new)[:400],
        })

    n = tt + tf + ft + ff
    stats = {
        'group': f'{framework}/{task}/{model}', 'n': n,
        'old': {'acc': (tt + ft) / n * 100, 'TT': tt, 'TF': tf, 'FT': ft, 'FF': ff},
        'new': {'acc': (n_tt + n_ft) / n * 100, 'TT': n_tt, 'TF': n_tf, 'FT': n_ft, 'FF': n_ff},
        'rescued': len(rescued), 'damaged': len(damaged),
        'rescued_ids': rescued[:20], 'damaged_ids': damaged[:20],
    }
    return stats, details


def main(models='2b'):
    qa_targets = load_qa_targets()
    out_dir = os.path.join(REPO, 'results/reextract')
    os.makedirs(out_dir, exist_ok=True)
    model_list = [m.strip() for m in models.split(',')]
    all_stats = []
    for model in model_list:
        for framework in ['orig', 'g']:
            for task in ['wikitq', 'tabfact']:
                stats, details = eval_group(framework, task, model, qa_targets)
                all_stats.append(stats)
                dpath = os.path.join(out_dir, f'{framework}_{task}_{model}_detail.jsonl')
                with open(dpath, 'w', encoding='utf8') as f:
                    for d in details:
                        f.write(json.dumps(d, ensure_ascii=False) + '\n')
                st, sd = stats['old'], stats['new']
                print(f"[{stats['group']}] n={stats['n']}")
                print(f"  旧: acc={st['acc']:.2f}  TT={st['TT']} TF={st['TF']} FT={st['FT']} FF={st['FF']}")
                print(f"  新: acc={sd['acc']:.2f}  TT={sd['TT']} TF={sd['TF']} FT={sd['FT']} FF={sd['FF']}")
                print(f"  挽回 {stats['rescued']} 题, 新毁损 {stats['damaged']} 题, Δacc={sd['acc']-st['acc']:+.2f}pp  (明细: {dpath})")
    # 汇总写盘
    spath = os.path.join(out_dir, f'summary_{"_".join(model_list)}.json')
    with open(spath, 'w', encoding='utf8') as f:
        json.dump(all_stats, f, ensure_ascii=False, indent=2)
    print(f"\n汇总已写入 {spath}")


if __name__ == '__main__':
    import fire
    fire.Fire(main)
