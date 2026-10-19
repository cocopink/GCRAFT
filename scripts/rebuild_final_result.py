"""从 cache 重建 final_result.pkl（仅跑 fixed_chain = simple_query）
用法: python rebuild_final_result.py [--tabfact] [--wikitq] [--n_proc N]
"""
import os, sys, pickle, argparse

def rebuild_tabfact(base_url, api_key, model, n_proc):
    sys.path.insert(0, 'thought/TableFV')
    sys.path.insert(0, 'critic/TableFV')
    sys.path.insert(0, '.')

    from utils.llm import LLM
    from operations import simple_query
    from utils.chain import fixed_chain_exec_mp
    from utils.evaluate import tabfact_match_func_for_samples

    results_dir = 'test/results/thought/tabfact/gpt-5.4'
    cache_dir = os.path.join(results_dir, 'cache')

    # 1. 从 cache 加载 proc_samples（按索引顺序）
    proc_samples = []
    idx = 0
    while True:
        cache_path = os.path.join(cache_dir, f'case-{idx}.pkl')
        if not os.path.exists(cache_path):
            break
        _, proc_sample, _ = pickle.load(open(cache_path, 'rb'))
        proc_samples.append(proc_sample)
        idx += 1
    print(f'[TabFact] Loaded {len(proc_samples)} proc_samples from cache', flush=True)

    # 2. 删除旧的 final_result.pkl
    final_path = os.path.join(results_dir, 'final_result.pkl')
    if os.path.exists(final_path):
        os.remove(final_path)
        print(f'[TabFact] Removed old {final_path}', flush=True)

    # 3. 初始化 LLM 并跑 fixed_chain（simple_query）
    gpt_llm = LLM(model_name=model, key=api_key, base=base_url)
    gpt_llm.set_token_log_dir(results_dir, run_tag='tabfact_thought_rebuild')

    fixed_chain = [
        ("Simple query", simple_query, dict(use_demo=True),
         dict(temperature=0, per_example_max_decode_steps=2048, per_example_top_p=1.0)),
    ]
    final_result, _ = fixed_chain_exec_mp(
        gpt_llm, proc_samples, fixed_chain, n_proc=n_proc, chunk_size=2
    )

    # 4. 保存并计算准确率
    pickle.dump(final_result, open(final_path, 'wb'))
    acc = tabfact_match_func_for_samples(final_result)
    print(f'[TabFact] Thought Stage Accuracy: {acc} ({len(final_result)} samples)', flush=True)
    with open(os.path.join(results_dir, 'acc.txt'), 'w') as f:
        f.write(f'Thought Stage Accuracy: {acc}\n')


def rebuild_wikitq(base_url, api_key, model, n_proc):
    sys.path.insert(0, 'thought/TableQA')
    sys.path.insert(0, 'critic/TableQA')
    sys.path.insert(0, '.')

    from utils.llm import LLM
    from operations import simple_query
    from utils.chain import fixed_chain_exec_mp
    from utils.evaluate import wikitq_match_func_for_samples

    results_dir = 'test/results/thought/wikitq/gpt-5.4'
    cache_dir = os.path.join(results_dir, 'cache')

    # 1. 从 cache 加载 proc_samples（按索引顺序）
    proc_samples = []
    idx = 0
    while True:
        cache_path = os.path.join(cache_dir, f'case-{idx}.pkl')
        if not os.path.exists(cache_path):
            break
        _, proc_sample, _ = pickle.load(open(cache_path, 'rb'))
        proc_samples.append(proc_sample)
        idx += 1
    print(f'[WikiTQ] Loaded {len(proc_samples)} proc_samples from cache', flush=True)

    # 2. 删除旧的 final_result.pkl
    final_path = os.path.join(results_dir, 'final_result.pkl')
    if os.path.exists(final_path):
        os.remove(final_path)
        print(f'[WikiTQ] Removed old {final_path}', flush=True)

    # 3. 初始化 LLM 并跑 fixed_chain（simple_query）
    gpt_llm = LLM(model_name=model, key=api_key, base=base_url)
    gpt_llm.set_token_log_dir(results_dir, run_tag='wikitq_thought_rebuild')

    fixed_chain = [
        ("Simple query", simple_query, dict(use_demo=True),
         dict(temperature=0, per_example_max_decode_steps=2048, per_example_top_p=1.0)),
    ]
    final_result, _ = fixed_chain_exec_mp(
        gpt_llm, proc_samples, fixed_chain, n_proc=n_proc, chunk_size=2
    )

    # 4. 保存并计算准确率
    pickle.dump(final_result, open(final_path, 'wb'))
    acc = wikitq_match_func_for_samples(final_result)
    print(f'[WikiTQ] Thought Stage Accuracy: {acc} ({len(final_result)} samples)', flush=True)
    with open(os.path.join(results_dir, 'acc.txt'), 'w') as f:
        f.write(f'Thought Stage Accuracy: {acc}\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--tabfact', action='store_true', default=False)
    parser.add_argument('--wikitq', action='store_true', default=False)
    parser.add_argument('--n_proc', type=int, default=4)
    args = parser.parse_args()

    base_url = 'https://113.44.247.131:47851/v1'
    api_key = os.environ.get('ANTHROPIC_AUTH_TOKEN', '')
    model = 'gpt-5.4'

    if not args.tabfact and not args.wikitq:
        args.tabfact = True
        args.wikitq = True

    if args.tabfact:
        rebuild_tabfact(base_url, api_key, model, args.n_proc)
    if args.wikitq:
        rebuild_wikitq(base_url, api_key, model, args.n_proc)
