def tabfact_match_func(sample, strategy="top"):
    results = sample["chain"][-1]["parameter_and_conf"]

    if strategy == "top":
        res = results[0][0]
    elif strategy == "weighted":
        res_conf_dict = {}
        for res, conf in results:
            if res not in res_conf_dict:
                res_conf_dict[res] = 0
            res_conf_dict[res] += conf
        res_conf_rank = sorted(res_conf_dict.items(), key=lambda x: x[1], reverse=True)
        res = res_conf_rank[0][0]
    else:
        raise NotImplementedError

    res = res.lower()
    if res == "true":
        res = "yes"
    if res == "false":
        res = "no"
    if res == "yes" and sample["label"] == 1:
        return True
    elif res == "no" and sample["label"] == 0:
        return True
    else:
        return False


def tabfact_match_func_for_samples(all_samples, strategy="top"):
    correct_list = []
    for i, sample in enumerate(all_samples):
        try:
            if tabfact_match_func(sample, strategy):
                correct_list.append(1)
            else:
                correct_list.append(0)
        except:
            print(f"Error in sample {i}")
            continue
    if not correct_list:
        print(f"WARNING: all {len(all_samples)} samples failed evaluation.")
        return 0.0
    return sum(correct_list) / len(correct_list)
