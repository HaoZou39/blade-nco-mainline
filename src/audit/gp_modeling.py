"""Frozen symbolic-GP protocol localized for the Cycle-0 audit only."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from typing import Any, Iterable

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


INVALID_FITNESS = 1e12
BINARY_OPS = ("add", "subtract", "multiply", "protected_divide")
UNARY_OPS = ("abs", "sqrt_abs", "log1p_abs")
MINMAX_BINARY_OPS = BINARY_OPS + ("min", "max")


@dataclass(frozen=True)
class Node:
    op: str
    args: tuple["Node", ...] = ()
    value: float | int | None = None


@dataclass
class Individual:
    tree: Node
    fitness: float = INVALID_FITNESS
    huber: float = INVALID_FITNESS


def tree_to_dict(node: Node) -> dict[str, Any]:
    return {"op":node.op, "value":node.value, "args":[tree_to_dict(arg) for arg in node.args]}


def tree_from_dict(payload: dict[str, Any]) -> Node:
    return Node(str(payload["op"]), tuple(tree_from_dict(arg) for arg in payload.get("args", [])), payload.get("value"))


def serialize_tree(node: Node) -> str:
    """Stage 4A canonicalization: stable JSON of the exact expression tree."""
    return json.dumps(tree_to_dict(node), sort_keys=True, separators=(",", ":"))


def tree_string(node: Node, names: list[str]) -> str:
    if node.op == "feature": return names[int(node.value)]
    if node.op == "const": return f"{float(node.value):.12g}"
    if len(node.args) == 2: return f"{node.op}({tree_string(node.args[0],names)}, {tree_string(node.args[1],names)})"
    return f"{node.op}({tree_string(node.args[0],names)})"


def op_count(node: Node, op: str) -> int:
    return int(node.op == op) + sum(op_count(arg, op) for arg in node.args)


def tree_nodes(node: Node) -> int:
    return 1 + sum(tree_nodes(arg) for arg in node.args)


def tree_depth(node: Node) -> int:
    return 1 + (max(tree_depth(arg) for arg in node.args) if node.args else 0)


def feature_indices(node: Node) -> set[int]:
    if node.op == "feature": return {int(node.value)}
    return set().union(*(feature_indices(arg) for arg in node.args)) if node.args else set()


def constant_values(node: Node) -> list[float]:
    if node.op == "const": return [float(node.value)]
    return [value for arg in node.args for value in constant_values(arg)]


def all_paths(node: Node, prefix: tuple[int,...]=()) -> list[tuple[int,...]]:
    return [prefix, *[path for index,arg in enumerate(node.args) for path in all_paths(arg,(*prefix,index))]]


def subtree_at(node: Node, path: tuple[int,...]) -> Node:
    for index in path: node=node.args[index]
    return node


def replace_subtree(node: Node, path: tuple[int,...], replacement: Node) -> Node:
    if not path: return replacement
    args=list(node.args); args[path[0]]=replace_subtree(args[path[0]],path[1:],replacement)
    return Node(node.op,tuple(args),node.value)


def evaluate_tree(node: Node, x: np.ndarray) -> np.ndarray:
    if node.op == "feature": return x[:,int(node.value)]
    if node.op == "const": return np.full(len(x),float(node.value),dtype=float)
    a=evaluate_tree(node.args[0],x)
    if node.op == "abs": return np.abs(a)
    if node.op == "sqrt_abs": return np.sqrt(np.abs(a))
    if node.op == "log1p_abs": return np.log1p(np.abs(a))
    b=evaluate_tree(node.args[1],x)
    if node.op == "add": return a+b
    if node.op == "subtract": return a-b
    if node.op == "multiply": return a*b
    if node.op == "protected_divide":
        denominator=np.where(np.abs(b)<1e-6,np.where(b>=0,1e-6,-1e-6),b)
        return a/denominator
    if node.op == "min": return np.minimum(a,b)
    if node.op == "max": return np.maximum(a,b)
    raise ValueError(f"unknown primitive: {node.op}")


def random_tree(rng: np.random.Generator, n_features: int, minimum: int, maximum: int, depth: int=0, target: int|None=None, binary_ops: tuple[str,...]=BINARY_OPS, unary_ops: tuple[str,...]=UNARY_OPS) -> Node:
    target=int(rng.integers(minimum,maximum+1)) if target is None else target
    if depth>=target or (depth>=minimum and rng.random()<.25):
        return Node("feature",value=int(rng.integers(n_features))) if rng.random()<.82 else Node("const",value=float(rng.uniform(-2,2)))
    if rng.random()<.62:
        return Node(str(rng.choice(binary_ops)),(random_tree(rng,n_features,minimum,maximum,depth+1,target,binary_ops,unary_ops),random_tree(rng,n_features,minimum,maximum,depth+1,target,binary_ops,unary_ops)))
    return Node(str(rng.choice(unary_ops)),(random_tree(rng,n_features,minimum,maximum,depth+1,target,binary_ops,unary_ops),))


def valid_tree(node: Node, config: dict[str,Any]) -> bool:
    return tree_depth(node)<=int(config["maximum_depth"]) and tree_nodes(node)<=int(config["maximum_nodes"]) and len(feature_indices(node))<=int(config["maximum_unique_features"])


def random_valid(rng: np.random.Generator, n_features: int, config: dict[str,Any], binary_ops: tuple[str,...]=BINARY_OPS, unary_ops: tuple[str,...]=UNARY_OPS) -> Individual:
    for _ in range(200):
        tree=random_tree(rng,n_features,int(config["initial_depth_min"]),int(config["initial_depth_max"]),binary_ops=binary_ops,unary_ops=unary_ops)
        if valid_tree(tree,config): return Individual(tree)
    return Individual(Node("feature",value=int(rng.integers(n_features))))


def safe_scaled_prediction(tree: Node, x: np.ndarray, config: dict[str,Any]) -> tuple[np.ndarray,float,bool]:
    try:
        raw=evaluate_tree(tree,x)
        valid=raw.shape==(len(x),) and np.isfinite(raw).all() and (len(raw)==0 or float(np.max(np.abs(raw)))<=float(config["prediction_invalid_threshold"]))
        if not valid: return np.zeros(len(x)),1.0,False
        clipped=np.clip(raw,-float(config["fitness_prediction_clip"]),float(config["fitness_prediction_clip"]))
        return clipped,float(np.mean(clipped!=raw)),True
    except Exception:
        return np.zeros(len(x)),1.0,False


def fitness(tree: Node, x: np.ndarray, y: np.ndarray, config: dict[str,Any]) -> tuple[float,float]:
    pred,_,valid=safe_scaled_prediction(tree,x,config)
    if not valid: return INVALID_FITNESS,INVALID_FITNESS
    error=np.abs(pred-y); delta=float(config["huber_delta"])
    loss=float(np.mean(np.where(error<=delta,.5*np.square(error),delta*(error-.5*delta))))
    return loss+float(config["parsimony_coefficient"])*tree_nodes(tree),loss


def tournament(population: list[Individual], rng: np.random.Generator, size: int) -> Individual:
    indices=rng.choice(len(population),size=min(size,len(population)),replace=False)
    return min((population[int(index)] for index in indices),key=lambda item:item.fitness)


def mutate_point(node: Node, rng: np.random.Generator, n_features: int, binary_ops: tuple[str,...]=BINARY_OPS, unary_ops: tuple[str,...]=UNARY_OPS) -> Node:
    path=all_paths(node)[int(rng.integers(len(all_paths(node))))]; current=subtree_at(node,path)
    if current.op=="feature": replacement=Node("feature",value=int(rng.integers(n_features)))
    elif current.op=="const": replacement=Node("const",value=float(rng.uniform(-2,2)))
    elif len(current.args)==1: replacement=Node(str(rng.choice(unary_ops)),current.args)
    else: replacement=Node(str(rng.choice(binary_ops)),current.args)
    return replace_subtree(node,path,replacement)


def train_seed(x: np.ndarray, y: np.ndarray, names: list[str], config: dict[str,Any], seed: int, observer=None, fitness_fn=None, on_init=None, log_enricher=None, binary_ops: tuple[str,...]|None=None, unary_ops: tuple[str,...]|None=None, init_binary_ops: tuple[str,...]|None=None, checkpoint_generations: Iterable[int]|None=None, on_checkpoint=None, stats: dict[str,Any]|None=None, parent_selector: str="tournament", selection_log: list|None=None, init_n_features: int|None=None) -> tuple[pd.DataFrame,list[dict[str,Any]]]:
    rng=np.random.default_rng(seed); size=int(config["population_size"])
    probabilities=np.asarray([config["crossover_probability"],config["subtree_mutation_probability"],config["point_mutation_probability"],config["hoist_mutation_probability"],config["reproduction_probability"]],float)
    if not np.isclose(probabilities.sum(),1.0,rtol=0,atol=1e-15): raise ValueError("variation probabilities do not sum to one")
    cumulative=np.cumsum(probabilities)
    evaluate=fitness if fitness_fn is None else fitness_fn
    vary_binary=tuple(binary_ops) if binary_ops is not None else BINARY_OPS
    vary_unary=tuple(unary_ops) if unary_ops is not None else UNARY_OPS
    init_binary=tuple(init_binary_ops) if init_binary_ops is not None else vary_binary
    checkpoints=set(int(g) for g in (checkpoint_generations or ()))
    n_vary=int(x.shape[1])
    n_init=int(init_n_features) if init_n_features is not None else n_vary
    if n_init < 1 or n_init > n_vary:
        raise ValueError("init_n_features must lie in 1..x.shape[1]")
    population=[random_valid(rng,n_init,config,init_binary,vary_unary) for _ in range(size)]
    if on_init is not None:
        on_init(tuple(item.tree for item in population))
    cache:dict[str,tuple[float,float]]={}; archive:dict[str,Individual]={}; first_seen:dict[str,int]={}; logs=[]
    import time
    start=time.perf_counter()
    for generation in range(int(config["generations"])+1):
        for individual in population:
            key=serialize_tree(individual.tree)
            if key not in cache:
                cache[key]=evaluate(individual.tree,x,y,config)
                first_seen[key]=generation
            individual.fitness,individual.huber=cache[key]
            prior=archive.get(key)
            if prior is None or individual.fitness<prior.fitness: archive[key]=Individual(individual.tree,individual.fitness,individual.huber)
        population.sort(key=lambda item:(item.fitness,serialize_tree(item.tree)))
        finite=[item for item in population if item.fitness<INVALID_FITNESS]
        usage=np.zeros(x.shape[1],dtype=int)
        for item in finite:
            for index in feature_indices(item.tree): usage[index]+=1
        best=population[0]
        row={
            "seed":seed,"generation":generation,"best_total_fitness":best.fitness,"best_huber":best.huber,
            "best_nodes":tree_nodes(best.tree),"best_depth":tree_depth(best.tree),"population_median_fitness":float(np.median([item.fitness for item in population])),
            "population_valid_fraction":len(finite)/size,"population_unique_formula_count":len({serialize_tree(item.tree) for item in population}),
            "mean_nodes":float(np.mean([tree_nodes(item.tree) for item in population])),"mean_depth":float(np.mean([tree_depth(item.tree) for item in population])),
            "feature_usage_frequency":json.dumps({names[index]:int(value) for index,value in enumerate(usage) if value},sort_keys=True),"runtime_sec":time.perf_counter()-start,
        }
        if log_enricher is not None:
            row.update(log_enricher(generation, population, best))
        logs.append(row)
        if observer is not None:
            # Observer-only snapshot: copies Individuals so the callback cannot mutate GP state or consume RNG.
            observer(generation, tuple(Individual(item.tree, item.fitness, item.huber) for item in population))
        if on_checkpoint is not None and generation in checkpoints:
            hof_now=sorted(archive.values(),key=lambda item:(item.fitness,serialize_tree(item.tree)))[:int(config["hall_of_fame_size"])]
            on_checkpoint(
                generation,
                tuple(Individual(item.tree, item.fitness, item.huber) for item in population),
                tuple(Individual(item.tree, item.fitness, item.huber) for item in hof_now),
                dict(first_seen),
            )
        if generation==int(config["generations"]): break
        parent_counts=np.zeros(size,dtype=np.int64)
        lex_gen=None
        filter_cases=[]
        inspected_mask=None
        decisive_counts=None
        t_select=0.0
        if parent_selector=="automatic_epsilon_lexicase":
            from blade_nco.gp.selection.epsilon_lexicase import prepare_generation as _prep_lex, select_parent as _lex_parent
            lex_gen=_prep_lex(population,x,y,config)
            inspected_mask=np.zeros(lex_gen.n_cases,dtype=bool)
            decisive_counts=np.zeros(lex_gen.n_cases,dtype=np.int64)
        elif parent_selector!="tournament":
            raise ValueError(f"unknown parent_selector: {parent_selector}")
        next_population=[Individual(item.tree,item.fitness,item.huber) for item in population[:int(config["elitism_count"])]]
        while len(next_population)<size:
            draw=rng.random()
            t0=time.perf_counter()
            if lex_gen is not None:
                idx,meta=_lex_parent(lex_gen,rng)
                parent_counts[idx]+=1
                parent=population[idx].tree
                filter_cases.append(meta["cases_used"])
                inspected_mask[meta["inspected"]]=True
                decisive_counts[int(meta["decisive_case"])]+=1
                t_select+=time.perf_counter()-t0
                if draw<cumulative[0]:
                    t1=time.perf_counter()
                    idx2,meta2=_lex_parent(lex_gen,rng)
                    parent_counts[idx2]+=1
                    other=population[idx2].tree
                    filter_cases.append(meta2["cases_used"])
                    inspected_mask[meta2["inspected"]]=True
                    decisive_counts[int(meta2["decisive_case"])]+=1
                    t_select+=time.perf_counter()-t1
                    paths_a=all_paths(parent); paths_b=all_paths(other)
                    child=replace_subtree(parent,paths_a[int(rng.integers(len(paths_a)))],subtree_at(other,paths_b[int(rng.integers(len(paths_b)))]))
                elif draw<cumulative[1]:
                    paths=all_paths(parent); child=replace_subtree(parent,paths[int(rng.integers(len(paths)))],random_tree(rng,n_vary,1,3,binary_ops=vary_binary,unary_ops=vary_unary))
                elif draw<cumulative[2]: child=mutate_point(parent,rng,n_vary,vary_binary,vary_unary)
                elif draw<cumulative[3]:
                    paths=all_paths(parent); child=subtree_at(parent,paths[int(rng.integers(len(paths)))])
                else: child=parent
            else:
                parent_ind=tournament(population,rng,int(config["tournament_size"]))
                parent_counts[next(i for i,item in enumerate(population) if item is parent_ind)]+=1
                parent=parent_ind.tree
                t_select+=time.perf_counter()-t0
                if draw<cumulative[0]:
                    t1=time.perf_counter()
                    other_ind=tournament(population,rng,int(config["tournament_size"]))
                    parent_counts[next(i for i,item in enumerate(population) if item is other_ind)]+=1
                    other=other_ind.tree
                    t_select+=time.perf_counter()-t1
                    paths_a=all_paths(parent); paths_b=all_paths(other)
                    child=replace_subtree(parent,paths_a[int(rng.integers(len(paths_a)))],subtree_at(other,paths_b[int(rng.integers(len(paths_b)))]))
                elif draw<cumulative[1]:
                    paths=all_paths(parent); child=replace_subtree(parent,paths[int(rng.integers(len(paths)))],random_tree(rng,n_vary,1,3,binary_ops=vary_binary,unary_ops=vary_unary))
                elif draw<cumulative[2]: child=mutate_point(parent,rng,n_vary,vary_binary,vary_unary)
                elif draw<cumulative[3]:
                    paths=all_paths(parent); child=subtree_at(parent,paths[int(rng.integers(len(paths)))])
                else: child=parent
            next_population.append(Individual(child) if valid_tree(child,config) else random_valid(rng,n_vary,config,vary_binary,vary_unary))
        if selection_log is not None:
            draws=int(parent_counts.sum())
            probs=parent_counts.astype(np.float64)/draws if draws else np.zeros(size)
            nz=probs[probs>0]
            entropy=float(-np.sum(nz*np.log(nz))) if len(nz) else 0.0
            neff=float(np.exp(entropy)) if draws else 0.0
            order=np.sort(parent_counts)[::-1]
            row={
                "seed":seed,"generation":generation,"parent_selector":parent_selector,
                "unique_parents_selected":int((parent_counts>0).sum()),
                "unique_parent_fraction":float((parent_counts>0).mean()),
                "maximum_parent_selection_count":int(parent_counts.max()) if draws else 0,
                "maximum_parent_selection_fraction":float(parent_counts.max()/draws) if draws else 0.0,
                "top5_parent_selection_fraction":float(order[:5].sum()/draws) if draws else 0.0,
                "top10_parent_selection_fraction":float(order[:10].sum()/draws) if draws else 0.0,
                "n_eff_parent":neff,"n_eff_parent_fraction":neff/size if size else 0.0,
                "hhi_parent":float(np.sum(np.square(probs))),"n_parent_draws":draws,
                "parent_selection_runtime_sec":t_select,
            }
            if lex_gen is not None and filter_cases:
                arr=np.asarray(filter_cases,float)
                row.update({
                    **{f"epsilon_{k}":v for k,v in lex_gen.epsilon_summary.items()},
                    "median_cases_used":float(np.median(arr)),"mean_cases_used":float(np.mean(arr)),
                    "p90_cases_used":float(np.quantile(arr,0.9)),"max_cases_used":float(np.max(arr)),
                    "fraction_resolved_after_1_case":float(np.mean(arr==1)),
                    "fraction_resolved_within_5_cases":float(np.mean(arr<=5)),
                    "fraction_reaching_all_cases":float(np.mean(arr==lex_gen.n_cases)),
                    "n_distinct_cases_inspected":int(inspected_mask.sum()),
                    "n_decisive_cases":int((decisive_counts>0).sum()),
                    "decisive_case_coverage_fraction":float((decisive_counts>0).mean()),
                    "zero_epsilon_fraction":lex_gen.zero_epsilon_fraction,
                })
            selection_log.append(row)
        population=next_population
    hof=sorted(archive.values(),key=lambda item:(item.fitness,serialize_tree(item.tree)))[:int(config["hall_of_fame_size"])]
    records=[{"seed":seed,"hof_rank":rank+1,"tree":tree_to_dict(item.tree),"canonical_formula":serialize_tree(item.tree),"formula":tree_string(item.tree,names),"nodes":tree_nodes(item.tree),"depth":tree_depth(item.tree),"feature_indices":sorted(feature_indices(item.tree)),"unique_features":len(feature_indices(item.tree)),"used_features":[names[index] for index in sorted(feature_indices(item.tree))],"constants":constant_values(item.tree),"train_huber":item.huber,"train_total_fitness":item.fitness,"first_generation":first_seen[serialize_tree(item.tree)]} for rank,item in enumerate(hof)]
    if stats is not None:
        stats["n_unique_evaluated"]=len(cache)
        stats["n_archive"]=len(archive)
        stats["n_evaluation_slots"]=(int(config["generations"])+1)*size
        stats["n_unique_invalid"]=sum(
            1 for total_fitness,_ in cache.values() if total_fitness>=INVALID_FITNESS
        )
        stats["final_population_unique_fraction"]=len(
            {serialize_tree(item.tree) for item in population}
        )/size
        stats["final_population_median_expression_size"]=float(
            np.median([tree_nodes(item.tree) for item in population])
        )
    return pd.DataFrame(logs),records


def target_transform(train_gap: np.ndarray) -> dict[str,Any]:
    y=np.log1p(np.asarray(train_gap,float)); center=float(np.median(y)); iqr=float(np.subtract(*np.percentile(y,[75,25])))
    mad=float(np.median(np.abs(y-center))); std=float(np.std(y,ddof=1)); candidates=[("IQR",iqr),("1.4826_MAD",1.4826*mad),("STD",std),("ONE",1.0)]
    method,scale=next((name,value) for name,value in candidates if np.isfinite(value) and value>1e-12)
    return {"status":"LOG_GAP_HUBER_TARGET_FROZEN","fit_split":"TRAIN_ONLY","formula":"log1p(gap_node0_pct)","center_train_median_log_gap":center,"scale_train_log_gap":float(scale),"scale_method":method,"fallback_candidates":{name:float(value) for name,value in candidates}}


def scale_target(gap: np.ndarray, transform: dict[str,Any]) -> np.ndarray:
    return (np.log1p(np.asarray(gap,float))-float(transform["center_train_median_log_gap"]))/float(transform["scale_train_log_gap"])


def inverse_prediction(scaled: np.ndarray, transform: dict[str,Any]) -> np.ndarray:
    return np.expm1(np.asarray(scaled,float)*float(transform["scale_train_log_gap"])+float(transform["center_train_median_log_gap"]))


def predictive_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str,float]:
    actual=np.asarray(actual,float); predicted=np.asarray(predicted,float); n=len(actual)
    order=np.argsort(-predicted,kind="mergesort"); base=float(np.mean(actual)); output={
        "count":n,"r2":float(r2_score(actual,predicted)),"mae":float(mean_absolute_error(actual,predicted)),"rmse":float(mean_squared_error(actual,predicted)**.5),
        "pearson":float(pearsonr(actual,predicted).statistic) if np.std(predicted)>0 else 0.0,"spearman":float(spearmanr(actual,predicted).statistic) if np.std(predicted)>0 else 0.0,
    }
    for percent in (5,10,20):
        count=max(1,int(math.ceil(n*percent/100))); mean=float(np.mean(actual[order[:count]])); output[f"top{percent}_actual_mean_gap"]=mean; output[f"mean_gap_enrichment{percent}"]=mean/base if base else float("nan")
    top10=order[:max(1,int(math.ceil(n*.1)))]; hard90=actual>=np.quantile(actual,.9); output["p90_hit_enrichment10"]=float(np.mean(hard90[top10])/.1)
    top5=order[:max(1,int(math.ceil(n*.05)))]; hard95=actual>=np.quantile(actual,.95); output["p95_hit_rate"]=float(np.mean(hard95[top5]))
    return output


def evaluate_formula(record: dict[str,Any], x: np.ndarray, gap: np.ndarray, transform: dict[str,Any], config: dict[str,Any], split: str) -> dict[str,Any]:
    tree=tree_from_dict(record["tree"]); scaled,clip,valid=safe_scaled_prediction(tree,x,config); predicted=inverse_prediction(scaled,transform)
    metrics=predictive_metrics(gap,predicted) if valid else {key:float("nan") for key in ("r2","mae","rmse","pearson","spearman","mean_gap_enrichment10")}
    return {**record,**{f"{split.lower()}_{key}":value for key,value in metrics.items()},f"{split.lower()}_clip_fraction":clip,f"{split.lower()}_valid":valid}


def select_formula(records: list[dict[str,Any]]) -> dict[str,Any]:
    usable=[record for record in records if record.get("validation_valid") and record["validation_clip_fraction"]<=.01 and np.isfinite(record["validation_spearman"])]
    if not usable: raise RuntimeError("no validation-eligible HOF formula")
    best_spearman=max(record["validation_spearman"] for record in usable); candidates=[record for record in usable if record["validation_spearman"]>=best_spearman-.02]
    best_enrichment=max(record["validation_mean_gap_enrichment10"] for record in candidates); candidates=[record for record in candidates if record["validation_mean_gap_enrichment10"]>=best_enrichment-.10]
    return min(candidates,key=lambda record:(record["nodes"],record["validation_mae"],record["canonical_formula"]))
