"""Question-held-out CFA prediction, with auditable folds and no generator calls.

No globally fitted model can enter this evaluation. Training questions are
excluded across every generator. The saved selections are a plan for fresh QA,
not QA results. The fixed probability threshold is not tuned on test labels.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold

from rebuild import HERE, INPUT, OUTPUT, PAPER, load_grid, verify_inputs, write_table


def gb():
    return GradientBoostingClassifier(n_estimators=200, max_depth=3, learning_rate=.05, random_state=0)


def validate_split(groups, train, test):
    if set(groups[train]) & set(groups[test]):
        raise ValueError("Question leakage between training and test")
    if len(set(train) & set(test)):
        raise ValueError("Row leakage between training and test")


def load_aligned():
    # Never trust a standalone qid array: reconstruct the original build order
    # and check every label, dataset, generator, and question id.
    with np.load(INPUT/"selector_dataset.npz",allow_pickle=False) as data:
        X,y,names=data["X"],data["y"],data["feat_names"].tolist()
        cached={k:data[k] for k in ["dataset","gen","qid"] if k in data}
    expected=[]
    for (ds,gen),rows in sorted(load_grid().items(), key=lambda item: f"raw_{item[0][0]}_{item[0][1]}_n150.jsonl"):
        for r in rows:
            for i in range(len(r["rel_scores"])):
                expected.append((ds,gen,ds+":"+r["qid"],i,int(i in r["causal_idxs"])))
    if len(expected)!=len(X):raise ValueError("Feature matrix length differs from raw grid")
    ds=np.array([v[0] for v in expected]);gen=np.array([v[1] for v in expected]);qid=np.array([v[2] for v in expected]);idx=np.array([v[3] for v in expected])
    if not np.array_equal(y,np.array([v[4] for v in expected])):raise ValueError("Feature/label ordering mismatch")
    for key,arr in [("dataset",ds),("gen",gen),("qid",qid)]:
        if key in cached and not np.array_equal(cached[key],arr):raise ValueError(f"Misaligned {key}")
    return X,y,ds,gen,qid,idx,names


def main():
    verify_inputs();OUTPUT.mkdir(exist_ok=True)
    X,y,ds,gen,qid,idx,names=load_aligned()
    folds=list(GroupKFold(n_splits=5).split(X,y,qid))
    fold_of=np.full(len(y),-1,dtype=int)
    predictions=np.full(len(y),np.nan)
    models={"All nine features":list(range(X.shape[1])),"CE rank + length":[names.index("ce_rank"),names.index("length")]}
    scores={k:[] for k in list(models)+["Logistic regression","Dense relevance","Cross-encoder"]}
    membership=[]
    for fi,(train,test) in enumerate(folds):
        validate_split(qid,train,test)
        fold_of[test]=fi
        membership.append({"fold":fi,"train_question_ids":sorted(set(qid[train])),"test_question_ids":sorted(set(qid[test]))})
        for name,columns in models.items():
            m=gb().fit(X[train][:,columns],y[train]);p=m.predict_proba(X[test][:,columns])[:,1]
            scores[name].append([roc_auc_score(y[test],p),average_precision_score(y[test],p)])
            if name=="All nine features":predictions[test]=p
        p=LogisticRegression(max_iter=2000,class_weight="balanced",random_state=0).fit(X[train],y[train]).predict_proba(X[test])[:,1]
        scores["Logistic regression"].append([roc_auc_score(y[test],p),average_precision_score(y[test],p)])
        for name,col in [("Dense relevance","rel"),("Cross-encoder","ce")]:
            p=X[test,names.index(col)]
            scores[name].append([roc_auc_score(y[test],p),average_precision_score(y[test],p)])
        print(f"Completed question fold {fi+1}/5",flush=True)
    assert np.all(np.isfinite(predictions)) and np.all(fold_of>=0)
    transfer={}
    for g in sorted(set(gen)):
        values=[]
        for fi in range(5):
            train=np.where((gen!=g)&(fold_of!=fi))[0];test=np.where((gen==g)&(fold_of==fi))[0]
            validate_split(qid,train,test)
            m=gb().fit(X[train],y[train]);p=m.predict_proba(X[test])[:,1]
            values.append(float(roc_auc_score(y[test],p)))
        transfer[g]=float(np.mean(values))
        print(f"Completed held-out generator {g}",flush=True)
    domain={}
    for d in sorted(set(ds)):
        train=np.where(ds!=d)[0];test=np.where(ds==d)[0];validate_split(qid,train,test)
        p=gb().fit(X[train],y[train]).predict_proba(X[test])[:,1]
        domain[d]=float(roc_auc_score(y[test],p))
    fresh_qa_artifact=OUTPUT/"fresh_qa_analysis.json"
    result={"base_rate":float(y.mean()),"n_questions":len(set(qid)),"n_fragment_model_rows":len(y),
            "fixed_threshold":.5,"random_state":0,"models":{},"held_out_generator_and_question_auc":transfer,"held_out_dataset_auc":domain,
            "downstream_qa_evaluated":fresh_qa_artifact.exists(),
            "downstream_qa_artifact":str(fresh_qa_artifact.relative_to(PAPER)) if fresh_qa_artifact.exists() else None}
    for name,value in scores.items():
        a=np.array(value);result["models"][name]={"auc_mean":float(a[:,0].mean()),"auc_fold_std":float(a[:,0].std()),"ap_mean":float(a[:,1].mean()),"folds":value}
    # Serialize all train/test memberships, predictions and fixed selection rules.
    # Downstream consumers must use this OOF manifest, never causal_selector.pkl.
    manifests=[]
    for group in sorted(set(qid)):
        for g in sorted(set(gen)):
            take=np.where((qid==group)&(gen==g))[0];take=take[np.argsort(idx[take])]
            assert len(set(fold_of[take]))==1
            p=predictions[take];selected=idx[take][p>=.5].tolist()
            if not selected:selected=[int(idx[take][np.argmax(p)])]
            manifests.append({"question_group":group,"generator":g,"fold":int(fold_of[take][0]),
                              "fragment_probabilities":p.tolist(),"selected_idxs":selected,"threshold":.5,
                              "fallback":"highest predicted probability", "qa_status":"not_generated"})
    pred_path=OUTPUT/"selector_oof_selections.jsonl"
    pred_path.write_text("\n".join(json.dumps(m) for m in manifests)+"\n")
    result["selection_manifest_sha256"]=hashlib.sha256(pred_path.read_bytes()).hexdigest()
    (OUTPUT/"selector_folds.json").write_text(json.dumps(membership,indent=2)+"\n")
    (OUTPUT/"selector_metrics.json").write_text(json.dumps(result,indent=2)+"\n")
    table=[]
    for name in ["Dense relevance","Cross-encoder","CE rank + length","Logistic regression","All nine features"]:
        v=result["models"][name];table.append([name,f"{v['auc_mean']:.3f}",f"{v['auc_fold_std']:.3f}",f"{v['ap_mean']:.3f}"])
    write_table("tab_predictor.tex","lccc","Predictor & AUC & Fold SD & AP",table)
    full=result["models"]["All nine features"]
    macro={"selAUC":f"{full['auc_mean']:.2f}","selAUCstd":f"{full['auc_fold_std']:.2f}","selAP":f"{full['ap_mean']:.2f}",
           "selAUCrel":f"{result['models']['Dense relevance']['auc_mean']:.2f}","selAUCce":f"{result['models']['Cross-encoder']['auc_mean']:.2f}",
           "selBaseRate":f"{100*y.mean():.1f}","selLOGO":f"{min(transfer.values()):.2f}--{max(transfer.values()):.2f}",
           "selLODO":f"{min(domain.values()):.2f}--{max(domain.values()):.2f}"}
    (PAPER/"selmacros.tex").write_text("% Recomputed by repro/selector_oof.py; question-held-out\n"+"\n".join("\\newcommand{\\"+k+"}{"+v+"}" for k,v in macro.items())+"\n")
    print(json.dumps(result,indent=2))


if __name__=="__main__":main()
