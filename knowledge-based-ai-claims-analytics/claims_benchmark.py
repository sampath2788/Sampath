"""Synthetic, reproducible evidence-sufficiency experiment for claims coverage review.

Usage: python claims_benchmark.py --out results
This is NOT a claims-adjudication system. All members, benefit rules and observations
are synthetic; no real-world coverage/policy validity is asserted.
"""
from __future__ import annotations
import argparse
import json
import platform
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import HistGradientBoostingClassifier

FIELDS = ('service', 'start', 'end', 'member_product', 'claim_product')
FEATURES = (*FIELDS, 'amount', 'network', 'start_distance', 'end_distance',
            'product_agreement', 'n_missing', 'chronology_conflict')
SEEDS = [11, 23, 37, 53, 71, 89, 107, 131, 157]
LEVELS = [0., .1, .2, .3]


def simulate(n_members: int, seed: int, corruption: float,
             mode: str = 'mixed', member_offset: int = 0) -> pd.DataFrame:
    """Generate latent clean claims and separately corrupted observed evidence.

    Three claims share each member's latent enrollment start/end and product.
    No hidden labels, latent fields, or member IDs are passed into the learner.
    Corruption is a scenario knob, not an empirical field-error estimate.
    """
    if not 0 <= corruption <= 1 or mode not in ('mixed', 'missing', 'silent'):
        raise ValueError('Invalid corruption setting/mode')
    r = np.random.default_rng(seed)
    member = np.repeat(np.arange(n_members), 3)
    start_m = r.integers(70, 300, n_members)
    end_m = start_m + r.integers(135, 335, n_members)
    product_m = r.integers(1, 5, n_members)
    a, b, m = start_m[member].astype(float), end_m[member].astype(float), product_m[member].astype(float)
    n = len(member)
    y = (r.random(n) < .63)
    wrong_by_date = (~y) & (r.random(n) < .65)
    wrong_by_product = (~y) & (~wrong_by_date)
    s = a + r.random(n) * (b - a)
    before = (r.random(n) < .5)
    gap = r.integers(8, 135, n)
    s[wrong_by_date & before] = a[wrong_by_date & before] - gap[wrong_by_date & before]
    s[wrong_by_date & ~before] = b[wrong_by_date & ~before] + gap[wrong_by_date & ~before]
    c = m.copy()
    c[wrong_by_product] = (m[wrong_by_product] % 4) + 1
    df = pd.DataFrame(dict(member_id=member+member_offset, truth=y.astype(int), service=s,
        start=a, end=b, member_product=m, claim_product=c,
        amount=r.lognormal(np.where(y, 5.05, 5.25), .75),
        network=(r.random(n) < np.where(y, .76, .57)).astype(float)))
    for field in FIELDS:
        if mode == 'mixed':
            pmiss = corruption * (.55 if field in ('service','start','end') else .40)
            pmut = corruption * (.13 if field in ('service','start','end') else .11)
        elif mode == 'missing':
            pmiss, pmut = corruption * .60, 0.
        else:
            pmiss, pmut = 0., corruption * .60
        delete = r.random(n) < pmiss
        df.loc[delete, field] = np.nan
        mutation = (r.random(n) < pmut) & ~delete
        if field in ('service','start','end'):
            df.loc[mutation, field] += r.choice([-180., -70., 70., 180.], mutation.sum())
        else:
            df.loc[mutation, field] = (df.loc[mutation, field] % 4) + 1
    return df


def inputs(df: pd.DataFrame) -> pd.DataFrame:
    f = df[list(FIELDS) + ['amount','network']].copy()
    f['start_distance'] = f.service - f.start
    f['end_distance'] = f.end - f.service
    f['product_agreement'] = (f.member_product == f.claim_product).astype(float)
    f.loc[f[['member_product','claim_product']].isna().any(axis=1),'product_agreement'] = np.nan
    f['n_missing'] = f[list(FIELDS)].isna().sum(axis=1)
    f['chronology_conflict'] = (f.start > f.end).astype(float)
    f.loc[f[['start','end']].isna().any(axis=1),'chronology_conflict'] = np.nan
    return f[list(FEATURES)].astype(float)


def evidence(df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    """Three-valued knowledge rules; observed contradiction is negative witness.

    Missing fields alone never imply noncoverage. A complete positive proof needs
    observed chronology, service inside interval, and product agreement.
    """
    n=len(df)
    has_dates = df[['service','start','end']].notna().all(axis=1).to_numpy()
    has_products = df[['member_product','claim_product']].notna().all(axis=1).to_numpy()
    has_span = df[['start','end']].notna().all(axis=1).to_numpy()
    bad_span = has_span & (df.start.to_numpy()>df.end.to_numpy())
    bad_product = has_products & (df.member_product.to_numpy()!=df.claim_product.to_numpy())
    s,a,b = (df[k].to_numpy() for k in ('service','start','end'))
    bad_service=has_dates & ((s<a)|(s>b))
    contradiction=bad_span|bad_product|bad_service
    proof = has_dates & has_products & ~contradiction
    pred=np.where(contradiction,0,np.where(proof,1,-1)).astype(int)
    reasons=np.full(n,'INSUFFICIENT_EVIDENCE',dtype=object)
    reasons[proof]='COMPLETE_SUPPORTING_EVIDENCE'
    reasons[bad_service]='SERVICE_DATE_CONTRADICTION'
    reasons[bad_product]='PRODUCT_CONTRADICTION'
    reasons[bad_span]='INVALID_ENROLLMENT_INTERVAL'
    missing=np.where(pred==-1)[0]
    for ix in missing:
        absent=','.join(k for k in FIELDS if pd.isna(df.iloc[ix][k]))
        reasons[ix]='INSUFFICIENT_EVIDENCE:'+absent
    return pred,reasons.tolist()


def decide(p: np.ndarray, known: np.ndarray, threshold: float,
           rules: bool, gate: bool) -> np.ndarray:
    ml=(p>=.5).astype(int)
    if not rules:
        return np.where(gate & (np.maximum(p,1-p)<threshold),-1,ml).astype(int)
    if not gate:
        return np.where(known>=0,known,ml).astype(int)
    certificate_positive=known==1
    corroborated_negative=(known==0)&(ml==0)
    fallback=(known==-1)&(np.maximum(p,1-p)>=threshold)
    return np.where(certificate_positive,1,
         np.where(corroborated_negative,0,np.where(fallback,ml,-1))).astype(int)


def strategy(df: pd.DataFrame, model, threshold: float, rules: bool,
             gate: bool) -> np.ndarray:
    p=model.predict_proba(inputs(df))[:,1]
    known,_=evidence(df)
    return decide(p,known,threshold,rules,gate)


def stats_for(y: np.ndarray, p: np.ndarray) -> dict[str,float]:
    keep = p!=-1
    err = keep&(p!=y)
    return dict(automation=float(keep.mean()), selective_error=float(err.sum()/max(keep.sum(),1)),
        incorrect_per_1000=float(err.mean()*1000),
        false_negative_per_1000=float(((y==1)&(p==0)).mean()*1000),
        review_per_1000=float((~keep).mean()*1000))


def tune_threshold(df: pd.DataFrame, model, y: np.ndarray,
                   rules: bool, target_risk=.03) -> float:
    """Choose maximum empirical validation automation under risk cap.
    Does not certify a future or shifted-distribution guarantee.
    """
    p=model.predict_proba(inputs(df))[:,1]
    known,_=evidence(df)
    candidates=np.linspace(.5,1.,101).tolist()+[1.01]
    good=[]
    for t in candidates:
        pred=decide(p,known,t,rules,True)
        s=stats_for(y,pred)
        if s['selective_error']<=target_risk:
            good.append((s['automation'],-t,t))
    return max(good)[2] if good else 1.01


def tune_coverage_threshold(df: pd.DataFrame, model, reference_automation: float) -> float:
    """Match a comparator's validation automation without inspecting test labels."""
    p = model.predict_proba(inputs(df))[:, 1]
    confidences = np.maximum(p, 1. - p)
    candidates = np.unique(np.r_[.5, confidences, np.nextafter(confidences, np.inf), 1.01])
    rates = (confidences[:, None] >= candidates[None, :]).mean(axis=0)
    delta = np.abs(rates - reference_automation)
    return float(candidates[np.flatnonzero(delta == delta.min())[0]])


def explain(df: pd.DataFrame, model, tau: float) -> pd.DataFrame:
    r, reason = evidence(df)
    p = model.predict_proba(inputs(df))[:,1]
    decision = strategy(df, model, tau, True, True)
    route = np.where(decision==-1,'REVIEW',np.where(r!=-1,'RULE_WITNESS','CONFIDENT_ML_FALLBACK'))
    route = np.where((decision==-1)&(r!=-1),'MODEL_RULE_CONFLICT',route)
    return pd.DataFrame(dict(decision=decision, route=route, reason=reason,
                              predicted_support_probability=np.round(p,5),
                              missing_core_fields=df[list(FIELDS)].isna().sum(axis=1)))


def run(seeds=SEEDS, n_members=2200, target_risk=.03) -> pd.DataFrame:
    rows=[]
    for seed in seeds:
        train = simulate(n_members,seed*1000+101,.12,member_offset=0)
        val = simulate(max(700,n_members//3),seed*1000+202,.12,member_offset=1000000)
        clf=HistGradientBoostingClassifier(max_iter=100,max_leaf_nodes=15,
                learning_rate=.07,l2_regularization=2,random_state=seed)
        clf.fit(inputs(train),train.truth)
        t_on=tune_threshold(val,clf,val.truth.to_numpy(),True,target_risk)
        t_off=tune_threshold(val,clf,val.truth.to_numpy(),False,target_risk)
        qa_val = strategy(val,clf,t_on,True,True)
        t_cover=tune_coverage_threshold(val,clf,float(np.mean(qa_val!=-1)))
        for corruption in LEVELS:
            for mode in ('mixed',) if corruption!=.2 else ('mixed','missing','silent'):
                test=simulate(max(800,n_members//2),seed*1000+303,corruption,mode,member_offset=2000000)
                y=test.truth.to_numpy()
                proba=clf.predict_proba(inputs(test))[:,1]
                rule_values,_=evidence(test)
                variants={'ML':decide(proba,rule_values,.5,False,False),
                          'Rules+ML':decide(proba,rule_values,.5,True,False),
                          'Selective-ML':decide(proba,rule_values,t_off,False,True),
                          'ML|val-coverage':decide(proba,rule_values,t_cover,False,True),
                          'Quality-aware':decide(proba,rule_values,t_on,True,True)}
                variants['ML|same-mask']=np.where(variants['Quality-aware']==-1,-1,variants['ML'])
                variants['Rules-only']=rule_values
                for method,p in variants.items():
                    rows.append(dict(seed=seed,mode=mode,corruption=corruption,method=method,
                              threshold=t_on if method=='Quality-aware' else t_off if method=='Selective-ML' else np.nan,
                              n_test=len(test),**stats_for(y,p)))
    return pd.DataFrame(rows)


def paired_interval(x: np.ndarray) -> tuple[float,float,float]:
    x=np.asarray(x,dtype=float)
    mean=float(np.mean(x))
    if len(x)<2:return mean,np.nan,np.nan
    e=stats.t.ppf(.975,len(x)-1)*stats.sem(x)
    return mean,float(mean-e),float(mean+e)


def tables(rows: pd.DataFrame) -> tuple[pd.DataFrame,pd.DataFrame]:
    base=rows[(rows.corruption==.2)&(rows['mode']=='mixed')]
    t1=base.groupby('method',sort=False).agg(automatic_pct=('automation','mean'),
          selective_error_pct=('selective_error','mean'),
          incorrect_per_1000=('incorrect_per_1000','mean'),
          false_negative_per_1000=('false_negative_per_1000','mean'),
          review_per_1000=('review_per_1000','mean')).round(3).reset_index()
    sub=rows[(rows.corruption==.2)&(rows['mode'].isin(['mixed','missing','silent'])) &
             (rows.method.isin(['ML','Quality-aware','ML|same-mask']))]
    t2=sub.groupby(['mode','method'],sort=False).agg(automatic_pct=('automation','mean'),
          selective_error_pct=('selective_error','mean'),
          incorrect_per_1000=('incorrect_per_1000','mean'),
          false_negative_per_1000=('false_negative_per_1000','mean')).round(3).reset_index()
    for tbl in (t1,t2):
        for field in ('automatic_pct','selective_error_pct'):
            tbl[field]*=100
    return t1,t2


def fig1(path: Path):
    fig,ax=plt.subplots(figsize=(10.2,4.8))
    fig.patch.set_facecolor('white');ax.set_xlim(0,12);ax.set_ylim(0,6);ax.axis('off')
    def block(x,y,w,h,title,subtitle,color='#f1f5f9'):
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.08,rounding_size=.10',
                         facecolor=color,edgecolor='#1e3a5f',linewidth=1.4))
        ax.text(x+w/2,y+h*.64,title,ha='center',va='center',fontsize=9.6,weight='bold',color='#173655')
        ax.text(x+w/2,y+h*.30,subtitle,ha='center',va='center',fontsize=8.3,color='#41556d')
    def arrow(a,b):ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=14,
                                               linewidth=1.35,color='#40648a'))
    block(.25,2.50,2.4,1.2,'CLAIM EVIDENCE','Service dates • plans')
    block(3.1,2.50,2.65,1.2,'QUALITY CHECKS','Completeness • validity')
    block(6.22,4.20,2.52,1.10,'SYMBOLIC RULES','Proof • conflict • unknown','#e8f2fd')
    block(6.22,1.02,2.52,1.10,'PREDICTIVE MODEL','Boosting • confidence','#e8f2fd')
    block(9.38,2.5,2.28,1.2,'EVIDENCE FUSION','Rules • confidence','#dff2ed')
    block(9.38,.2,2.28,1.04,'DECISION TRACE','Outcome • reason','#f5f0e7')
    for a,b in [((2.72,3.1),(3.02,3.1)),((5.82,3.43),(6.15,4.73)),((5.82,2.77),(6.15,1.57)),
                ((8.81,4.78),(9.31,3.46)),((8.81,1.57),(9.31,2.78)),((10.52,2.42),(10.52,1.32))]:arrow(a,b)
    fig.savefig(path.with_suffix('.png'),dpi=320,bbox_inches='tight',facecolor='white')
    fig.savefig(path.with_suffix('.pdf'),bbox_inches='tight',facecolor='white');plt.close(fig)


def fig2(rows: pd.DataFrame, path: Path):
    df=rows[rows['mode']=='mixed'];colors={'ML':'#64748b','Rules+ML':'#ef8b2c','Selective-ML':'#16a085',
             'Quality-aware':'#215ab0'}
    fig,axs=plt.subplots(1,2,figsize=(10,4.2),sharex=True)
    for method in colors:
        d=df[df.method==method]
        grp=d.groupby('corruption')
        x=[];error=[];er_lo=[];er_hi=[];cov=[];cov_lo=[];cov_hi=[]
        for rate,g in grp:
            x.append(rate*100)
            m,l,h=paired_interval(g.incorrect_per_1000.to_numpy());error.append(m);er_lo.append(l);er_hi.append(h)
            m,l,h=paired_interval(g.automation.to_numpy()*100);cov.append(m);cov_lo.append(l);cov_hi.append(h)
        axs[0].plot(x,error,marker='o',linewidth=2,label=method,color=colors[method])
        axs[0].fill_between(x,er_lo,er_hi,alpha=.10,color=colors[method])
        axs[1].plot(x,cov,marker='o',linewidth=2,label=method,color=colors[method])
        axs[1].fill_between(x,cov_lo,cov_hi,alpha=.10,color=colors[method])
    axs[0].set(title='A  |  Incorrect definitive decisions',ylabel='Errors per 1,000 claims')
    axs[1].set(title='B  |  Automated decision coverage',ylabel='Claims automatically decided (%)')
    for ax in axs:
        ax.set_xlabel('Synthetic field-corruption setting (%)');ax.grid(alpha=.18);ax.spines[['top','right']].set_visible(False)
        ax.set_xlim(-1,31)
    axs[0].set_ylim(bottom=0);axs[1].set_ylim(0,105)
    hh,ll=axs[0].get_legend_handles_labels()
    fig.legend(hh,ll,loc='lower center',bbox_to_anchor=(.5,.015),ncol=4,frameon=False,fontsize=9)
    fig.tight_layout(rect=[0,.12,1,1]);fig.savefig(path.with_suffix('.png'),dpi=320,bbox_inches='tight')
    fig.savefig(path.with_suffix('.pdf'),bbox_inches='tight');plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',default='results');p.add_argument('--members',type=int,default=2200)
    p.add_argument('--seeds',type=int,nargs='+',default=SEEDS);p.add_argument('--target-risk',type=float,default=.03)
    a=p.parse_args()
    if a.members<300 or not 0<a.target_risk<.5: p.error('Need members >=300 and target-risk in (0,.5)')
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    rows=run(a.seeds,a.members,a.target_risk);t1,t2=tables(rows)
    rows.to_csv(out/'metrics.csv',index=False);t1.to_csv(out/'table1.csv');t2.to_csv(out/'table2.csv')
    fig1(out/'figure1');fig2(rows,out/'figure2')
    import sklearn, scipy, matplotlib
    (out/'manifest.json').write_text(json.dumps(dict(seeds=a.seeds,members_train=a.members,
         validation_claims_per_seed=max(700,a.members//3)*3,
         test_claims_per_seed=max(800,a.members//2)*3,levels=LEVELS,threshold_target=a.target_risk,
         corruption={'mixed_missing_dates':.55,'mixed_missing_products':.40,
           'mixed_mutation_dates':.13,'mixed_mutation_products':.11,
           'stress_missing_only':.60,'stress_mutation_only':.60},
         versions={'python':platform.python_version(),'numpy':np.__version__,
           'pandas':pd.__version__,'scipy':scipy.__version__,
           'sklearn':sklearn.__version__,'matplotlib':matplotlib.__version__},
         dataset='synthetic only, illustrative non-operational rules',
         tuning='empirical validation threshold without formal guarantee'),indent=2))
    print('TABLE 1\n'+t1.to_string(index=False));print('\nTABLE 2\n'+t2.to_string(index=False))
    print('\nFiles written to',out.resolve())

if __name__=='__main__':main()
