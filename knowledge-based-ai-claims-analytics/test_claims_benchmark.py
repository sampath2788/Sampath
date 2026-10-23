import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from claims_benchmark import evidence,simulate,inputs,stats_for,strategy,explain,tune_threshold

def test_deterministic_and_independent():
    pd.testing.assert_frame_equal(simulate(50,7,.2),simulate(50,7,.2))
    assert len(simulate(50,7,.2))==150
    assert set(simulate(50,7,.2).truth.unique())=={0,1}

def test_three_way_logic():
    df=pd.DataFrame([dict(service=20.,start=10.,end=30.,member_product=1.,claim_product=1.),
      dict(service=20.,start=np.nan,end=30.,member_product=1.,claim_product=1.),
      dict(service=20.,start=np.nan,end=30.,member_product=1.,claim_product=2.),
      dict(service=50.,start=10.,end=30.,member_product=1.,claim_product=1.),
      dict(service=20.,start=30.,end=10.,member_product=1.,claim_product=1.)])
    pred,reasons=evidence(df)
    assert pred.tolist()==[1,-1,0,0,0]
    assert reasons[2]=='PRODUCT_CONTRADICTION'
    assert reasons[1].startswith('INSUFFICIENT_EVIDENCE')

def test_no_label_leakage():
    z=simulate(40,19,.2)
    assert not {'member_id','truth'} & set(inputs(z).columns)

def test_data_corruption_modes():
    x=simulate(90,8,.3,'missing')
    y=simulate(90,8,.3,'silent')
    assert x[list(('service','start','end','member_product','claim_product'))].isna().any().any()
    assert not y[list(('service','start','end','member_product','claim_product'))].isna().any().any()

def test_deferred_not_counted_as_negative():
    y=np.array([1,0,1,0]);p=np.array([1,-1,-1,0]);s=stats_for(y,p)
    assert s['automation']==.5 and s['incorrect_per_1000']==0

def test_explain_and_gate():
    data=simulate(150,42,.1)
    model=DummyClassifier(strategy='prior').fit(inputs(data),data.truth)
    tau=tune_threshold(data,model,data.truth.to_numpy(),True)
    exp=explain(data.head(20),model,tau)
    assert len(exp)==20 and exp.reason.str.len().min()>4
    assert set(exp.route)<= {'REVIEW','RULE_WITNESS','CONFIDENT_ML_FALLBACK','MODEL_RULE_CONFLICT'}
    for knowledge in (True,False):
        for gate in (True,False):
            y=strategy(data,model,tau,knowledge,gate)
            assert set(y)<= {-1,0,1}
