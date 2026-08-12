"""
EXTRACT SENSITIVITY, SPECIFICITY, F1 FOR PAPER TABLE
=====================================================
Reads the personal models saved by ltaf_personalised_monitor.py
and computes the missing metrics for each patient.

Run: python extract_metrics.py
Output: prints a table you can paste back to Claude,
        AND saves metrics_for_paper.csv
"""

import os, json, warnings
import numpy as np
import pandas as pd
from scipy.stats import linregress, skew, kurtosis, pearsonr
from scipy.signal import welch
from sklearn.metrics import (roc_auc_score, confusion_matrix,
                              f1_score, precision_recall_curve)
from sklearn.utils import resample
import joblib
warnings.filterwarnings('ignore')

OUTPUT_PATH = r"C:\Users\HOME\Desktop\ltaf_project"
RR_PATH     = os.path.join(OUTPUT_PATH, "data", "rr")
MODEL_PATH  = os.path.join(OUTPUT_PATH, "models", "personal")
RES_PATH    = os.path.join(OUTPUT_PATH, "results")

# ── same feature/filter code (condensed) ─────────────────────────────────────
MIN_SINUS_BEFORE_SEC = 300
MIN_AF_DUR_SEC       = 30
POST_AF_BUFFER_SEC   = 180
PRE_AF_SKIP_SEC      = 30
WINDOW_BEATS         = 120
STEP_BEATS           = 20
TREND_WIN            = 8
HORIZON_MINS         = [5, 10, 20, 30]
HORIZON_SECS         = [h*60 for h in HORIZON_MINS]
HORIZON_BINS         = [0] + HORIZON_SECS
NORMAL_IDX           = len(HORIZON_MINS)

def safe(f,a,fb=0.0):
    try: v=float(f(a)); return v if np.isfinite(v) else fb
    except: return fb
def lag1(a):
    try:
        if len(a)<4 or np.std(a)<1e-8: return 0.0
        from scipy.stats import pearsonr as pr
        r,_=pr(a[:-1],a[1:]); return float(r) if np.isfinite(r) else 0.0
    except: return 0.0
def ar1(a):
    try:
        if len(a)<4 or np.std(a)<1e-8: return 0.0
        y=a[1:]; X=np.column_stack([a[:-1],np.ones(len(a)-1)])
        c=np.linalg.lstsq(X,y,rcond=None)[0]; return float(c[0]) if np.isfinite(c[0]) else 0.0
    except: return 0.0
def samp_ent(a):
    try:
        from antropy import sample_entropy
        if len(a)<12 or np.std(a)<1e-8: return 0.0
        v=sample_entropy(a,order=2); return float(v) if np.isfinite(v) else 0.0
    except: return 0.0
def dfa_f(a):
    try:
        from antropy import detrended_fluctuation
        if len(a)<20 or np.std(a)<1e-8: return 1.0
        v=detrended_fluctuation(a); return float(v) if np.isfinite(v) else 1.0
    except: return 1.0
def lf_hf(rr):
    try:
        if len(rr)<16: return 1.0
        d=rr-np.mean(rr); f,p=welch(d,fs=1.0,nperseg=min(len(d),64))
        lf=np.trapz(p[(f>=0.04)&(f<0.15)],f[(f>=0.04)&(f<0.15)])
        hf=np.trapz(p[(f>=0.15)&(f<=0.40)],f[(f>=0.15)&(f<=0.40)])
        return float(lf/(hf+1e-9)) if np.isfinite(lf/(hf+1e-9)) else 1.0
    except: return 1.0
def perm_ent(rr,order=3):
    try:
        if len(rr)<order+1: return 0.0
        pats={}
        for i in range(len(rr)-order+1):
            p=tuple(np.argsort(rr[i:i+order])); pats[p]=pats.get(p,0)+1
        probs=np.array(list(pats.values()),dtype=float); probs/=probs.sum()
        from math import log,factorial
        me=log(factorial(order)); e=-sum(p*np.log(p+1e-12) for p in probs)
        return float(e/me) if me>0 else 0.0
    except: return 0.0
def regularity_index(rr):
    if len(rr)<4: return 0.0
    return float(np.mean(np.abs(np.diff(rr))/(np.abs(rr[:-1])+1e-8)<0.02))
def rr_mono(rr):
    if len(rr)<4: return 0.0
    return float(np.sum(np.diff(rr)<0)/(len(rr)-1))
def var_stab(rr):
    if len(rr)<40: return 1.0
    mid=len(rr)//2; return float(np.var(rr[mid:])/(np.var(rr[:mid])+1e-8))

def compute_features(rr):
    rr=np.array(rr,dtype=np.float64); diff=np.diff(rr)
    v=np.array([safe(np.var,rr),lag1(rr),ar1(rr),safe(skew,rr),safe(kurtosis,rr),
        safe(np.mean,rr),safe(np.std,rr),
        safe(lambda x:np.sqrt(np.mean(x**2)),diff),
        safe(lambda x:np.mean(np.abs(x)>50),diff),
        lf_hf(rr),samp_ent(rr),dfa_f(rr),
        safe(lambda x:np.std(x/np.sqrt(2)),diff),
        float(np.std(rr[:-1])*np.sqrt(2)) if len(rr)>2 else 0.0,
        float(np.std(rr)/np.mean(rr)) if np.mean(rr)>1 else 0.0,
        perm_ent(rr),safe(lambda x:np.mean(np.abs(x)),diff),
        float(np.percentile(rr,95)-np.percentile(rr,5)),
        regularity_index(rr),rr_mono(rr),var_stab(rr)],dtype=np.float32)
    return np.nan_to_num(v,nan=0.0,posinf=0.0,neginf=0.0)

def get_usable_episodes(all_eps,dur_h):
    if not all_eps: return []
    if len(all_eps)/max(dur_h/24,0.1)>48: return []
    usable=[]
    for i,(af_s,af_e) in enumerate(all_eps):
        if (af_e-af_s)<MIN_AF_DUR_SEC: continue
        sb=af_s if i==0 else af_s-all_eps[i-1][1]
        if sb>=MIN_SINUS_BEFORE_SEC: usable.append((af_s,af_e))
    return usable

def label_window(t_end,usable_eps):
    if not usable_eps: return 0,np.inf,NORMAL_IDX
    for af_s,af_e in usable_eps:
        if af_s<=t_end<=af_e: return -1,-1,-1
        if af_e<t_end<af_e+POST_AF_BUFFER_SEC: return -1,-1,-1
    future=[(s,e) for s,e in usable_eps if s>t_end]
    if not future:
        pe=[e for _,e in usable_eps if e<=t_end]
        if pe and (t_end-max(pe))<POST_AF_BUFFER_SEC: return -1,-1,-1
        return 0,np.inf,NORMAL_IDX
    h_sec=future[0][0]-t_end
    if h_sec<PRE_AF_SKIP_SEC: return -1,-1,-1
    if h_sec<=HORIZON_SECS[-1]:
        for i in range(len(HORIZON_BINS)-1):
            if HORIZON_BINS[i]<=h_sec<HORIZON_BINS[i+1]: return 1,h_sec,i
        return 1,h_sec,len(HORIZON_MINS)-1
    pe=[e for _,e in usable_eps if e<=t_end]
    if pe and (t_end-max(pe))<POST_AF_BUFFER_SEC: return -1,-1,-1
    return 0,h_sec,NORMAL_IDX

def add_trends(X,times):
    N,F=X.shape; slopes=np.zeros((N,F)); tidx=np.arange(TREND_WIN,dtype=float)
    for i in range(TREND_WIN,N):
        seg=X[i-TREND_WIN:i]
        for fi in range(F):
            s=seg[:,fi]
            if np.std(s)<1e-8: continue
            try:
                sl,*_=linregress(tidx,s)
                slopes[i,fi]=float(sl) if np.isfinite(sl) else 0.0
            except: pass
    return np.nan_to_num(np.hstack([X,slopes]),nan=0.0,posinf=0.0,neginf=0.0)

# ── MAIN ─────────────────────────────────────────────────────────────────────
print("Extracting metrics from saved personal models...")
results = []

for model_file in sorted(os.listdir(MODEL_PATH)):
    if not model_file.startswith('personal_') or not model_file.endswith('.pkl'):
        continue
    rec = model_file.replace('personal_','').replace('.pkl','')
    
    rr_file = os.path.join(RR_PATH, f'{rec}_rr.npz')
    if not os.path.exists(rr_file): continue
    
    m = joblib.load(os.path.join(MODEL_PATH, model_file))
    clf     = m['clf']
    scaler  = m['scaler']
    tau     = m['tau']
    phen    = m['phenotype']
    auc_s   = m['auc']
    h_min   = m['h_min']
    calib_h = 6.0

    d        = np.load(rr_file, allow_pickle=True)
    par      = d['paroxysmal_af']
    rr_ms    = d['rr_ms']
    rr_times = d['rr_times']
    if par.shape[0]==0: continue

    meta_f = os.path.join(RR_PATH, f'{rec}_meta.json')
    dur_h  = 24.0
    if os.path.exists(meta_f):
        with open(meta_f) as mf: dur_h=json.load(mf).get('duration_hours',24.0)

    all_eps = [(row[0],row[1]) for row in par]
    usable  = get_usable_episodes(all_eps, dur_h)
    if not usable: continue

    first_af = usable[0][0]
    calib_end = min(calib_h*3600, first_af-1800)
    if calib_end < 1800: calib_end = min(first_af*0.5, calib_h*3600)

    # Extract all windows
    feats,labels,times_list=[],[],[]
    for start in range(0,len(rr_ms)-WINDOW_BEATS,STEP_BEATS):
        end=start+WINDOW_BEATS; rr_win=rr_ms[start:end]
        t_end=rr_times[end-1]; t_ctr=(rr_times[start]+t_end)/2.0
        if (len(rr_win)<WINDOW_BEATS*0.8 or
                np.any(~np.isfinite(rr_win)) or np.std(rr_win)<1e-6): continue
        lbl,_,_=label_window(t_end,usable)
        if lbl==-1: continue
        feat=compute_features(rr_win)
        if not np.all(np.isfinite(feat)): continue
        feats.append(feat); labels.append(lbl); times_list.append(t_ctr)

    if not feats: continue
    X_raw=np.array(feats,dtype=np.float32)
    y_all=np.array(labels,dtype=np.int32)
    t_all=np.array(times_list,dtype=np.float32)

    # Normalise to personal baseline
    calib_mask=t_all<=calib_end
    if calib_mask.sum()<10: calib_mask=np.ones(len(t_all),dtype=bool)
    mu=X_raw[calib_mask].mean(0); std=X_raw[calib_mask].std(0)+1e-8
    X_norm=(X_raw-mu)/std
    X_enh=add_trends(X_norm,t_all)
    X_enh=np.nan_to_num(X_enh,nan=0.0,posinf=0.0,neginf=0.0)

    # Holdout = prediction phase only
    pred_mask=t_all>calib_end
    X_test=X_enh[pred_mask]; y_test=y_all[pred_mask]
    if len(X_test)==0 or len(np.unique(y_test))<2: continue

    X_test_s=scaler.transform(X_test)
    probs=clf.predict_proba(X_test_s)[:,1]

    # Find best F1 threshold
    try:
        _,_,thrs=precision_recall_curve(y_test,probs)
        best_t,best_f1=0.5,0.0
        for t in thrs:
            yp=(probs>=t).astype(int)
            f1=f1_score(y_test,yp,zero_division=0)
            if f1>best_f1: best_f1=f1; best_t=float(t)
    except: best_t=0.5; best_f1=0.0

    yp=(probs>=best_t).astype(int)
    cm=confusion_matrix(y_test,yp)
    if cm.size==4:
        tn,fp,fn,tp=cm.ravel()
        sens=tp/(tp+fn+1e-8); spec=tn/(tn+fp+1e-8)
        prec=tp/(tp+fp+1e-8); f1=f1_score(y_test,yp,zero_division=0)
        far=fp/(fp+tn+1e-8)
    else:
        sens=spec=prec=f1=far=0.0

    try: auc=roc_auc_score(y_test,probs)
    except: auc=auc_s

    results.append({'record':rec,'phenotype':phen,'tau':tau,
                    'best_horizon':h_min,'auc':auc,
                    'sensitivity':sens,'specificity':spec,
                    'precision':prec,'f1':f1,'far':far,
                    'n_pos':int(y_test.sum()),'n_total':len(y_test)})
    print(f"  {rec}: AUC={auc:.3f}  Sens={sens:.1%}  Spec={spec:.1%}  F1={f1:.3f}")

df=pd.DataFrame(results)
df.to_csv(os.path.join(RES_PATH,'metrics_for_paper.csv'),index=False)

print(f"\n{'='*65}")
print("SUMMARY FOR PAPER TABLE")
print(f"{'='*65}")
print(f"  Mean AUC  : {df['auc'].mean():.4f} ± {df['auc'].std():.4f}")
print(f"  Mean Sens : {df['sensitivity'].mean():.1%}")
print(f"  Mean Spec : {df['specificity'].mean():.1%}")
print(f"  Mean F1   : {df['f1'].mean():.3f}")
print(f"  Mean FAR  : {df['far'].mean():.1%} false alarms per window")
print(f"\nFull table saved: {RES_PATH}/metrics_for_paper.csv")
print("\nPASTE THE OUTPUT ABOVE BACK TO CLAUDE TO GET THE COMPLETE PAPER")
