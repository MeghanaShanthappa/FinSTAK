"""Recompute report facts directly from the expanded-study artifacts."""
from collections import Counter
from datetime import datetime, timezone
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import numpy as np

OUT = Path(__file__).resolve().parents[1]
OLD = Path('/var/tmp/work-mshanthappa/finstak_recovered/finstak-100x10-v1')
RUN = OLD.parent / 'fresh_run_20261113T005202595991Z'
sys.path.insert(0, str(OLD / 'source_snapshot'))
spec = importlib.util.spec_from_file_location('report_statistics', OLD / 'source_snapshot/scripts/report_scale_study.py')
stats = importlib.util.module_from_spec(spec)
spec.loader.exec_module(stats)
stats.ROOT = RUN


def read(path):
    return json.loads(path.read_text())


def digest(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def main():
    plans = {s: read(RUN / s / 'plan.json') for s in ('data_stage', 'architecture_stage')}
    models, parts, checks, checkpoint_checks = {}, {}, [], []
    summaries = {s: {y: read(RUN / s / y / 'summary.json') for y in ('2021', '2022', '2023')}
                 for s in plans}
    for stage, plan in plans.items():
        for variant in plan['variants']:
            annual, chunks = {}, []
            for year in ('2021', '2022', '2023'):
                name = f'{variant}_predictions.npz'
                with np.load(RUN / stage / year / name) as p, np.load(OLD / stage / year / name) as original:
                    chunk = {k: p[k].copy() for k in ('dates','companies','truth','anchor','prediction')}
                    equal = all(np.array_equal(p[k], original[k]) for k in p.files)
                    max_price_diff=float(np.max(np.abs(p['prediction']-original['prediction'])))
                    checks.append({'stage':stage,'year':year,'variant':variant,'all_prediction_arrays_identical':equal,
                                   'maximum_price_difference_from_original':max_price_diff})
                    loss = np.abs(chunk['prediction']-chunk['truth'])/chunk['anchor'][:,None]
                    reported = summaries[stage][year][variant]['price']
                    assert abs(loss.mean()*100-reported['mean_OHLC_MAE_percent_of_today_close']) < 1e-10
                    assert (chunk['prediction'][:,1] >= np.maximum(chunk['prediction'][:,0],chunk['prediction'][:,3])).all()
                    assert (chunk['prediction'][:,2] <= np.minimum(chunk['prediction'][:,0],chunk['prediction'][:,3])).all()
                    annual[year] = summaries[stage][year][variant]
                    chunks.append(chunk)
                for seed in plan['seeds']:
                    p = RUN / stage / year / f'{variant}_seed{seed}.pt'
                    h = read(RUN / stage / year / f'{variant}_seed{seed}_history.json')
                    current = digest(p)
                    assert current == h['checkpoint_sha256']
                    checkpoint_checks.append({'stage':stage,'year':year,'variant':variant,'seed':seed,
                                              'same_as_original':current == digest(OLD/stage/year/p.name),
                                              'seconds':h['elapsed_seconds']})
            parts[variant] = chunks
            total = sum(p['price']['samples'] for p in annual.values())
            weighted = lambda f: sum(f(p)*p['price']['samples'] for p in annual.values())/total
            models[variant] = {'stage':stage,'samples':total,
                'MAE_percent':weighted(lambda p:p['price']['mean_OHLC_MAE_percent_of_today_close']),
                'direction_percent':100*weighted(lambda p:p['price']['next_close_direction_accuracy']),
                'within_1_percent':weighted(lambda p:p['tolerance']['1%']['pooled_fields_percent']),
                'whole_candle_1_percent':weighted(lambda p:p['tolerance']['1%']['whole_candle_percent']),
                'annual':{y:{'MAE_percent':a['price']['mean_OHLC_MAE_percent_of_today_close'],
                             'samples':a['price']['samples']} for y,a in annual.items()}}
    comparisons = {}
    for reference, candidate in [('no_events','no_decay'),('no_decay','hybrid_decay'),('no_decay','combined'),
                                  ('volume_shared','volume_pcgrad')]:
        comparisons[candidate+'_vs_'+reference] = stats.paired_interval(parts[reference],parts[candidate])
    selected, baseline_parts, new_parts, overall_parts = {}, [], [], []
    for i,y in enumerate(('2021','2022','2023')):
        scores = {v:np.mean([read(RUN/s/y/f'{v}_seed{seed}_history.json')['best_validation_MAE']
                             for seed in p['seeds']]).item()
                  for s,p in plans.items() for v in p['variants']}
        baseline = min(plans['data_stage']['variants'],key=lambda v:(scores[v],v))
        new = min(plans['architecture_stage']['variants'],key=lambda v:(scores[v],v))
        overall = min(scores,key=lambda v:(scores[v],v))
        selected[y] = {'baseline':baseline,'new':new,'overall':overall}
        baseline_parts.append(parts[baseline][i]);new_parts.append(parts[new][i]);overall_parts.append(parts[overall][i])
    comparisons['selected_new_vs_selected_baseline'] = stats.paired_interval(baseline_parts,new_parts)
    comparisons['selected_overall_vs_selected_baseline'] = stats.paired_interval(baseline_parts,overall_parts)
    pooled = lambda a: float(np.concatenate([np.abs(p['prediction']-p['truth'])/p['anchor'][:,None] for p in a]).mean()*100)
    majority = sum(summaries['data_stage'][y]['previous_close']['direction_baselines']['training_majority_accuracy']*
                   summaries['data_stage'][y]['previous_close']['price']['samples'] for y in selected)/74900*100
    baselines = {}
    for v in ('previous_close','previous_OHLC'):
        baselines[v] = sum(summaries['data_stage'][y][v]['price']['mean_OHLC_MAE_percent_of_today_close']*
                           summaries['data_stage'][y][v]['price']['samples'] for y in selected)/74900
    volume = {}
    for v in ('volume_shared','volume_uncertainty','volume_pcgrad','dual_stream','combined'):
        volume[v]={}
        for correction in ('naive','gaussian','smearing'):
            volume[v][correction] = {k:sum(summaries['architecture_stage'][y][v]['volume'][correction][k]*
                                          summaries['architecture_stage'][y][v]['volume'][correction]['samples']
                                          for y in selected)/74900
                                    for k in ('MAPE_positive_volume_percent','MSE_shares_squared')}
    for v in ('previous_volume','trailing_log_volume_median'):
        volume[v] = {'baseline':{k:sum(summaries['data_stage'][y]['previous_close']['volume_baselines'][v][k]*
                                       summaries['data_stage'][y]['previous_close']['volume_baselines'][v]['samples']
                                       for y in selected)/74900
                                   for k in ('MAPE_positive_volume_percent','MSE_shares_squared')}}
    panel = np.load(OLD/'packed/panel.npz')
    event_arrays = np.load(OLD/'packed/event_arrays.npz')
    event_ptr,event_age=event_arrays['indptr'],event_arrays['age']
    counts = np.diff(event_ptr).reshape(len(panel['dates']),100)
    coverage = {}
    for y in selected:
        dates = np.unique(parts['no_decay'][list(selected).index(y)]['dates'])
        keep=np.isin(panel['dates'],dates);a=counts[keep]
        recent=np.zeros_like(counts)
        for row in np.flatnonzero(keep).tolist():
            for company in range(100):
                ix=row*100+company
                age=event_age[event_ptr[ix]:event_ptr[ix+1]]
                recent[row,company]=int((age<5).sum())
        coverage[y]={'stock_dates':int(a.size),'active_event_coverage_percent':float((a>0).mean()*100),
                     'event_under_five_sessions_percent':float((recent[keep]>0).mean()*100),
                     'mean_active_events_per_stock_date':float(a.mean())}
    confidence, sources, categories, year_counts, event_stocks, primary = Counter(),Counter(),Counter(),Counter(),Counter(),0
    with (OLD/'expanded/events.jsonl').open() as f:
        for line in f:
            e=json.loads(line);confidence[str(e['confidence'])]+=1;sources[e['source_type']]+=1
            categories[e['category']]+=1;year_counts[e['known_at'][:4]]+=1
            for c in e['companies']:event_stocks[c['company_id']]+=1
            primary+=bool(e['primary_discloser_verified'])
    config=read(OLD/'dataset/config.json')
    sector_of={c['company_id']:c['sector'] for c in config['companies']}
    sector={}
    for sector_name in sorted(set(sector_of.values())):
        a=[];b=[]
        for p,q in zip(parts['no_events'],parts['no_decay']):
            keep=np.array([sector_of[c]==sector_name for c in p['companies']])
            a.extend((np.abs(p['prediction'][keep]-p['truth'][keep])/p['anchor'][keep,None]).mean(1)*100)
            b.extend((np.abs(q['prediction'][keep]-q['truth'][keep])/q['anchor'][keep,None]).mean(1)*100)
        sector[sector_name]={'stocks':sum(v==sector_name for v in sector_of.values()),
                             'no_events_MAE':float(np.mean(a)),'no_decay_MAE':float(np.mean(b)),
                             'reduction_percentage_points':float(np.mean(a)-np.mean(b))}
    feature={}
    for i,f in enumerate(('open','high','low','close')):
        errors=np.concatenate([np.abs(p['prediction'][:,i]-p['truth'][:,i])/p['anchor'] for p in parts['no_decay']])
        feature[f]={'MAE_percent':float(errors.mean()*100),'within_1_percent':float((errors<=.01).mean()*100)}
    report={'generated_utc':datetime.now(timezone.utc).isoformat(),'models':models,'comparisons':comparisons,
            'selection':selected,'selected_baseline_MAE':pooled(baseline_parts),'selected_new_MAE':pooled(new_parts),
            'selected_overall_MAE':pooled(overall_parts),'majority_direction_percent':majority,
            'baselines':baselines,'volume':volume,'event_coverage':coverage,'sectors':sector,'per_price':feature,
            'event_field_audit':{'confidence_counts':confidence,'sources':sources,'categories':categories,
                                 'years':year_counts,'primary_discloser_true':primary,
                                 'stock_event_count_min':min(event_stocks.values()),'stock_event_count_max':max(event_stocks.values())},
            'reproduction':{'prediction_array_checks':len(checks),'all_arrays_identical':all(c['all_prediction_arrays_identical'] for c in checks),
                             'checkpoint_checks':len(checkpoint_checks),'all_checkpoint_bytes_identical':all(c['same_as_original'] for c in checkpoint_checks),
                             'fit_compute_minutes':sum(c['seconds'] for c in checkpoint_checks)/60,
                             'max_price_difference_from_original':max(c['maximum_price_difference_from_original'] for c in checks)},
            'prediction_checks':checks,'checkpoint_checks':checkpoint_checks}
    (OUT/'evidence/report_facts.json').write_text(json.dumps(report,indent=2)+'\n')
    with (OUT/'FinSTAK_Results.csv').open('w') as f:
        w=csv.writer(f);w.writerow(['Model','OHLC_MAE_percent','Direction_percent','Fields_within_1_percent','Test_stock_dates'])
        for v,m in sorted(models.items(),key=lambda p:p[1]['MAE_percent']):
            w.writerow([v,m['MAE_percent'],m['direction_percent'],m['within_1_percent'],m['samples']])
    print(json.dumps({k:report[k] for k in ['reproduction','comparisons','selection','majority_direction_percent','per_price','event_coverage','volume']},indent=2))


if __name__=='__main__':main()
