"""Post-hoc one-session return IC; not a matched replication of KiT."""
import json
from pathlib import Path
import numpy as np
import pandas as pd

RUN = Path('/var/tmp/work-mshanthappa/finstak_recovered/fresh_run_20261113T005202595991Z')


def correlation(a, b):
    a = a - a.mean(axis=1, keepdims=True)
    b = b - b.mean(axis=1, keepdims=True)
    denominator = np.sqrt((a*a).sum(1) * (b*b).sum(1))
    if not (denominator > 0).all():
        raise ValueError('A date has an undefined correlation; do not silently replace it with zero.')
    return (a*b).sum(1) / denominator


def main():
    models = {}
    for stage, variant in [('data_stage','no_events'), ('data_stage','no_decay'), ('architecture_stage','hybrid_decay')]:
        all_ic, all_rank, annual = [], [], {}
        for year in ('2021','2022','2023'):
            with np.load(RUN / stage / year / f'{variant}_predictions.npz') as p:
                dates, counts = np.unique(p['dates'], return_counts=True)
                assert (counts == 100).all()
                assert np.array_equal(p['dates'], np.repeat(dates,100))
                assert len(np.unique(p['companies'][:100])) == 100
                assert all(np.array_equal(p['companies'][i:i+100],p['companies'][:100])
                           for i in range(0,len(p['companies']),100))
                predicted = np.log(p['prediction'][:,3] / p['anchor']).reshape(-1,100)
                observed = np.log(p['truth'][:,3] / p['anchor']).reshape(-1,100)
            ic = correlation(predicted, observed)
            rank = correlation(pd.DataFrame(predicted).rank(axis=1,method='average').to_numpy(),
                               pd.DataFrame(observed).rank(axis=1,method='average').to_numpy())
            annual[year] = {'dates':len(dates),'return_IC':float(ic.mean()),'return_RankIC':float(rank.mean())}
            all_ic.extend(ic); all_rank.extend(rank)
        models[variant] = {'return_IC':float(np.mean(all_ic)), 'return_RankIC':float(np.mean(all_rank)),
                           'dates':len(all_ic), 'annual':annual}
    record = {'definition':'Per-date cross-sectional Pearson and Spearman correlation between predicted and realized next-session close-to-close log returns; arithmetic mean over 749 dates, 100 stocks per date.',
              'horizon':'one trading session','test_years':'2021-2023','analysis_status':'Post-hoc diagnostic; no new training; no significance claim.',
              'comparison_warning':'Not matched to KiT daily resolution, which uses a 20-session forecast horizon and a different market universe and test period.',
              'models':models}
    path = Path(__file__).with_name('finstak_return_ic_comparison.json')
    path.write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps({name:{k:v for k,v in row.items() if k!='annual'} for name,row in models.items()}))


if __name__ == '__main__':
    main()
