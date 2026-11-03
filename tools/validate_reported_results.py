#!/usr/bin/env python3
"""Validate manuscript-summary CSVs only. SPDX-License-Identifier: AGPL-3.0-only.

This newly written utility does not reproduce detector inference or AP from boxes.
"""
import csv
import json
from decimal import Decimal
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def read(name):
    with (ROOT/'data'/'reported_results'/name).open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))

def main():
    checks=[]
    def check(name, ok, detail):
        checks.append({'check':name,'passed':bool(ok),'detail':detail})
    files=list((ROOT/'data'/'reported_results').glob('*.csv'))
    records=[r for p in files for r in read(p.name)]
    check('summary CSV inventory',len(files)==36 and len(records)==265,{'files':len(files),'records':len(records)})
    check('source-labelled summaries',all(r.get('record_type')=='reported_manuscript_summary' and r.get('source_table') and r.get('source_file') for r in records),'No original per-image observations are implied.')
    native={r['checkpoint']:r for r in read('main_native_performance.csv')}
    check('main AP50_95 difference',Decimal(native['W-StageB']['AP50_95'])-Decimal(native['B-main']['AP50_95'])==Decimal('.0110'),'B-main .2647; W-StageB .2757')
    hierarchy=read('hierarchy_overall.csv')
    ap=sorted(Decimal(r['AP50_95']) for r in hierarchy)
    check('hierarchy AP50_95 values',ap==[Decimal('.2639'),Decimal('.2747')],ap and '.2639 to .2747')
    external=read('external_native_performance.csv');appendix=read('appendix_external_native_performance.csv')
    metrics=['precision','recall','AP50','AP50_95']
    def signature(rows):return sorted((r.get('model'),tuple(r[k] for k in metrics)) for r in rows)
    check('external main/appendix equality',signature(external)==signature(appendix),'Four author-confirmed rows preserved.')
    check('registered hash formats',all(len(r['sha256'])==64 and all(c in '0123456789abcdef' for c in r['sha256']) for r in read('checkpoint_identities.csv')),'These are recorded identities, not local payload rehashes.')
    interventions=read('interventions_all_thresholds.csv')
    keys={(r['removal_type'],r['level'],r['matching_IoU']) for r in interventions}
    check('intervention identity grid',len(interventions)==60 and len(keys)==60,'Six interventions by ten matching thresholds.')
    check('reentry partition',all(int(r['reentered'])==int(r['unmatched'])+int(r['replacement'])+int(r['new_match']) for r in interventions),'Checked directly on every supplied intervention row.')
    check('gained minus lost balance',all(int(r['delta_matched'])==int(r['gained_GT'])-int(r['lost_GT']) for r in interventions),'Checked directly on every supplied intervention row.')
    composition=read('dataset_composition.csv')
    totals={r['split'].lower():r for r in composition}
    check('dataset totals',len(composition)==4 and all(sum(int(r[k]) for r in composition if r['split'].lower()!='total')==int(totals['total'][k]) for k in ['images','annotations','background_images']),'Counts recomputed from the CSV.')
    scale=read('complete_detector_scale.csv')
    check('scale GT coverage',all(sum(int(r['GT_count']) for r in scale if r['checkpoint']==model)==2301 for model in ['B-post','W-StageB']),'Each checkpoint covers the same 2,301 annotations.')
    check('metric bounds',all(Decimal('0')<=Decimal(r[k])<=Decimal('1') for r in records for k in metrics if r.get(k) not in (None,'','--')),'Checks P, R and AP columns without conflating percentage-point differences or interval bounds.')
    ok=all(c['passed'] for c in checks)
    print(json.dumps({'summary_validation_passed':ok,'checks':checks,'original_inference_or_metric_reproduction_performed':False},ensure_ascii=False,indent=2))
    return 0 if ok else 2

if __name__=='__main__':
    raise SystemExit(main())
