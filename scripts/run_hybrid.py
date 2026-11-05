"""Train-only 계약이 있는 두 Validation CSV 결합."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import pandas as pd
from src.hybrid import align_predictions,tune_weight


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--item-dir',type=Path,required=True)
    p.add_argument('--content-dir',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    args=p.parse_args()
    frames=[];contracts=[]
    for directory in [args.item_dir,args.content_dir]:
        path=directory/'hybrid_ready_validation_predictions.csv'
        contract=json.loads((directory/'hybrid_contract.json').read_text(encoding='utf-8'))
        with path.open('rb') as f:digest=hashlib.file_digest(f,'sha256').hexdigest()
        if contract.get('csv_sha256')!=digest or contract.get('train_plus_validation_refit') is not False:
            raise ValueError('CSV 출처 또는 Train-only 계약 불일치')
        if 'train' not in str(contract.get('fit_data','')).lower():
            raise ValueError('Train-only 학습 근거 필요')
        frames.append(pd.read_csv(path));contracts.append(contract)
    for key in ['train_triple_sha256','validation_triple_sha256']:
        if not contracts[0].get(key) or contracts[0][key]!=contracts[1].get(key):
            raise ValueError('공통 분할 계약 불일치: '+key)
    paired=align_predictions(*frames)
    alpha,grid=tune_weight(paired.rating_item,paired.predicted_rating_item,paired.predicted_rating_content)
    args.output_dir.mkdir(parents=True,exist_ok=False)
    grid.to_csv(args.output_dir/'validation_grid.csv',index=False)
    paired['hybrid_prediction']=alpha*paired.predicted_rating_item+(1-alpha)*paired.predicted_rating_content
    paired.to_csv(args.output_dir/'validation_predictions.csv',index=False)
    (args.output_dir/'selection.json').write_text(json.dumps({'alpha':alpha,'contracts':contracts},indent=2),encoding='utf-8')
    print('alpha=',alpha)


if __name__=='__main__':main()
