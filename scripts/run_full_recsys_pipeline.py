"""공통 Train/Validation/Test 학습·평가. 기존 결과 덮어쓰기 금지."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import importlib.metadata
from copy import copy
os.environ.setdefault('OPENBLAS_NUM_THREADS','4')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
import psutil
from src.item_based import ItemBasedCF
from src.content_based import ContentBasedCF
from src.common_item_data import validate_common_data
from src.transfer_data import load_splits, validate_splits, load_content, SharedContentCF, canonical_genre
from src.hybrid import metrics, align_predictions, tune_weight
from src.processed_validation import validate_processed, sha256, npz_schema
RUN_STAGES=[]
RUN_LOG_DIR=None


def record_stage(name,started):
    info=psutil.Process().memory_info()
    row={'stage':name,'seconds':time.perf_counter()-started,'rss_mib':info.rss/2**20,
         'peak_working_set_mib':getattr(info,'peak_wset',0)/2**20}
    RUN_STAGES.append(row)
    if RUN_LOG_DIR is not None:save(RUN_LOG_DIR/'stage_resources.json',RUN_STAGES)
    print(json.dumps(row),flush=True)


def fingerprint(frame):
    """dtype 차이에 독립적인 행 순서 포함 분할 해시."""
    digest=hashlib.sha256()
    for col,dtype in [('user_id','<i8'),('item_id','<i8'),('rating','<f8')]:
        digest.update(col.encode())
        for start in range(0,len(frame),100000):
            digest.update(frame[col].iloc[start:start+100000].to_numpy(dtype=dtype).tobytes())
    return digest.hexdigest()


def save(path, value):
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,default=lambda v: v.item() if isinstance(v,np.generic) else str(v)),encoding='utf-8')


def predict(model, pairs):
    """정답 없는 쌍을 배치 예측."""
    if 'rating' in pairs: raise ValueError('예측 입력에는 정답 제외')
    if isinstance(model,ItemBasedCF):
        n = len(pairs)
        scores = np.empty(n)
        methods = np.full(n,'global_mean',dtype=object)
        users = model.user_ids_.get_indexer(pairs.user_id)
        items = model.item_ids_.get_indexer(pairs.item_id)
        for start in range(0,n,8192):
            end = min(start+8192,n)
            u,i = users[start:end], items[start:end]
            base = pairs.user_id.iloc[start:end].map(model.user_means_).fillna(
                pairs.item_id.iloc[start:end].map(model.item_means_)).fillna(model.global_mean_).to_numpy(copy=True)
            meth = np.where(u>=0,'user_mean',np.where(i>=0,'item_mean','global_mean')).astype(object)
            ok = (u>=0)&(i>=0)
            if ok.any():
                neighbors = model.neighbor_indices_[i[ok]]
                weights = model.neighbor_similarities_[i[ok]]
                selected = model.user_item_[u[ok,None],neighbors]
                values = selected.toarray() if hasattr(selected,'toarray') else np.asarray(selected)
                effective = weights*((values>0)&(weights>1e-12))
                denom = effective.sum(1)
                hit = denom>0
                pos = np.flatnonzero(ok)[hit]
                base[pos] = (effective*values).sum(1)[hit]/denom[hit]
                meth[pos] = 'cf'
            scores[start:end] = np.clip(base,*model.rating_range_)
            methods[start:end] = meth
        return scores,methods
    scores,methods,_ = model.predict_many(pairs.user_id.to_numpy(),pairs.item_id.to_numpy())
    return scores,methods


def scored(model, frame):
    p,m = predict(model,frame[['user_id','item_id']])
    result = frame.copy()
    result['predicted_rating'],result['method'] = p,m
    return result


def experiment(parts, factory, out):
    out.mkdir(parents=True)
    train,val = parts['train'],parts['validation']
    candidates=[]
    # 최대 K에서 이웃을 한 번 계산하고 동일 순서의 앞 K개 재사용
    tick=time.perf_counter()
    item = ItemBasedCF(50).fit(train)
    record_stage('movie_item_fit',tick)
    tick=time.perf_counter()
    indices, similarities = item.neighbor_indices_,item.neighbor_similarities_
    best_item = None
    best_key = None
    for k in (20,30,50):
        item.neighbor_indices_,item.neighbor_similarities_ = indices[:,:k],similarities[:,:k]
        prediction=scored(item,val)
        row={'model':'Item-based','parameter':k,**metrics(val.rating,prediction.predicted_rating)}
        candidates.append(row)
        key=(row['mae'],row['rmse'],k)
        if best_key is None or key<best_key:
            best_key,best_k,best_item=key,k,prediction
    item.neighbor_indices_,item.neighbor_similarities_=indices[:,:best_k],similarities[:,:best_k]
    item.n_neighbors=best_k
    record_stage('movie_item_validation_grid',tick)
    best_key=None
    shared=factory(1.)
    if isinstance(shared,SharedContentCF):
        print('Content sparse profiles + sufficient statistics',flush=True)
        tick=time.perf_counter()
        shared.fit(train)
        record_stage('movie_content_profiles_and_sufficient_statistics',tick)
    tick=time.perf_counter()
    for a in (1.,10.,100.):
        content=copy(shared.set_alpha(a)) if isinstance(shared,SharedContentCF) else factory(a).fit(train)
        prediction=scored(content,val)
        row={'model':'Content-based','parameter':a,**metrics(val.rating,prediction.predicted_rating)}
        candidates.append(row)
        key=(row['mae'],row['rmse'],a)
        if best_key is None or key<best_key:
            best_key,best_content,best_content_pred=key,content,prediction
    record_stage('movie_content_validation_grid',tick)
    paired=align_predictions(best_item,best_content_pred)
    weight,grid=tune_weight(paired.rating_item,paired.predicted_rating_item,paired.predicted_rating_content)
    grid.to_csv(out/'hybrid_grid.csv',index=False)
    pd.DataFrame(candidates).to_csv(out/'validation_metrics.csv',index=False)
    best_item.to_csv(out/'item_validation_predictions.csv',index=False)
    best_content_pred.to_csv(out/'content_validation_predictions.csv',index=False)
    save(out/'selection.json',{'k':best_k,'ridge_alpha':best_content.ridge_alpha,'hybrid_alpha':weight,
         'train_sha256':fingerprint(train),'validation_sha256':fingerprint(val),'fit':'Train only',
         'selection':'Validation MAE, RMSE tie break','test_tuning':False})
    if isinstance(best_content,SharedContentCF):
        np.savez_compressed(out/'content_parameters.npz',coef=best_content.ridge_.coef_,intercept=best_content.ridge_.intercept_,
              feature_names=np.asarray(best_content.feature_names_),genre_names=np.asarray(best_content.genre_cols_),idf=best_content.idf_,
              year_mean=best_content.year_mean_,year_std=best_content.year_std_,
              projection=best_content.plot_projection_.components_ if best_content.plot_projection_ is not None else np.empty((0,0)))
    print('Selected',best_k,best_content.ridge_alpha,weight,flush=True)
    rows=evaluate_all(parts,item,best_content,weight,out,'movie')
    return item,best_content,weight,rows


def evaluate_all(parts,item,content,weight,out,domain):
    tick=time.perf_counter()
    test=parts['test']
    a,b=scored(item,test),scored(content,test)
    paired=align_predictions(a,b)
    hybrid=test.copy()
    hybrid['predicted_rating']=weight*a.predicted_rating.to_numpy()+(1-weight)*b.predicted_rating.to_numpy()
    hybrid['method']='hybrid'
    rows=[];diagnostics={}
    for name,pred in [('Item-based',a),('Content-based',b),('Hybrid',hybrid)]:
        pred.to_csv(out/(name.lower()+'_test_predictions.csv'),index=False)
        reread=pd.read_csv(out/(name.lower()+'_test_predictions.csv'))
        score=metrics(test.rating,pred.predicted_rating)
        np.testing.assert_allclose(list(score.values()),list(metrics(reread.rating,reread.predicted_rating).values()),rtol=1e-12)
        fallback=pred.method.isin(['user_mean','item_mean','global_mean']) if name!='Hybrid' else a.method.ne('cf')&b.method.ne('content')
        rows.append({'domain':domain,'model':name,**score,'rows':len(test),'fallback_ratio':float(fallback.mean()),
                     'cold_user_rows':int((~test.user_id.isin(parts['train'].user_id)).sum()),
                     'cold_item_rows':int((~test.item_id.isin(parts['train'].item_id)).sum())})
        groups={}
        cold_user=~test.user_id.isin(parts['train'].user_id)
        cold_item=~test.item_id.isin(parts['train'].item_id)
        for label,mask in [('cold_user',cold_user),('cold_item',cold_item),('warm',~cold_user&~cold_item),('fallback',fallback)]:
            groups[label]={'rows':int(mask.sum()),**(metrics(test.loc[mask,'rating'],pred.loc[mask,'predicted_rating']) if mask.any() else {})}
        diagnostics[name]={'groups':groups,'method_counts':pred.method.value_counts().to_dict()}
    diagnostics['hybrid_component_fallback']={'either_component_ratio':float((a.method.ne('cf')|b.method.ne('content')).mean()),
                                              'both_components_ratio':float((a.method.ne('cf')&b.method.ne('content')).mean())}
    save(out/'evaluation_diagnostics.json',diagnostics)
    pd.DataFrame(rows).to_csv(out/'test_metrics.csv',index=False)
    record_stage(domain+'_evaluation',tick)
    return rows


def main():
    global RUN_LOG_DIR
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',choices=['1m','32m'],default='32m')
    parser.add_argument('--output-dir',type=Path)
    parser.add_argument('--processed-dir',type=Path,default=ROOT/'data/processed')
    parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--validate-only',action='store_true')
    args=parser.parse_args()
    out=args.output_dir or ROOT/'results/hybrid'/('run_'+time.strftime('%Y%m%d_%H%M%S'))
    out.mkdir(parents=True,exist_ok=False)
    RUN_LOG_DIR=out
    started=time.perf_counter()
    os.environ['MPLCONFIGDIR']=str(out.resolve()/'matplotlib_cache')
    processed=args.processed_dir
    missing=[]
    if args.dataset=='32m':
        needed=['movie_item_user_data.npz','movie_content.npz','movie_user_profiles.npz',
                'book_item_user_data.npz','book_content.npz','book_user_profiles.npz']
        missing=[str(processed/n) for n in needed if not (processed/n).exists()]
    save(out/'preflight.json',{'dataset':args.dataset,'missing':missing,'smoke':args.smoke})
    if missing:
        (out/'REPORT.md').write_text('# 실행 미완료\n\n32M/TMDB 및 Goodbooks 전처리 데이터 부재. 1M NPZ를 32M 데이터로 대체하지 않음.\n\n'+'\n'.join('- '+p for p in missing),encoding='utf-8')
        print('Missing data; see',out/'preflight.json',flush=True)
        return 2
    if args.dataset=='1m':
        parts,checks=validate_common_data(processed/'item_user_data.npz',ROOT/'data/raw/ml-1m/ratings.dat',ROOT/'preprocessing_and_data/movielens-1m/preprocess_item_based.py')
        factory=lambda a:ContentBasedCF(movies_path=ROOT/'data/raw/ml-1m/movies.dat',ridge_alpha=a)
    else:
        tick=time.perf_counter()
        inventory={n:{'sha256':sha256(processed/n),'bytes':(processed/n).stat().st_size,'arrays':npz_schema(processed/n)} for n in needed}
        inventory['item_user_data.npz']={'sha256':sha256(processed/'item_user_data.npz'),'role':'preserved MovieLens 1M'}
        save(out/'input_inventory.json',inventory)
        record_stage('input_inventory_and_hashes',tick)
        tick=time.perf_counter()
        print('Loading movie CF splits',flush=True)
        parts=load_splits(processed/'movie_item_user_data.npz',validate=False)
        checks=validate_splits(parts)
        content=load_content(processed/'movie_content.npz','movie',include_plot=True)
        print('Validating movie split/profile/Train statistics',flush=True)
        checks['processed_validation']=validate_processed(processed,parts,content)
        record_stage('movie_npz_validation',tick)
        factory=lambda a:SharedContentCF(content,ridge_alpha=a)
    save(out/'data_validation.json',checks)
    if args.dataset=='32m':
        tick=time.perf_counter()
        print('Loading and validating book CF splits',flush=True)
        books=load_splits(processed/'book_item_user_data.npz','book',validate=False)
        book_checks=validate_splits(books)
        book_features=load_content(processed/'book_content.npz','book',include_plot=True)
        book_checks['processed_validation']=validate_processed(processed,books,book_features,'book')
        save(out/'book_data_validation.json',book_checks)
        record_stage('book_npz_validation',tick)
    if args.validate_only:
        print('Validation completed',out,flush=True)
        return
    if args.smoke:
        # 원본 분할 위치 유지, 실제 사용자 일부만 선택
        users=parts['train'].user_id.unique()[:12]
        parts={s:f[f.user_id.isin(users)].reset_index(drop=True) for s,f in parts.items()}
        if args.dataset=='32m':
            book_users=books['train'].user_id.unique()[:12]
            books={s:f[f.user_id.isin(book_users)].reset_index(drop=True) for s,f in books.items()}
    item,content,weight,rows=experiment(parts,factory,out/('movielens_'+args.dataset))
    if args.dataset=='32m':
        shared=set(content.genre_cols_)&{canonical_genre(n) for n in book_features[1]}
        target_cols=[i for i,n in enumerate(book_features[1]) if canonical_genre(n) in shared]
        save(out/'feature_alignment.json',{'input_genre_order':list(content.genre_cols_),
             'book_unmapped_genres':sorted(str(n) for n,v in zip(book_features[1],book_features[2].sum(0)) if v>0 and canonical_genre(n) not in shared),
             'book_items_without_mapped_genres':int((book_features[2][:,target_cols].sum(1)==0).sum()),
             'genre_semantics':'이름 일치가 의미 일치를 보장하지 않음',
             'excluded':['author','runtime','pages','rating'],
             'plot_embedding':'existing BGE-M3 1024 -> movie Train-item fitted SVD32; fixed book transform',
             'ridge_input_names':content.feature_names_})
        tick=time.perf_counter()
        book_content=content.transfer(book_features,books['train'])
        np.testing.assert_array_equal(content.ridge_.coef_,book_content.ridge_.coef_)
        np.testing.assert_array_equal(content.feature_names_,book_content.feature_names_)
        print('Book adaptation: fixed movie K, Ridge and hybrid weight; Book Train support only',flush=True)
        book_item=ItemBasedCF(item.n_neighbors).fit(books['train'])
        record_stage('book_support_profiles_and_item_adaptation',tick)
        book_out=out/'goodbooks';book_out.mkdir()
        diagnostic=book_out/'validation_diagnostic';diagnostic.mkdir()
        evaluate_all({'train':books['train'],'test':books['validation']},book_item,book_content,weight,diagnostic,'book_validation_diagnostic')
        rows+=evaluate_all(books,book_item,book_content,weight,book_out,'book_support_adapted')
    table=pd.DataFrame(rows);table.to_csv(out/'comparison.csv',index=False)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    for ax,metric in zip(axes,['mae','rmse']):
        table.pivot(index='model',columns='domain',values=metric).plot.bar(ax=ax,color=['#555555','#aaaaaa'])
        ax.set_ylabel(metric.upper());ax.tick_params(axis='x',rotation=20)
    fig.tight_layout();fig.savefig(out/'comparison.png',dpi=150);plt.close(fig)
    runtime={'seconds':time.perf_counter()-started,'peak_working_set_mib':getattr(psutil.Process().memory_info(),'peak_wset',0)/2**20,
             'versions':{n:importlib.metadata.version(n) for n in ['numpy','pandas','scipy','scikit-learn','matplotlib','psutil']}}
    save(out/'runtime.json',runtime)
    if args.dataset=='32m':
        final_hashes={n:sha256(processed/n) for n in inventory}
        if any(final_hashes[n]!=inventory[n]['sha256'] for n in inventory):raise ValueError('실행 중 입력 파일 변경')
        save(out/'input_preservation.json',{'all_input_sha256_unchanged':True,'sha256':final_hashes})
    (out/'REPORT.md').write_text('# 실제 실행 결과\n\n'+table.to_string(index=False)+'\n\n'+json.dumps(runtime,indent=2)+
       '\n\n모든 설정은 영화 Validation에서 선택. Test는 최종 평가에만 사용. Train+Validation 재학습 없음.\n'+
       '도서 Item-based는 support 적응이며 영화 유사도의 직접 전이가 아님. Content는 영화 Ridge·IDF·연도 변환 고정, 도서 Train 프로필 사용.\n'+
       '공통 특성: 정규화 장르, 연도, 기존 BGE-M3 임베딩의 영화 Train-item SVD32 및 취향 cosine. 작가·페이지·상영시간·평점은 콘텐츠 입력에서 제외.\n'+
       '1M 실행은 교차 도메인 최종 평가를 대신하지 않음. Smoke 결과는 전체 성능이 아님.\n',encoding='utf-8')
    print(table.to_string(index=False),flush=True)
    print(out,flush=True)
    if args.dataset=='32m' and not args.smoke:
        from scripts.verify_recsys_run import verify
        from scripts.build_recsys_report import build_report
        verify(out,processed)
        print(build_report(out,processed),flush=True)


if __name__=='__main__':sys.exit(main())
