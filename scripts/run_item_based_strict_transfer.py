"""Git 없는 장르-only CF 증류·strict 도서 평가 실행기."""
import argparse
from copy import copy
from datetime import datetime,timezone,timedelta
import json
import os
from pathlib import Path
import sys
import time
os.environ.setdefault('OPENBLAS_NUM_THREADS','4')
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
import psutil
from src.item_based_transfer import ItemBasedTransfer
from src.item_based_transfer_data import StrictArrayReader,GenreCatalog,save_json,sha256,capture_originals,verify_originals

GRID={'rank':[8,16,32],'alpha':[1.,10.,100.],'k':[10,30,50]}


class _LogTee:
    def __init__(self,stream,log):self.stream,self.log=stream,log
    def write(self,text):
        self.stream.write(text);self.log.write(text);self.log.flush()
        return len(text)
    def flush(self):self.stream.flush();self.log.flush()


def errors(y,p):
    a,b=np.asarray(y,dtype=float),np.asarray(p,dtype=float)
    if a.shape!=b.shape or not a.size or not np.isfinite(a).all() or not np.isfinite(b).all():raise ValueError('평가 배열 오류')
    d=b-a
    return {'mae':float(np.abs(d).mean()),'rmse':float(np.sqrt(np.mean(d*d)))}


def catalog_subset(content,ids):
    ids=np.asarray(ids,dtype=np.int64)
    loc=pd.Index(content.ids).get_indexer(ids)
    if (loc<0).any():raise ValueError('평가 아이템 콘텐츠 누락')
    return GenreCatalog(ids,content.names,content.binary[loc])


def report(run,selection,validation,book_metrics,runtime,preservation):
    def table(frame):
        lines=['| '+' | '.join(map(str,frame.columns))+' |','| '+' | '.join(['---']*len(frame.columns))+' |']
        for row in frame.itertuples(index=False,name=None):lines.append('| '+' | '.join(f'{x:.6f}' if isinstance(x,(float,np.floating)) else str(x) for x in row)+' |')
        return '\n'.join(lines)
    text=f'''# Item-based strict genre-only 전이 실행 보고서

## 목적과 구분
영화 Train의 협업 cosine 저차원 표현을 장르에서 Ridge로 증류하고 영화 anchor의 실제 협업 표현 이웃으로 영화 Train 평균 평점을 집계함. 기존 ItemBasedCF/Content/Hybrid/전처리/공유 NPZ/기존 결과는 수정하지 않음. Git/GitHub 호출 없음.

도서 사용자와 영화 사용자는 연결되지 않음. 도서 평점·사용자 속성도 학습에 사용하지 않아 개인화는 식별 불가. **같은 책은 모든 사용자에게 같은 점수**. 이 모델은 비개인화 CF metric distillation이며 기존 개인화 Item-based CF의 무수정 직접 적용이 아님. 기존 Hybrid.fit 경로와 자동 호환되지 않음.

## 데이터와 접근 순서
Movie Train {selection['n_train']:,}행, Movie Validation {selection['n_validation']:,}행 전부 사용. Movie Test 배열 접근 금지. Book Train/Validation 및 모든 user-profile 배열 접근 금지. 장르 이외의 plot/year/numeric 배열도 읽지 않음. Book 콘텐츠는 source 선택 동결 후에만 읽음.

source model/config/Validation 지표를 저장·해시 고정한 뒤 도서 콘텐츠와 Test ID를 읽음. 사용자별 예측 배열·CSV·책별 점수를 모두 저장하고 prediction_seal.json으로 봉인한 뒤 **처음으로** Book Test 정답을 로딩. array_access.json과 phase_journal.json 참조. 파일 전체 SHA-256을 위한 byte 읽기는 NPZ 평점 배열의 디코딩·학습 사용과 구분함.

## 모델과 사전 grid
행 L2 정규화한 영화 item-user 원점수 행렬 A를 사용. A Aᵀ가 cosine 협업 유사도임. rank-32 SVD의 prefix rank 8/16/32를 후보로 사용. 협업 표현 Q_r=normalize(A V_r), 장르 X는 영화 Train 등장 아이템의 고정 사전만으로 이진 장르를 정렬한 뒤 행 L2 정규화함. IDF·연도·사전학습 임베딩은 사용하지 않음.

Ridge B=argmin ||XB+c-Q_r||² + alpha||B||². 타깃 표현 z_b=normalize(x_b B+c). 실제 협업 anchor Q_r와 z_b의 cosine을 양수 부분만 사용하고 Top-K 영화 Train 평균 평점을 가중평균함. 가중치 합이 0이거나 source 장르가 없으면 영화 Train 전체 평균 fallback. 영화 Validation에서는 자기 anchor를 제외하고 도서 namespace에서는 숫자 ID 충돌을 무시함.

사전 grid: {json.dumps(GRID)}. 선택 rank={selection['rank']}, alpha={selection['alpha']}, K={selection['k']}. Movie Validation MAE→RMSE→작은 rank/alpha/K 순으로 선택. Baseline 및 permutation은 주 모델 선택 후보에 혼합하지 않음. 도서 결과에 따른 재선택 없음.

## 영화 Validation 및 협업 기여 진단
{table(validation)}

global mean, raw genre cosine anchor, 선택 설정의 teacher-row permutation ablation을 비교함. permutation은 영화 Validation에서만 평가하며 실제 anchor 표현·평점은 유지하고 콘텐츠-협업 teacher 대응만 섞음. 그 결과가 나빠지거나 달라지더라도 cross-domain 우수성을 입증하는 것은 아님. 추가 영화 Train item-holdout 진단은 고정한 설정으로만 수행하며 별도 cold_item_audit.json 참조. 설정 선택 이후의 진단이고 독립적인 새 도메인 성능 보장은 아님.

## 최종 Goodbooks 기존 Test (정답 후개방)
{table(book_metrics)}

도서 평점 {selection['n_book_test']:,}행 최종 평가. 모든 모델은 source 설정으로 고정됨. permutation은 Book에서 평가하지 않음. 지표가 나빠도 재튜닝하지 않음. 도서 점수는 알려진 척도 1~5로 제한하며 데이터에서 min/max를 학습하지 않음. 영화 source 척도는 0.5~5임.

## 한계
장르-only 선형 학생은 표현력이 작고 같은 장르 조합은 같은 점수가 됨. CF 항이 실제로 작동하는지는 Validation permutation 차이, 협업 표현 reconstruction 진단과 예측 차이를 함께 봐야 함. 단순 콘텐츠 KNN과 협업 공간의 metric을 학습하는 점이 다르지만 좋은 성능을 보장하지는 않음.

공동 전처리의 genre union에서는 영화 Train 활성 사전만 다시 구성하므로 도서 전용 열은 fit에서 제외됨. 공동 정규화된 연도와 출처 미확인 임베딩은 전혀 사용하지 않음. 그러나 upstream의 TMDB 매칭·데이터 선택·원래 도서 메타데이터 수집 과정 자체는 NPZ만으로 되돌리거나 독립 검증할 수 없음. 기존 도서 Test가 이전 실험에서 노출된 반복 벤치마크라는 점도 남음. 이번 설계·튜닝에는 도서 결과를 사용하지 않았지만 완전히 처음 보는 평가셋이라고 주장하지 않음.

## 실행·검증·보존
측정 {runtime['seconds']:.2f}초, 최대 working set {runtime['peak_working_set_mib']:.2f} MiB. 기존 {preservation['checked_files']}개 파일 해시·크기·mtime 재검증, 변경 {len(preservation['changed_files'])}개. stage_resources.json, preservation.json, independent_verification.json, unit_tests.txt 참조. 전체 사용자×아이템 dense 행렬은 만들지 않았으며 256개 콘텐츠 query 배치와 sparse source 행렬을 사용함.

실행: `.venv\\Scripts\\python.exe -B scripts/run_item_based_strict_transfer.py`. 검증: `.venv\\Scripts\\python.exe -B scripts/verify_item_based_strict_transfer.py --run-dir <이 폴더>`.

## 논문과 구현의 차이
협업 잠재요인을 콘텐츠에서 예측한다는 발상은 [van den Oord et al., 2013](https://proceedings.neurips.cc/paper/2013/file/b3ba8f1bee1238a2f37603d90b58898d-Paper.pdf)을 참고함. 해당 논문의 오디오 신경망 모델을 재현한 것은 아님.

[Zhou et al., 2023](https://arxiv.org/abs/2302.02151)은 cold item 콘텐츠-협업 표현의 불일치/blur 문제와 contrastive 접근을 다룸. 본 구현은 contrastive 학습이나 CCFCRec 재현이 아니라 선형 Ridge 증류이며 그 한계를 인정함. 두 논문은 비연결된 도서 사용자의 개인화 가능성을 보장하지 않음.
'''
    (run/'REPORT.md').write_text(text,encoding='utf-8')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--processed-dir',type=Path,default=ROOT/'data/processed')
    parser.add_argument('--output-dir',type=Path)
    parser.add_argument('--smoke',action='store_true')
    args=parser.parse_args()
    stamp=datetime.now(timezone(timedelta(hours=9))).strftime('%Y%m%d_%H%M%S_KST')
    out=args.output_dir or ROOT/'results/item_based'/('strict_transfer_'+stamp)
    out=out.resolve();out.mkdir(parents=True,exist_ok=False)
    log=(out/'execution.log').open('x',encoding='utf-8')
    sys.stdout=_LogTee(sys.stdout,log);sys.stderr=_LogTee(sys.stderr,log)
    os.environ['MPLCONFIGDIR']=str(out/'matplotlib_cache')
    started=time.perf_counter();stages=[];journal=[]
    def phase(name,**kwargs):
        event={'index':len(journal),'phase':name,'elapsed_seconds':time.perf_counter()-started,**kwargs}
        journal.append(event);save_json(out/'phase_journal.json',journal)
        print(json.dumps(event,ensure_ascii=False),flush=True)
    def stage(name,tick):
        info=psutil.Process().memory_info()
        row={'stage':name,'seconds':time.perf_counter()-tick,'rss_mib':info.rss/2**20,'peak_working_set_mib':getattr(info,'peak_wset',0)/2**20}
        stages.append(row);save_json(out/'stage_resources.json',stages);print(json.dumps(row),flush=True)
    original=capture_originals(ROOT,out)
    save_json(out/'originals_before.json',original)
    save_json(out/'predeclared_protocol.json',{'grid':GRID,'teacher_rank':32,'teacher_iterations':5,'random_state':42,'permutation_seed':314159,
         'selection':'Movie Validation MAE/RMSE/rank/alpha/K','target':'book existing test only','features':'movie-Train binary genre only; row L2',
         'movie_test':'forbidden','book_train_validation':'forbidden','target_content_load':'after source configuration freeze',
         'prediction':'nonpersonalized same book same score; source anchor self-exclusion on Movie Validation',
         'cold_audit':'fixed selected settings; 5% Movie Train items held out, source only'})
    # 기존 실행기·다른 테스트는 내부 Git/학습 호출 가능성이 있어 실행하지 않음
    import unittest
    os.environ['STRICT_TRANSFER_TEST_DIR']=str(out/'unit_test_artifacts')
    with (out/'unit_tests.txt').open('w',encoding='utf-8') as log:
        suite=unittest.defaultTestLoader.discover(str(ROOT/'tests'),pattern='test_item_based_strict_transfer.py')
        tests=unittest.TextTestRunner(stream=log,verbosity=2).run(suite)
    if not tests.wasSuccessful() or tests.testsRun==0:raise RuntimeError('새 전용 테스트 실패')
    phase('unit_tests_passed',count=tests.testsRun)
    reader=StrictArrayReader(args.processed_dir)
    tick=time.perf_counter()
    train,val=reader.source_split('train'),reader.source_split('validation')
    content=reader.catalog('movie')
    radix=max(int(train.item_id.max()),int(val.item_id.max()))+1
    train_keys=train.user_id.to_numpy(dtype=np.uint64)*np.uint64(radix)+train.item_id.to_numpy(dtype=np.uint64)
    val_keys=val.user_id.to_numpy(dtype=np.uint64)*np.uint64(radix)+val.item_id.to_numpy(dtype=np.uint64)
    if np.isin(val_keys,train_keys,assume_unique=True).any():raise ValueError('Movie Train/Validation 중복')
    del train_keys,val_keys
    stage('load_and_validate_source_only',tick)
    if args.smoke:
        train=train.iloc[:20000].copy();val=val.iloc[:5000].copy()
        model=ItemBasedTransfer(rank=4,teacher_rank=8).fit(train,content)
        query=catalog_subset(content,np.unique(val.item_id))
        scores,_=model.score_catalog(query,namespace='movie')
        pred=scores[model.n_neighbors][pd.Index(query.ids).get_indexer(val.item_id)]
        save_json(out/'smoke_metrics.json',errors(val.rating,pred))
        save_json(out/'array_access.json',reader.events)
        check=verify_originals(ROOT,original);save_json(out/'preservation.json',check)
        if not check['all_preserved']:raise RuntimeError('기존 파일 변경')
        phase('source_only_smoke_passed_no_book_access')
        return
    tick=time.perf_counter();phase('teacher_training_source_only',n_train=len(train),n_validation=len(val))
    teacher=ItemBasedTransfer(rank=32,teacher_rank=32).fit(train,content)
    stage('source_collaborative_svd',tick)
    query=catalog_subset(content,np.unique(val.item_id));row_index=pd.Index(query.ids).get_indexer(val.item_id)
    rows=[];best=None;best_key=None;tick=time.perf_counter()
    for rank in GRID['rank']:
        for alpha in GRID['alpha']:
            candidate=teacher.candidate(rank,alpha,max(GRID['k']))
            score,_=candidate.score_catalog(query,GRID['k'],namespace='movie')
            for k in GRID['k']:
                metric=errors(val.rating,score[k][row_index]);row={'rank':rank,'alpha':alpha,'k':k,**metric};rows.append(row)
                key=(metric['mae'],metric['rmse'],rank,alpha,k)
                if best_key is None or key<best_key:best_key=key;best=candidate.candidate(rank,alpha,k)
    grid=pd.DataFrame(rows);grid.to_csv(out/'movie_validation_grid.csv',index=False)
    stage('movie_validation_fixed_grid',tick)
    baseline_score,_=teacher.score_catalog(query,GRID['k'],mode='genre',namespace='movie')
    baseline_rows=[{'model':'raw_genre_cosine','k':k,**errors(val.rating,baseline_score[k][row_index])} for k in GRID['k']]
    baseline_k=int(pd.DataFrame(baseline_rows).sort_values(['mae','rmse','k']).iloc[0].k)
    main_scores,detail=best.score_catalog(query,namespace='movie')
    main_pred=main_scores[best.n_neighbors][row_index]
    shuffled=teacher.candidate(best.rank,best.ridge_alpha,best.n_neighbors,314159)
    shuffled_scores,_=shuffled.score_catalog(query,namespace='movie')
    shuffled_pred=shuffled_scores[best.n_neighbors][row_index]
    val_summary=pd.DataFrame([
        {'model':'source_global_mean',**errors(val.rating,np.full(len(val),best.global_mean_))},
        {'model':'raw_genre_cosine',**errors(val.rating,baseline_score[baseline_k][row_index])},
        {'model':'cf_metric_distillation',**errors(val.rating,main_pred)},
        {'model':'teacher_permutation_validation_only',**errors(val.rating,shuffled_pred)}])
    val_summary.to_csv(out/'movie_validation_summary.csv',index=False)
    pd.DataFrame(baseline_rows).to_csv(out/'genre_baseline_validation_grid.csv',index=False)
    validation=val.copy();validation['distilled']=main_pred;validation['genre']=baseline_score[baseline_k][row_index]
    validation['permuted']=shuffled_pred;validation['global_mean']=best.global_mean_
    validation.to_csv(out/'movie_validation_predictions.csv',index=False)
    sample=best.teacher_audit_indices_
    latent=best.anchor_vectors_[sample]@best.anchor_vectors_[sample].T
    mask=~np.eye(len(sample),dtype=bool)
    save_json(out/'collaborative_signal_audit.json',{'permutation_mean_absolute_prediction_change':float(np.abs(main_pred-shuffled_pred).mean()),
        'permutation_max_absolute_prediction_change':float(np.abs(main_pred-shuffled_pred).max()),
        'main_prediction_std':float(main_pred.std()),'teacher_reconstruction_offdiagonal_cosine_mse':float(np.mean((latent-best.teacher_audit_cosine_)[mask]**2)),
        'singular_values':best.singular_values_.tolist(),'explained_variance_ratio':best.explained_variance_ratio_.tolist(),
        'teacher_matrix_shape':best.teacher_shape_,'teacher_matrix_nnz':best.teacher_nnz_,'genre_dimension':len(best.genre_names_)})
    tick=time.perf_counter()
    held=np.sort(np.random.default_rng(2026).choice(best.anchor_ids_,max(1,len(best.anchor_ids_)//20),replace=False))
    hold_mask=train.item_id.isin(held)
    audit_train=train.loc[~hold_mask];audit_test=train.loc[hold_mask]
    audit_model=ItemBasedTransfer(best.rank,best.ridge_alpha,best.n_neighbors,32).fit(audit_train,content)
    audit_query=catalog_subset(content,held);idx=pd.Index(held).get_indexer(audit_test.item_id)
    audit_scores,audit_details=audit_model.score_catalog(audit_query,namespace='movie')
    audit_genre,_=audit_model.score_catalog(audit_query,[baseline_k],mode='genre',namespace='movie')
    save_json(out/'cold_item_audit.json',{'heldout_items':len(held),'fit_rows':len(audit_train),'evaluation_rows':len(audit_test),
        'all_held_items_absent_from_teacher':bool(not np.isin(held,audit_model.anchor_ids_).any()),
        'distilled':errors(audit_test.rating,audit_scores[best.n_neighbors][idx]),
        'genre':errors(audit_test.rating,audit_genre[baseline_k][idx]),
        'global_mean':errors(audit_test.rating,np.full(len(audit_test),audit_model.global_mean_)),
        'selection_changed':False,'role':'source-only descriptive item holdout after fixed Movie Validation selection'})
    stage('source_only_item_holdout_audit',tick)
    del audit_model,audit_train,audit_test,validation,shuffled,teacher
    selection={'rank':best.rank,'alpha':best.ridge_alpha,'k':best.n_neighbors,'genre_baseline_k':baseline_k,
               'n_train':len(train),'n_validation':len(val),'source_global_mean':best.global_mean_,
               'target_ratings_used_for_selection':False,'movie_test_used':False,'book_train_validation_used':False,
               'genre_vocabulary':best.genre_names_}
    best.save(out/'source_model_frozen.npz')
    save_json(out/'selected_config.json',selection)
    source_paths=[out/'source_model_frozen.npz',out/'selected_config.json',out/'movie_validation_grid.csv',out/'movie_validation_summary.csv']
    phase('source_model_and_configuration_frozen',sha256={p.name:sha256(p) for p in source_paths})
    reader.allow_target_features(source_paths)
    book_content=reader.catalog('book')
    book_users=reader.read('book_item_user_data.npz','test_user_ids')
    book_items=reader.read('book_item_user_data.npz','test_book_ids')
    if len(book_users)!=len(book_items):raise ValueError('Book Test ID 길이 오류')
    restored=ItemBasedTransfer.load(out/'source_model_frozen.npz').bind_target_content(book_content)
    best.bind_target_content(book_content)
    predictions=restored.predict_many(book_users,book_items)
    np.testing.assert_allclose(predictions,best.predict_many(np.zeros(len(book_users)),book_items),rtol=0,atol=1e-12)
    genre_scores,genre_details=best.score_catalog(book_content,[baseline_k],mode='genre',namespace='book')
    loc=pd.Index(book_content.ids).get_indexer(book_items)
    genre_pred=np.full(len(loc),np.clip(best.global_mean_,1.,5.));known=loc>=0;genre_pred[known]=genre_scores[baseline_k][loc[known]]
    global_pred=np.full(len(loc),np.clip(best.global_mean_,1.,5.))
    np.savez_compressed(out/'book_test_predictions_frozen.npz',user_ids=book_users,book_ids=book_items,
                        distilled=predictions,genre=genre_pred,global_mean=global_pred)
    pd.DataFrame({'book_id':book_content.ids,'distilled':best.bound_scores_,'genre':genre_scores[baseline_k],
                  'mapped_genre':best.bound_details_['mapped_genre'],
                  'contributors':best.bound_details_['contributors'][best.n_neighbors]}).to_csv(out/'book_item_scores_frozen.csv',index=False)
    pd.DataFrame({'user_id':book_users,'book_id':book_items,'distilled':predictions,'genre':genre_pred,'global_mean':global_pred}).to_csv(
                 out/'book_test_predictions_frozen.csv',index=False)
    prediction_paths=[out/n for n in ['book_test_predictions_frozen.npz','book_test_predictions_frozen.csv','book_item_scores_frozen.csv']]
    seal=reader.seal_predictions(prediction_paths)
    save_json(out/'prediction_seal.json',seal)
    phase('target_predictions_frozen_before_truth',n_book_test=len(book_items),prediction_sha256={p.name:sha256(p) for p in prediction_paths})
    truth=reader.read('book_item_user_data.npz','test_ratings')
    phase('book_test_truth_first_opened_after_seal')
    if len(truth)!=len(predictions) or not np.isfinite(truth).all() or not np.all((truth>=1)&(truth<=5)):raise ValueError('Book Test 정답 오류')
    book_metrics=pd.DataFrame([{'model':name,'rows':len(truth),**errors(truth,p)} for name,p in [
        ('source_global_mean',global_pred),('raw_genre_cosine',genre_pred),('cf_metric_distillation',predictions)]])
    book_metrics.to_csv(out/'book_test_metrics.csv',index=False)
    no_genre=~best.bound_details_['mapped_genre']
    save_json(out/'target_diagnostics.json',{'content_items':len(book_content.ids),'test_rows':len(truth),'unknown_content_rows':int((~known).sum()),
        'no_mapped_genre_items':int(no_genre.sum()),'no_mapped_genre_test_rows':int(no_genre[loc[known]].sum()),
        'same_book_same_score_verified':bool(pd.DataFrame({'book':book_items,'prediction':predictions}).groupby('book').prediction.nunique().max()==1),
        'target_genre_vocabulary_fit':False,'target_profile_or_similarity_fit':False,
        'genre_input_count':len(best.genre_names_),'score_std_items':float(best.bound_scores_.std()),
        'clipped_to_known_scale':[1,5]})
    save_json(out/'array_access.json',reader.events)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    for ax,metric in zip(axes,['mae','rmse']):
        ax.bar(book_metrics.model,book_metrics[metric],color=['#bbbbbb','#888888','#444444'])
        ax.set_ylabel(metric.upper());ax.tick_params(axis='x',labelrotation=20)
    fig.tight_layout();fig.savefig(out/'book_metrics.png',dpi=150);plt.close(fig)
    runtime={'seconds':time.perf_counter()-started,'peak_working_set_mib':getattr(psutil.Process().memory_info(),'peak_wset',0)/2**20}
    save_json(out/'runtime.json',runtime)
    check=verify_originals(ROOT,original);save_json(out/'preservation.json',check)
    if not check['all_preserved']:raise RuntimeError('기존 파일 변경 발견')
    selection_for_report={**selection,'n_book_test':len(truth)}
    report(out,selection_for_report,val_summary,book_metrics,runtime,check)
    from scripts.verify_item_based_strict_transfer import verify
    verify(out,args.processed_dir)
    phase('completed_without_reselection',output_dir=str(out))
    print(val_summary.to_string(index=False),flush=True);print(book_metrics.to_string(index=False),flush=True);print(out,flush=True)


if __name__=='__main__':main()
