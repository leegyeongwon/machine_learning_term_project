"""검증된 실제 실행 결과로 표·그래프·최종 보고서 생성."""
import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
from src.transfer_data import load_content,canonical_genre
from src.hybrid import metrics
from src.processed_validation import sha256


def markdown(frame):
    lines=['| '+' | '.join(map(str,frame.columns))+' |','| '+' | '.join(['---']*len(frame.columns))+' |']
    for row in frame.itertuples(index=False,name=None):
        lines.append('| '+' | '.join(f'{v:.6f}' if isinstance(v,(float,np.floating)) else str(v) for v in row)+' |')
    return '\n'.join(lines)


def build_report(run,processed,test_log=None):
    run,processed=Path(run),Path(processed)
    read=lambda name:json.loads((run/name).read_text(encoding='utf-8'))
    verified=read('independent_verification.json')
    if not verified['all_checks_passed']:raise ValueError('독립 결과 검증 필요')
    table=pd.read_csv(run/'comparison.csv')
    val=pd.read_csv(run/'movielens_32m/validation_metrics.csv')
    selection=read('movielens_32m/selection.json')
    movie,book=read('data_validation.json'),read('book_data_validation.json')
    runtime,stages,alignment=read('runtime.json'),read('stage_resources.json'),read('feature_alignment.json')
    movie_diag,book_diag=read('movielens_32m/evaluation_diagnostics.json'),read('goodbooks/evaluation_diagnostics.json')
    rows=[]
    for name in ['Item-based','Content-based','Hybrid']:
        a=table[table.model.eq(name)&table.domain.eq('movie')].iloc[0]
        b=table[table.model.eq(name)&table.domain.eq('book_support_adapted')].iloc[0]
        rows.append({'모델':name,'영화 MAE':a.mae,'영화 RMSE':a.rmse,'도서 적응 MAE':b.mae,'도서 적응 RMSE':b.rmse,
                     'MAE 차이(도서-영화)':b.mae-a.mae,'RMSE 차이(도서-영화)':b.rmse-a.rmse})
    comparison=pd.DataFrame(rows);comparison.to_csv(run/'final_comparison.csv',index=False,encoding='utf-8-sig')
    bc=load_content(processed/'book_content.npz','book')
    ids,names,g,year=bc
    lookup={int(i):j for j,i in enumerate(ids)}
    shared=set(alignment['input_genre_order'])
    mapped_cols=[j for j,n in enumerate(names) if canonical_genre(n) in shared]
    unmapped=g[:,mapped_cols].sum(1)==0
    groups=[]
    for name in ['Item-based','Content-based','Hybrid']:
        pred=pd.read_csv(run/'goodbooks'/(name.lower()+'_test_predictions.csv'))
        pos=np.array([lookup[int(i)] for i in pred.item_id])
        flags={'mapped_genre':~unmapped[pos],'no_mapped_genre':unmapped[pos],
               'future_year':year[pos]>2026,'missing_year':~np.isfinite(year[pos])}
        for label,mask in flags.items():
            if mask.any():groups.append({'model':name,'group':label,'rows':int(mask.sum()),**metrics(pred.loc[mask,'rating'],pred.loc[mask,'predicted_rating'])})
    feature_groups=pd.DataFrame(groups);feature_groups.to_csv(run/'feature_group_diagnostics.csv',index=False)
    def group_mae(label):
        selected=feature_groups[feature_groups.model.eq('Content-based')&feature_groups.group.eq(label)]
        return f'{selected.iloc[0].mae:.6f}' if len(selected) else '해당 평가 행 없음'
    population=[]
    for domain,stats in [('movie',movie),('book',book)]:
        with np.load(processed/(domain+'_item_user_data.npz'),allow_pickle=False) as z:
            r=z['train_ratings']
            population.append({'domain':domain,'train_mean':float(np.mean(r,dtype=np.float64)),
                               'train_std':float(np.std(r,dtype=np.float64)),
                               'support_density':len(r)/(stats['processed_validation']['users']*stats['processed_validation']['items']),
                               'ratings_per_user':len(r)/stats['train']['users']})
    pd.DataFrame(population).to_csv(run/'population_diagnostics.csv',index=False)
    os.environ['MPLCONFIGDIR']=str((run/'matplotlib_cache').resolve())
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(13,4))
    grid=pd.read_csv(run/'movielens_32m/hybrid_grid.csv')
    for ax,data,x,label in [(axes[0],val[val.model.eq('Item-based')],'parameter','Item K'),
                            (axes[1],val[val.model.eq('Content-based')],'parameter','Ridge alpha'),
                            (axes[2],grid,'alpha','Hybrid item weight')]:
        ax.plot(data[x],data.mae,'o-',color='#444444',markersize=3,label='MAE')
        ax.plot(data[x],data.rmse,'--',color='#999999',label='RMSE')
        ax.set(xlabel=label,ylabel='Validation error');ax.legend();ax.grid(alpha=.15)
    axes[1].set_xscale('log')
    fig.tight_layout();fig.savefig(run/'validation_curves.png',dpi=150);plt.close(fig)
    git_read=lambda *args:subprocess.check_output(['git','--no-optional-locks',*args],cwd=ROOT).decode('utf-8',errors='replace').strip()
    git_status=git_read('status','--short')
    (run/'git_status_final.txt').write_text(git_status,encoding='utf-8')
    provenance={str(p.relative_to(ROOT)):sha256(p) for p in [ROOT/'src/item_based.py',ROOT/'src/content_based.py',
                ROOT/'src/hybrid.py',ROOT/'src/transfer_data.py',ROOT/'src/processed_validation.py',
                ROOT/'scripts/run_full_recsys_pipeline.py',ROOT/'scripts/verify_recsys_run.py',Path(__file__).resolve()]}
    (run/'code_sha256.json').write_text(json.dumps(provenance,indent=2),encoding='utf-8')
    execution_provenance={'branch':git_read('branch','--show-current'),'head':git_read('rev-parse','HEAD'),
        'hybrid_reference':git_read('rev-parse','origin/feature/hybrid'),'python':sys.version,
        'git_status':git_status,'code_sha256':provenance,'reference_source':'local remote-tracking ref; no network lookup'}
    (run/'execution_provenance.json').write_text(json.dumps(execution_provenance,ensure_ascii=False,indent=2),encoding='utf-8')
    test_status='자동 테스트 로그는 별도 확인.'
    if test_log is not None:
        text=Path(test_log).read_text(encoding='utf-8',errors='replace')
        match=re.search(r'Ran (\d+) tests',text)
        if match and text.strip().endswith('OK'):test_status=f'자동 테스트 {match.group(1)}개 통과. 로그: {Path(test_log).name}.'
    movie_h=table[table.domain.eq('movie')].set_index('model')
    book_h=table[table.domain.eq('book_support_adapted')].set_index('model')
    comparison_text=[]
    for domain,t in [('영화',movie_h),('도서 적응',book_h)]:
        for base in ['Item-based','Content-based']:
            comparison_text.append(f'{domain} Hybrid - {base}: MAE {t.loc["Hybrid","mae"]-t.loc[base,"mae"]:+.6f}, RMSE {t.loc["Hybrid","rmse"]-t.loc[base,"rmse"]:+.6f}.')
    split_rows=[]
    for domain,stats in [('movie',movie),('book',book)]:
        for s in ['train','validation','test']:split_rows.append({'domain':domain,'split':s,**stats[s]})
    report=f'''# 추천시스템 최종 실행 보고서 — 새 영화 NPZ 및 Goodbooks

## 1. 실행 코드 및 출처
실행 시 로컬 브랜치는 `{execution_provenance['branch']}`, HEAD는 `{execution_provenance['head']}`이다. 비교 기준인 로컬 `origin/feature/hybrid`는 `{execution_provenance['hybrid_reference']}`이다. 커밋은 저장소 기준점이며 미커밋 변경을 포함한 실제 실행 파일은 `code_sha256.json`의 해시로 구분한다. Python과 Git 상태는 `execution_provenance.json`에 기록하였다. 보고서 생성 단계에서는 로컬 Git 정보를 조회하였으며 네트워크 조회나 브랜치 변경은 수행하지 않았다.

## 2. 실제 수정·생성 파일
기존 Hybrid 실행기와 모델·데이터 어댑터를 사용하였다. 이번 재현 검증에서는 알고리즘과 데이터 처리 코드를 변경하지 않고 `scripts/build_recsys_report.py`의 실행 출처 기록 부분만 수정하였다. 수정 전 보고서 생성기는 상위 재현 결과 폴더의 `backups/build_recsys_report_before.py`에 보존하였다. 새 결과는 별도 출력 폴더에 생성하며 기존 모델·전처리 원문·실험 결과를 덮어쓰지 않는다.

## 3. Hybrid 구현 및 선택
`예측 = a × Item + (1-a) × Content`. Validation MAE 우선, RMSE 동률 처리, a=0~1의 0.01 간격 101개 후보. 선택 a={selection['hybrid_alpha']:.2f}. 예를 들어 Item=4, Content=3이면 {4*selection['hybrid_alpha']+3*(1-selection['hybrid_alpha']):.2f}점. 선택 K={selection['k']}, Ridge alpha={selection['ridge_alpha']:g}. 세 모델 모두 동일 Train으로 학습하며 Train+Validation 재학습 없음.

## 4. MovieLens 1M 보존
기존 `item_user_data.npz`는 새 영화 경로에서 로딩하지 않음. 기존 1M 결과 폴더를 유지했으며 1M을 포함한 7개 입력 NPZ의 실행 전후 SHA-256이 동일함. 앞선 1M 결과와 이번 새 영화 결과는 별도 폴더에 있음.

## 5. 새 영화·도서 데이터 검증
{markdown(pd.DataFrame(split_rows))}

실제 영화 NPZ 평점 {movie['processed_validation']['matched_ratings']:,}건, 사용자 {movie['processed_validation']['users']:,}명, 영화 {movie['processed_validation']['items']:,}개. 도서 평점 {book['processed_validation']['matched_ratings']:,}건, 사용자 {book['processed_validation']['users']:,}명, 도서 {book['processed_validation']['items']:,}권. 두 모델 분할의 값·행 순서 일치, 분할 내·분할 간 쌍 중복 0, 유한 원점수와 마스크 검증 통과.

영화 CF float32와 프로필 NPZ float64의 평점 값은 동일함. MovieLens ID→TMDB ID는 두 NPZ 내부에서 모두 일치함. 영화 4,633개에 고유 TMDB ID는 4,630개로 3개 TMDB 그룹이 두 MovieLens ID에 연결됨. 원본 링크 확인 없이 삭제하지 않았음. 원본 ratings.csv/links.csv가 없으므로 원본 32M의 실제 총수 및 외부 매핑의 정확성은 독립 검증 불가. 32M이라는 이름을 실제 사용 평점 수로 사용하지 않음.

영화 seed42 분할은 NPZ 쌍 집합의 사용자·영화 정렬 순서에서 재현함. 도서는 원본 행 순서가 없어 원본 seed 재현을 확인하지 못했고, 저장 seed와 두 NPZ 분할 일치를 확인함. Train 평균·표준편차·Z-score·마스크는 전체 데이터에서 검증. 저장 장르·수치 프로필 전체, 줄거리 프로필 사용자 64개 표본을 Train으로 재계산함. 영화 수치 프로필 최대 차이는 {movie['processed_validation']['numeric_profile_max_abs_error']:.9g}로 float32 누적·연도 재정규화 반올림 허용오차 1e-5 이내였음. 저장 프로필은 모델 입력에 사용하지 않고 Train에서 재계산함.

도서의 미래 연도 10건(2050~2075년), 연도 결측 95건을 확인. 원본을 수정하거나 행을 삭제하지 않았음. 미래 연도는 실제 입력에 유지, 결측은 영화 기준 평균 대체. 영화에는 2026년 이후 연도 없음. 상세 ID는 `book_data_validation.json`에 있음.

## 6. 새 영화 Validation 결과
{markdown(val)}

모든 후보와 가중치는 위 영화 Validation만으로 선택. SVD 차원 32는 메모리 계획에서 사전에 고정했으며 도서 또는 Test 지표로 조정하지 않았음. 원시 1,024차원 Ridge와 같은 실험이라고 주장하지 않음.

## 7. 영화 Test 성능
{markdown(table[table.domain.eq('movie')])}

{chr(10).join(comparison_text[:2])}

Test를 보고 재튜닝하지 않음. 통계적 유의성 검정은 수행하지 않았음.

## 8. 공통 Feature와 교차 도메인 처리
영화에서 실제 관측된 장르 20개를 기준으로 대소문자·공백·하이픈만 정규화. 원본 합집합 59개 열에서는 실제 양수 장르가 영화와 도서에서 서로 분리되어 있었음. 정규화 후 활동 장르 9개가 연결됨: crime, fantasy, history, horror, music, mystery, romance, sciencefiction, thriller. `Drama=fiction` 등 임의 의미 매핑 없음. 명칭 유사성이 의미의 완전한 동일성을 증명하지는 않음.

도서 {int(unmapped.sum()):,}권({unmapped.mean():.2%})은 연결 장르 없음. 이 경우에도 연도·줄거리·사용자 support를 사용함. 영화는 장르 없는 아이템 11개. 페이지·상영시간·작가·평점은 공통 콘텐츠 입력에서 제외함.

공통 Ridge 입력은 {len(alignment['ridge_input_names'])}차원: 장르 TF-IDF 20 + 연도 z-score 1 + 사용자 장르 취향×장르 20 + 기존 줄거리 임베딩의 영화 SVD32 + 줄거리 취향 cosine 1. 이름·순서는 `feature_alignment.json`과 `content_parameters.npz`에 저장. 영화 Train에 등장한 아이템으로만 SVD를 학습하고 도서는 같은 변환을 사용함. NPZ 원문 코드는 양쪽 모두 BGE-M3를 지정하나 NPZ에 모델 해시가 없어 생성 출처의 암호학적 증명은 제공하지 못함.

Content는 영화 Ridge 계수·IDF·연도 기준·SVD 고정. Book Train으로 도서 사용자 평균과 잔차 취향만 생성. 숫자 사용자·아이템 ID가 겹쳐도 별도 카탈로그와 프로필을 사용함. Item은 영화에서 고른 K를 고정하고 Book Train으로 책 유사도를 새로 계산하는 도서 적응임. 영화 유사도 행렬의 책 카탈로그 직접 전이는 정의되지 않음.

## 9. Goodbooks 최종 평가 — support/적응 방식
{markdown(table[table.domain.eq('book_support_adapted')])}

{chr(10).join(comparison_text[2:])}

Book Train은 지원 정보이며 Book Validation은 진단만 수행함. Movie K/Ridge/Hybrid 가중치는 변경하지 않음. Test 평점은 예측 입력에서 제거하고 예측 완료 후 정답 대조·지표 계산에만 사용. 이 표는 세 모델 모두의 엄격한 zero-shot 평가가 아님. 엄격한 영화 Item 모델 직접 전이와 그에 기반한 Hybrid의 Book 수치는 정의되지 않아 산출하지 않았음.

## 10. 최종 비교와 그래프
{markdown(comparison)}

![영화 및 도서 지원·적응 비교](comparison.png)

![Validation 후보 비교](validation_curves.png)

MovieLens 범위는 0.5~5, Goodbooks는 1~5이며 재척도화하지 않았음. 도서 예측은 알려진 1~5점 범위로 제한함. 도메인별 평점 분포·지원 밀도·카탈로그·사용자 구성이 달라 오차 차이를 순수한 전이의 인과 효과로 해석할 수 없음.

{markdown(pd.DataFrame(population))}

장르 연결 여부·미래 연도·결측 연도에 따른 실제 도서 Test 오차는 아래 진단과 같음. 그룹 차이는 관측된 상관이며 장르 정렬의 인과 효과를 분리한 ablation은 아님.

{markdown(feature_groups)}

Content의 도서 Test MAE는 영화보다 {book_h.loc['Content-based','mae']-movie_h.loc['Content-based','mae']:.6f} 증가함. 도서의 연결 장르 없는 그룹 MAE는 {group_mae('no_mapped_genre')}, 연결 장르 있는 그룹은 {group_mae('mapped_genre')}임. 미래 연도 그룹의 Content MAE는 {group_mae('future_year')}로 전체 도서 Content MAE보다 높음. 장르 표현 차이와 연도 이상값은 성능 변화의 가능한 요인이지만 독립적인 인과 검증은 하지 않았음.

Item fallback 비율은 영화 {movie_h.loc['Item-based','fallback_ratio']:.2%}, 도서 {book_h.loc['Item-based','fallback_ratio']:.2%}임. 도서의 지원 행렬 밀도가 더 낮고 평점 평균은 더 높아 동일한 영화 설정의 효과가 달라질 수 있음. Item 도서 결과는 새 유사도를 학습한 적응 결과라서 영화 파라미터의 직접 전이 성공 근거가 아님.

## 11. 실행 시간·메모리
총 측정 시간 {runtime['seconds']:.2f}초, 프로세스 최대 working set {runtime['peak_working_set_mib']:.2f} MiB. 그래프 생성까지의 측정값이며 후속 독립 검증·보고서 생성 시간은 제외함.

{markdown(pd.DataFrame(stages))}

CSR 평점 행렬·희소 프로필 곱·8,192행 배치·Ridge 충분통계 재사용. 전체 사용자×아이템 dense 행렬 또는 전체 평점×콘텐츠 dense 행렬을 생성하지 않음. GPU 전처리·대규모 임베딩 재계산 없음.

사용 라이브러리: {json.dumps(runtime['versions'],ensure_ascii=False)}.

## 12. 테스트 및 독립 검증
{test_status} 실제 영화·도서 전체 NPZ 감사 및 12명씩의 실제 Smoke Test 통과. 전체 결과의 Test 쌍·정답·유한성·범위·저장 MAE/RMSE·영화 Validation 최적 선택·두 도메인 고정 가중치 조합·7개 NPZ 보존을 별도 검증기로 재확인했음. 증거: `independent_verification.json`, `input_preservation.json`.

cold-start와 fallback 그룹별 건수·지표는 각 도메인의 `evaluation_diagnostics.json`에 저장. Hybrid가 한쪽 fallback을 포함하는 비율과 양쪽 모두 fallback인 비율을 구분함.

영화 진단:
```json
{json.dumps(movie_diag,ensure_ascii=False,indent=2)}
```
도서 진단:
```json
{json.dumps(book_diag,ensure_ascii=False,indent=2)}
```

## 13. 구현하지 않은 부분 및 한계
원본 CSV의 평점 수·외부 링크 정확성과 도서 원본 행 순서의 분할 재현은 원본 파일 부재로 미검증. 저장 줄거리 프로필은 64명 표본 검사이며 전체 사용자 증명이 아님. 기존 전처리 파일의 미래 연도 원인은 원문 출판일 확인 없이 단정하지 않았음. 원시 1,024차원 Ridge·별도 Ridge stacking·엄격한 Item zero-shot·신뢰구간·의미 기반 장르 매핑 ablation은 수행하지 않았음.

## 14. Git 변경 상태
입력 NPZ 6개와 기존 1M NPZ의 실행 전후 SHA-256 일치는 `input_preservation.json`에서 확인한다. 실행 시 추적 파일과 미추적 파일의 상태는 `git_status_final.txt`에 기록하였다. HEAD만으로 미커밋 변경이 없는 상태라고 판단하지 않으며 실제 실행 코드는 `code_sha256.json`과 함께 확인한다. 이번 검증에서 허용된 소스 변경은 보고서 생성기의 출처 기록 부분이다.

## 수식과 실행
`mean_u = 평균(Train/support 평점)`, `profile_u = normalize(Σ (r_ui-mean_u) content_i)`.
`Content = clip(mean_u + b + xᵀw)`, Ridge는 `||Xw+b-y||² + lambda||w||²` 최소화.
`Item = Σ sim(i,j) r_uj / Σ sim(i,j)` (사용자 관측 이웃만 사용), 이웃이 없으면 사용자/아이템/전체 평균 fallback.
`Hybrid = a Item + (1-a) Content`.

실행 방법과 배열 계약은 `docs/RECSYS_PIPELINE.md` 참조. 전체 실행 후 `scripts/verify_recsys_run.py --run-dir <결과폴더>`로 독립 검증 가능.
'''
    path=run/'FINAL_REPORT.md'
    path.write_text(report,encoding='utf-8')
    return path


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir',type=Path,required=True)
    parser.add_argument('--processed-dir',type=Path,default=ROOT/'data/processed')
    parser.add_argument('--test-log',type=Path)
    args=parser.parse_args()
    print(build_report(args.run_dir,args.processed_dir,args.test_log))
