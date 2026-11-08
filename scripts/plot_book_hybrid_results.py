import argparse
import json
import os
from pathlib import Path
import tempfile

os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir())/'recsys_book_matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results-dir', type=Path, default=ROOT/'results/hybrid/rebuilt_content_book')
    args = parser.parse_args()
    out = args.results_dir.resolve()
    book = pd.read_csv(out/'book_test_metrics.csv', float_precision='round_trip')
    movie = pd.read_csv(ROOT/'results/hybrid/rebuilt_content/movie_test_metrics.csv', float_precision='round_trip')
    info = json.loads((out/'book_eval_info.json').read_text(encoding='utf-8'))
    if book.to_dict('records') != info['metrics']:
        raise ValueError('Book metrics and provenance disagree')
    paired = movie.merge(book, on='model', suffixes=('_movie', '_book'), validate='one_to_one')
    if len(paired) != 3 or list(paired.model) != ['Item-based', 'Content-based', 'Hybrid']:
        raise ValueError('Unexpected comparison models')
    names = [f'{prefix}_{metric}_comparison.png' for prefix in ['book_test','movie_book'] for metric in ['mae','rmse']]
    names += ['movie_book_test_comparison.csv','book_hybrid_comparison.md']
    if any((out/n).exists() for n in names):
        raise FileExistsError('Plot/report output already exists')
    paired.to_csv(out/'movie_book_test_comparison.csv', index=False)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'axes.spines.top': False, 'axes.spines.right': False})
    labels = ['Item-based', 'Content-based Ridge', 'Hybrid (98:2)']
    for metric in ['mae','rmse']:
        fig, ax = plt.subplots(figsize=(7.2,4.4), layout='constrained')
        bars = ax.bar(labels, book[metric], color=['#aaaaaa','#777777','#333333'], width=.6)
        ax.bar_label(bars, labels=[f'{x:.6f}' for x in book[metric]], padding=4)
        ax.set(ylabel=metric.upper(), title=f'Book Test {metric.upper()} (support-adapted)', ylim=(0,book[metric].max()*1.16))
        ax.grid(axis='y', alpha=.2);ax.set_axisbelow(True)
        fig.savefig(out/f'book_test_{metric}_comparison.png', dpi=300);plt.close(fig)
        fig, ax = plt.subplots(figsize=(7.8,4.8), layout='constrained')
        x = np.arange(3)
        for shift, domain, color in [(-.18,'movie','#aaaaaa'),(.18,'book','#444444')]:
            values = paired[metric+'_'+domain]
            bars = ax.bar(x+shift, values, width=.35, label=domain.title(), color=color)
            ax.bar_label(bars, labels=[f'{v:.6f}' for v in values], padding=3, fontsize=8)
        ax.set(xticks=x, xticklabels=labels, ylabel=metric.upper(), title=f'Movie and Book Test {metric.upper()}',
               ylim=(0,max(paired[metric+'_movie'].max(),paired[metric+'_book'].max())*1.22))
        ax.legend(frameon=False,loc='upper left');ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
        fig.savefig(out/f'movie_book_{metric}_comparison.png', dpi=300);plt.close(fig)
    table=['| Model | Movie Test MAE | Movie Test RMSE | Book Test MAE | Book Test RMSE |',
           '|---|---:|---:|---:|---:|']
    for label,row in zip(['Item-based','Content-based','Hybrid (98:2)'],paired.itertuples()):
        table.append(f'| {label} | {row.mae_movie:.6f} | {row.rmse_movie:.6f} | {row.mae_book:.6f} | {row.rmse_book:.6f} |')
    item=book.iloc[0];hybrid=book.iloc[2]
    counts={s:info['data_checks']['book']['splits'][s]['count'] for s in ['train','validation','test']}
    hashes='\n'.join(f'- `{name}`: `{value}`' for name,value in info['input_sha256'].items())
    report=f'''# Ridge 기반 Hybrid의 도서 평가

## 목적과 평가 조건

영화에서 선택한 Item-based K=50, Content-based Ridge alpha=10, Hybrid 비중 98:2를 고정하고 Goodbooks에 적용하였다. 이번 평가는 Book Train을 사용하는 Support-adapted Cross-domain Evaluation이며 Zero-shot Transfer가 아니다. 기존 SharedContentCF·85:15 기반 도서 Hybrid와는 별도 실험이다.

도서 Train은 {counts['train']:,}행, Validation은 {counts['validation']:,}행, Test는 {counts['test']:,}행이다. Book Train으로 Item-based 유사도와 사용자 평균·선호 프로필을 구성하였다. Book Validation은 데이터 정합성 확인에만 사용하고 K·회귀 계수·가중치 선택에는 사용하지 않았다. Book Test는 예측 후 정답 평가에만 사용하였다.

## 모델과 특징 호환성

Item-based는 기존 ItemBasedCF와 공통 scored 함수를 재사용하여 Book Train에서 책 간 코사인 유사도를 계산하였다. Content-based는 저장된 영화 모델 객체가 없어 동일한 영화 Train 프로필과 Validation A {info['movie_training_rows']:,}행으로 팀원의 비음수 Ridge를 다시 학습하였다. 절편은 자유롭게 두었다. 영화 계수 1,027개를 도서 예측 전후 대조하여 변경되지 않았음을 확인하였다. Book 평점으로 Ridge를 학습하지 않았다.

피처 순서는 장르 코사인 유사도, 정규화한 사용자·아이템 Plot 임베딩 원소곱 1,024개, 길이 근접도, 연도 근접도이다. 두 도메인의 임베딩은 모두 1,024차원이며 전처리 코드에는 BAAI/bge-m3 dense 표현을 사용하도록 되어 있다. 다만 NPZ에 모델 가중치 해시나 차원별 생성 이력이 없어 동일한 생성 환경까지 증명하지는 못하였다.

장르 열은 같은 순서의 59개이다. 사용자와 아이템의 같은 열끼리 각 도메인 내부 유사도를 구한다. 열이 같다는 사실이 영화 장르와 도서 장르의 의미가 완전히 같음을 보장하지는 않으며, 임의 장르 매핑을 추가하지 않았다.

연도는 기존 전처리의 공통 범위 1907~2075를 유지하였다. 이상 출판연도도 원본에서 수정하지 않았다. 길이는 영화 상영시간과 도서 페이지 수를 각각 도메인 안에서 정규화한 값의 사용자 선호 근접도로 해석하였다. 분과 페이지를 같은 물리량으로 간주하거나 서로 변환한 것이 아니다. 이 차원 없는 근접도에도 도메인 간 동일한 영향이 있다는 보장은 없으며, 팀원의 기존 전이 정의를 유지한 제한적 적용이다.

## 결과와 검증

Book Test 전체 {counts['test']:,}행을 세 모델에서 동일하게 평가하였다. 분할 내·분할 간 중복, 원본과 프로필의 평점 기록, ID 대응, 예측 범위, CSV 재로딩, 독립 지표 재계산 검증을 통과하였다. 저장된 장르·수치 프로필은 Train에서 전체 재계산하여 비교하고 Plot 프로필은 각 도메인 128명을 표본 검사하였다. 표본 밖 Plot 프로필과 과거 생성 이력까지 입증한 것은 아니다.

도서 예측은 알려진 평점 범위 1~5로 제한하였다. 팀원의 저장 함수는 0.5~5를 사용하지만, 이번에는 도서의 하한을 적용하였으며 원시 Content 예측 중 1 미만은 {info['raw_content_below_one']:,}행, 5 초과는 {info['raw_content_above_five']:,}행이다. Hybrid는 두 예측을 0.98와 0.02로 가중합하였다. 도서 결과를 보고 모델이나 비중을 조정하지 않았다.

기존 동일 조건의 도서 Item-based MAE 0.635693, RMSE 0.862414와 원래 정밀도에서 일치하였다. Hybrid−Item-based 차이는 MAE {hybrid.mae-item.mae:+.6f}, RMSE {hybrid.rmse-item.rmse:+.6f}이다. 음수는 오차 감소, 양수는 오차 증가를 뜻한다. 단순 결합이 두 지표 모두에서 개선을 보장하지는 않는다.

![Book MAE](book_test_mae_comparison.png)

![Book RMSE](book_test_rmse_comparison.png)

![Movie Book MAE](movie_book_mae_comparison.png)

![Movie Book RMSE](movie_book_rmse_comparison.png)

영화와 도서는 사용자·아이템·평점 분포와 지원 정보가 다르다. 따라서 두 데이터셋의 MAE/RMSE 차이가 곧바로 순수한 전이 성능의 우열을 뜻하지 않는다. 기존 전처리는 두 도메인의 장르와 연도 기준을 함께 구성했으므로 영화만으로 모든 변환을 학습한 평가로 설명할 수 없다.

## 실행 및 재현

저장소 루트에서 `python -B scripts/run_book_hybrid_eval.py`를 실행하고, 완료 후 `python -B scripts/plot_book_hybrid_results.py`로 표·그래프·보고서를 생성한다. 동일 버전의 팀원 소스 Git 객체가 없으면 평가 실행에 `--team-source <원본.py>`를 지정한다. 이미 출력 폴더가 있으면 덮어쓰지 않고 중단한다. 새 폴더는 실행기의 `--output-dir`과 시각화기의 `--results-dir`에 동일하게 지정한다.

영화·도서 콘텐츠/평점/프로필 NPZ, 영화 A/B NPZ·인덱스·split_info, 기존 영화 가중치·출처 JSON·지표 CSV가 필요하다. 패키지는 requirements-item-based.txt를 따른다. 기존 모델·공통 예측 모듈과 팀원 원본 함수 구현을 재사용한다. 대용량 전체 예측 CSV는 로컬 보관용이며 GitHub에 추가할 필요가 없다.

팀원 소스 Git blob은 `{info['team_source_git_blob']}`, 소스 SHA-256은 `{info['team_source_sha256']}`이다. 입력 해시는 다음과 같다.

{hashes}

## PPT 13페이지용 요약

{chr(10).join(table)}
'''
    (out/'book_hybrid_comparison.md').write_text(report,encoding='utf-8')
    if not pd.read_csv(out/'movie_book_test_comparison.csv',float_precision='round_trip').equals(paired):
        raise ValueError('Comparison CSV round-trip mismatch')
    print(paired.to_string(index=False,float_format=lambda v:f'{v:.6f}'))


if __name__ == '__main__':
    main()
