import pandas as pd
import numpy as np
import re

# 연도 패턴 (괄호 안의 4자리 연도)
YEAR_RE = re.compile(r"\((\d{4})\)")

def parse_year(title):
    """제목에서 개봉연도를 뽑는다. 없으면 NaN."""
    match = YEAR_RE.search(str(title))
    return float(match.group(1)) if match else np.nan


# CSV 읽기
df = pd.read_csv("movies.csv")

# 제목에서 연도 추출
df["release_year"] = df["title"].apply(parse_year)

# 중복 제거 후 정렬하여 출력
years = sorted(df["release_year"].dropna().unique())

print("추출된 연도 목록:")
print(years)

print(f"\n총 연도 종류: {len(years)}")
print(f"연도 추출 실패 영화 수: {df['release_year'].isna().sum()}")