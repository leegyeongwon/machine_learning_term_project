"""Strict Item 전이용 NPZ 접근 통제 및 원본 보존 도구."""
import hashlib
import json
from pathlib import Path
import re
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix


def sha256(path):
    with Path(path).open('rb') as handle:return hashlib.file_digest(handle,'sha256').hexdigest()


def save_json(path,value):
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,default=lambda x:x.item() if isinstance(x,np.generic) else str(x)),encoding='utf-8')


def canonical_genre(name):
    return re.sub(r'[\s_-]+','',str(name).casefold())


class GenreCatalog:
    def __init__(self,ids,names,binary):
        self.ids=np.asarray(ids,dtype=np.int64)
        self.names=np.asarray(names,dtype=str)
        self.binary=np.asarray(binary,dtype=np.float64)
        if len(np.unique(self.ids))!=len(self.ids) or self.binary.shape!=(len(self.ids),len(self.names)):
            raise ValueError('콘텐츠 ID/차원 오류')
        if not np.isfinite(self.binary).all() or not np.isin(self.binary,[0,1]).all():raise ValueError('이진 장르 필요')


class StrictArrayReader:
    """타깃 정답은 모델·예측 파일 해시 봉인 후에만 읽음."""
    def __init__(self,processed_dir):
        self.directory=Path(processed_dir)
        self.events=[]
        self.phase='source_only'
        self.seal=None

    def _allowed(self,file,key):
        movie_parts={f'{s}_{k}' for s in ['train','validation'] for k in ['user_ids','movie_ids','ratings']}
        genres={'genre_names','genres_data','genres_indices','genres_indptr','genres_shape'}
        if file=='movie_item_user_data.npz':return key in movie_parts
        if file=='movie_content.npz':return key in genres|{'movie_ids'}
        if file=='book_content.npz':return self.phase in ['target_features','evaluation'] and key in genres|{'book_ids'}
        if file=='book_item_user_data.npz':
            if key in ['test_user_ids','test_book_ids']:return self.phase in ['target_features','evaluation']
            if key=='test_ratings':return self.phase=='evaluation' and self.seal is not None
        return False

    def read(self,file,key):
        allowed=self._allowed(file,key)
        self.events.append({'index':len(self.events),'phase':self.phase,'file':file,'key':key,'allowed':allowed})
        if not allowed:raise PermissionError(f'금지된 배열 접근: {file}:{key} ({self.phase})')
        if key=='test_ratings':self.verify_seal()
        with np.load(self.directory/file,allow_pickle=False) as data:return data[key].copy()

    def source_split(self,name):
        if name not in ['train','validation']:raise PermissionError('영화 Test 사용 금지')
        f=pd.DataFrame({'user_id':self.read('movie_item_user_data.npz',name+'_user_ids'),
                        'item_id':self.read('movie_item_user_data.npz',name+'_movie_ids'),
                        'rating':self.read('movie_item_user_data.npz',name+'_ratings')})
        if f.empty or f.isna().any().any() or not np.isfinite(f.rating).all() or not f.rating.between(.5,5).all():
            raise ValueError('영화 원점수 오류')
        if f.duplicated(['user_id','item_id']).any():raise ValueError('영화 평점 쌍 중복')
        return f

    def catalog(self,domain):
        if domain not in ['movie','book']:raise ValueError('도메인 오류')
        file=domain+'_content.npz'
        ids=self.read(file,domain+'_ids');names=self.read(file,'genre_names')
        g=csr_matrix((self.read(file,'genres_data'),self.read(file,'genres_indices'),self.read(file,'genres_indptr')),
                     shape=tuple(self.read(file,'genres_shape')))
        g.check_format(full_check=True)
        return GenreCatalog(ids,names,(g.toarray()>0).astype(float))

    def allow_target_features(self,selection_paths):
        if self.phase!='source_only':raise RuntimeError('source 동결은 한 번만 수행')
        self.selection_hashes={str(Path(p).resolve()):sha256(p) for p in selection_paths}
        self.phase='target_features'

    def seal_predictions(self,paths):
        if self.phase!='target_features':raise RuntimeError('타깃 예측 단계 필요')
        self.seal={**self.selection_hashes,**{str(Path(p).resolve()):sha256(p) for p in paths}}
        self.verify_seal()
        self.phase='evaluation'
        return dict(self.seal)

    def verify_seal(self):
        if self.seal is None:raise PermissionError('예측 봉인 필요')
        for name,expected in self.seal.items():
            if sha256(name)!=expected:raise PermissionError('동결 파일 변경: '+name)


def capture_originals(root,exclude):
    root,exclude=Path(root).resolve(),Path(exclude).resolve()
    paths=[]
    for name in ['src','scripts','tests','docs','preprocessing_and_data','data','results','local_recovery']:
        for p in (root/name).rglob('*'):
            if p.is_file() and '__pycache__' not in p.parts and not p.resolve().is_relative_to(exclude):paths.append(p)
    paths += [p for p in root.iterdir() if p.is_file()]
    return [{'path':str(p.relative_to(root)),'bytes':p.stat().st_size,'mtime_ns':p.stat().st_mtime_ns,'sha256':sha256(p)} for p in sorted(set(paths))]


def verify_originals(root,records):
    changed=[]
    for record in records:
        p=Path(root)/record['path']
        if not p.exists() or p.stat().st_size!=record['bytes'] or p.stat().st_mtime_ns!=record['mtime_ns'] or sha256(p)!=record['sha256']:
            changed.append(record['path'])
    return {'checked_files':len(records),'changed_files':changed,'all_preserved':not changed}
