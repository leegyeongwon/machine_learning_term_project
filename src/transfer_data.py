"""공통 콘텐츠 및 Train/support 전용 프로필 어댑터."""
from copy import copy
import re
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
from src.content_based import ContentBasedCF, _RidgeModel
from src.item_based import validate_ratings


def canonical_genre(value):
    """대소문자·공백·하이픈만 통일. 의미 기반 임의 매핑 없음."""
    return re.sub(r'[\s_-]+', '', str(value).casefold())


def load_splits(path, domain='movie', validate=True):
    with np.load(path, allow_pickle=False) as z:
        parts = {s: pd.DataFrame({'user_id': z[s+'_user_ids'], 'item_id': z[s+'_'+domain+'_ids'],
                                 'rating': z[s+'_ratings']}) for s in ('train','validation','test')}
    if validate: validate_splits(parts)
    return parts


def validate_splits(parts):
    """쌍 키를 압축하여 전체 triple 복사 없이 중복 검사."""
    max_item=max(int(f.item_id.max()) for f in parts.values())+1
    keys=[]
    for f in parts.values():
        validate_ratings(f)
        if not f.rating.between(.5,5).all() or (f.user_id<0).any() or (f.item_id<0).any():
            raise ValueError('평점 또는 ID 범위 오류')
        keys.append(f.user_id.to_numpy(dtype=np.uint64)*np.uint64(max_item)+f.item_id.to_numpy(dtype=np.uint64))
    all_keys=np.concatenate(keys)
    if len(np.unique(all_keys))!=len(all_keys):raise ValueError('분할 간 중복')
    count=len(all_keys)
    if np.max(np.abs(np.array([len(f) for f in parts.values()])-np.array([.8,.1,.1])*count))>2:
        raise ValueError('8:1:1 비율 불일치')
    return {s:{'count':len(f),'ratio':len(f)/count,'users':int(f.user_id.nunique()),'items':int(f.item_id.nunique()),
               'cold_users':int(f.loc[~f.user_id.isin(parts['train'].user_id),'user_id'].nunique()),
               'cold_items':int(f.loc[~f.item_id.isin(parts['train'].item_id),'item_id'].nunique()),
               'cold_user_rows':int((~f.user_id.isin(parts['train'].user_id)).sum()),
               'cold_item_rows':int((~f.item_id.isin(parts['train'].item_id)).sum()),
               'rating_min':float(f.rating.min()),'rating_max':float(f.rating.max())} for s,f in parts.items()}


def load_content(path, domain, include_plot=False):
    with np.load(path, allow_pickle=False) as z:
        ids=z[domain+'_ids'].copy()
        names=z['genre_names'].astype(str)
        g=csr_matrix((z['genres_data'],z['genres_indices'],z['genres_indptr']),shape=tuple(z['genres_shape'])).toarray()
        num_names=z['numeric_feature_names'].astype(str).tolist()
        target='release_year' if domain=='movie' else 'publish_year'
        year=z['numeric_features'][:,num_names.index(target)].astype(float)*(float(z['year_max'])-float(z['year_min']))+float(z['year_min'])
        if 'year_missing' in z:year[z['year_missing'].astype(bool)]=np.nan
        plot=z['plot_embeddings'].copy() if include_plot else None
    if len(set(ids))!=len(ids) or g.shape!=(len(ids),len(names)) or not np.isfinite(g).all():
        raise ValueError('콘텐츠 ID/행렬 오류')
    if include_plot and (plot.shape[0]!=len(ids) or not np.isfinite(plot).all()):raise ValueError('임베딩 오류')
    result=(ids,names,(g>0).astype(float),year)
    return result+(plot,) if include_plot else result


def normalized_rows(values):
    return values/np.maximum(np.linalg.norm(values,axis=1,keepdims=True),1e-12)


class _SeenItems:
    def __init__(self,matrix,user_index,item_ids):
        self.matrix,self.users,self.items=matrix,user_index,item_ids

    def get(self,user_id,default=None):
        row=self.users.get(int(user_id),-1)
        return set(self.items[self.matrix[row].indices]) if row>=0 else (default if default is not None else set())


class SharedContentCF(ContentBasedCF):
    """기존 유저 평균+콘텐츠 Ridge를 희소 프로필·배치 통계로 학습."""
    def __init__(self,content,ridge_alpha=10,plot_components=32):
        super().__init__(ridge_alpha=ridge_alpha)
        self.content,self.plot_components=content,plot_components

    def _load_item_content(self):
        self._reset_extra_item_features()
        ids,names,g,year=self.content[:4]
        self.item_ids_=ids
        self.genre_cols_=sorted({canonical_genre(n) for n,v in zip(names,g.sum(0)) if v>0})
        self.item_multihot_=self._align_genres(names,g)
        self.item_index_={int(v):i for i,v in enumerate(ids)}
        self.titles_=np.full(len(ids),'')
        valid=year[np.isfinite(year)]
        self.year_mean_=float(valid.mean()) if len(valid) else 0.
        self.year_std_=float(valid.std()) if len(valid) and valid.std()>0 else 1.
        self.year_norm_=(np.where(np.isfinite(year),year,self.year_mean_)-self.year_mean_)/self.year_std_
        self.idf_=np.log((len(ids)+1)/(self.item_multihot_.sum(0)+1))+1
        self.item_tfidf_=normalized_rows(self.item_multihot_*self.idf_)
        self.plot_projection_=None

    def _align_genres(self,names,g):
        aligned=np.zeros((len(g),len(self.genre_cols_)))
        lookup={n:i for i,n in enumerate(self.genre_cols_)}
        for j,n in enumerate(names):
            k=lookup.get(canonical_genre(n))
            if k is not None:aligned[:,k]=np.maximum(aligned[:,k],g[:,j])
        return aligned

    def _support_profiles(self,support):
        """평점별 feature 펼침 없이 CSR 곱으로 잔차 취향 계산."""
        validate_ratings(support)
        if not support.item_id.isin(self.item_ids_).all():raise ValueError('지원 콘텐츠 누락')
        self.user_means_=support.groupby('user_id').rating.mean().to_dict()
        self.item_means_=support.groupby('item_id').rating.mean().to_dict()
        self.user_ids_=np.array(sorted(self.user_means_))
        self.user_index_={int(v):i for i,v in enumerate(self.user_ids_)}
        self.user_mean_arr_=np.array([self.user_means_[u] for u in self.user_ids_])
        rows=pd.Index(self.user_ids_).get_indexer(support.user_id)
        cols=pd.Index(self.item_ids_).get_indexer(support.item_id)
        residual=support.rating.to_numpy(dtype=float)-self.user_mean_arr_[rows]
        matrix=csr_matrix((residual,(rows,cols)),shape=(len(self.user_ids_),len(self.item_ids_)))
        self.user_profiles_=normalized_rows(matrix@self.item_tfidf_)
        self.user_plot_=normalized_rows((matrix@self.item_plot_).astype(np.float32)) if self.item_plot_ is not None else None
        self.seen_=_SeenItems(matrix,self.user_index_,self.item_ids_)
        self.has_demo_=False
        return rows,cols,residual

    def fit(self,train_ratings):
        self._load_item_content()
        if not train_ratings.item_id.isin(self.item_ids_).all():raise ValueError('학습 콘텐츠 누락')
        if len(self.content)>4 and self.content[4].shape[1]:
            plot=normalized_rows(self.content[4].astype(np.float32))
            observed=pd.Index(self.item_ids_).get_indexer(train_ratings.item_id.unique())
            self.plot_projection_=TruncatedSVD(n_components=min(self.plot_components,len(observed)-1,plot.shape[1]-1),random_state=42)
            self.plot_projection_.fit(plot[observed])
            self.item_plot_=normalized_rows(self.plot_projection_.transform(plot).astype(np.float32))
        self.global_mean_=float(train_ratings.rating.mean())
        self.rating_range_=(float(train_ratings.rating.min()),float(train_ratings.rating.max()))
        rows,cols,y=self._support_profiles(train_ratings)
        dim=self._features_from_indices(rows[:1],cols[:1]).shape[1]+1
        self.xtx_=np.zeros((dim,dim));self.xty_=np.zeros(dim)
        for start in range(0,len(y),8192):
            sl=slice(start,start+8192)
            x=self._features_from_indices(rows[sl],cols[sl]).astype(float)
            augmented=np.column_stack([np.ones(len(x)),x])
            self.xtx_+=augmented.T@augmented
            self.xty_+=augmented.T@y[sl]
        self.feature_names_=(['genre:'+n for n in self.genre_cols_]+['year_z']+
                             ['genre_preference:'+n for n in self.genre_cols_])
        if self.item_plot_ is not None:
            self.feature_names_+=['plot_svd:'+str(j) for j in range(self.item_plot_.shape[1])]+['plot_preference_cosine']
        return self.set_alpha(self.ridge_alpha)

    def set_alpha(self,alpha):
        self.ridge_alpha=float(alpha)
        gram=self.xtx_.copy()
        gram.flat[::len(gram)+1]+=alpha
        gram[0,0]-=alpha
        weights=np.linalg.solve(gram,self.xty_)
        self.ridge_=_RidgeModel(weights[1:],weights[0])
        return self

    def transfer(self,content,support):
        """영화 콘텐츠 변환·Ridge를 고정하고 도서 지원 프로필만 계산."""
        ids,names,g,year=content[:4]
        m=copy(self)
        m.content=content
        m.item_ids_,m.item_index_=ids,{int(v):i for i,v in enumerate(ids)}
        m.item_multihot_=m._align_genres(names,g)
        m.item_tfidf_=normalized_rows(m.item_multihot_*self.idf_)
        m.year_norm_=(np.where(np.isfinite(year),year,self.year_mean_)-self.year_mean_)/self.year_std_
        if self.plot_projection_ is not None:
            if len(content)<5 or content[4].shape[1]!=self.plot_projection_.n_features_in_:
                raise ValueError('영화·도서 임베딩 차원 불일치')
            m.item_plot_=normalized_rows(self.plot_projection_.transform(normalized_rows(content[4].astype(np.float32))).astype(np.float32))
        m._support_profiles(support)
        m.rating_range_=(1.,5.)
        return m

    def predict_many(self,user_ids,item_ids):
        if len(user_ids)!=len(item_ids):raise ValueError('ID 길이 불일치')
        users,items=np.asarray(user_ids),np.asarray(item_ids)
        p=np.empty(len(users));methods=np.empty(len(users),dtype=object);counts=np.empty(len(users),dtype=int)
        for start in range(0,len(users),8192):
            sl=slice(start,start+8192)
            p[sl],methods[sl],counts[sl]=super().predict_many(users[sl],items[sl])
        return p,methods,counts
