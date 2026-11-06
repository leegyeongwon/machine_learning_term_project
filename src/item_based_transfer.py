"""영화 cosine 협업 표현을 장르에서 증류하는 비개인화 전이."""
import json
from copy import copy
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import Ridge
from sklearn.preprocessing import normalize
from src.item_based_transfer_data import GenreCatalog,canonical_genre


class ItemBasedTransfer:
    """같은 타깃 아이템은 사용자 ID와 무관하게 같은 점수 반환."""
    def __init__(self,rank=16,ridge_alpha=10.,n_neighbors=30,teacher_rank=32,random_state=42):
        self.rank=int(rank);self.ridge_alpha=float(ridge_alpha);self.n_neighbors=int(n_neighbors)
        self.teacher_rank=int(teacher_rank);self.random_state=int(random_state)

    def fit(self,train_ratings,source_content=None):
        if source_content is None:raise ValueError('source_content가 필요함. 기존 Hybrid.fit과 자동 호환되지 않음')
        f=train_ratings
        if f.empty or f[['user_id','item_id','rating']].isna().any().any() or not np.isfinite(f.rating).all():raise ValueError('잘못된 Train')
        if not f.rating.between(.5,5).all() or f.duplicated(['user_id','item_id']).any():raise ValueError('평점 범위/중복 오류')
        self.anchor_ids_=np.sort(f.item_id.unique()).astype(np.int64)
        source_index=pd.Index(source_content.ids).get_indexer(self.anchor_ids_)
        if (source_index<0).any():raise ValueError('Train 콘텐츠 누락')
        binary=source_content.binary[source_index]
        self.genre_names_=sorted({canonical_genre(n) for n,v in zip(source_content.names,binary.sum(0)) if v>0})
        self.anchor_genres_=self.transform_genres(GenreCatalog(self.anchor_ids_,source_content.names,binary))
        users=pd.Index(np.sort(f.user_id.unique()))
        row=pd.Index(self.anchor_ids_).get_indexer(f.item_id).astype(np.int32)
        col=users.get_indexer(f.user_id).astype(np.int32)
        matrix=csr_matrix((f.rating.to_numpy(dtype=np.float32),(row,col)),shape=(len(self.anchor_ids_),len(users)))
        matrix=normalize(matrix,norm='l2',axis=1).tocsr()
        maximum=min(self.teacher_rank,min(matrix.shape)-1)
        if maximum<1:raise ValueError('협업 표현 학습에 최소 2개 아이템·사용자 필요')
        svd=TruncatedSVD(n_components=maximum,n_iter=5,random_state=self.random_state)
        self.teacher_raw_=svd.fit_transform(matrix).astype(np.float64)
        self.singular_values_=svd.singular_values_.astype(float)
        self.explained_variance_ratio_=svd.explained_variance_ratio_.astype(float)
        self.teacher_shape_=matrix.shape;self.teacher_nnz_=matrix.nnz
        sample=np.sort(np.random.default_rng(2026).choice(len(self.anchor_ids_),min(128,len(self.anchor_ids_)),replace=False))
        self.teacher_audit_indices_=sample
        self.teacher_audit_cosine_=(matrix[sample]@matrix[sample].T).toarray()
        self.global_mean_=float(np.mean(f.rating.to_numpy(),dtype=np.float64))
        groups=f.groupby('item_id').rating
        self.anchor_means_=groups.mean().reindex(self.anchor_ids_).to_numpy(dtype=float)
        self.anchor_counts_=groups.size().reindex(self.anchor_ids_).to_numpy(dtype=np.int64)
        self.rating_range_=(.5,5.)
        self.bound_namespace_=None
        return self.configure(self.rank,self.ridge_alpha,self.n_neighbors)

    def transform_genres(self,catalog):
        aligned=np.zeros((len(catalog.ids),len(self.genre_names_)),dtype=np.float64)
        lookup={name:i for i,name in enumerate(self.genre_names_)}
        for j,name in enumerate(catalog.names):
            column=lookup.get(canonical_genre(name))
            if column is not None:aligned[:,column]=np.maximum(aligned[:,column],catalog.binary[:,j])
        return normalize(aligned,norm='l2',axis=1)

    def configure(self,rank,alpha,k,permutation_seed=None):
        if rank<1 or rank>self.teacher_raw_.shape[1] or alpha<=0 or k<1:raise ValueError('후보 범위 오류')
        self.rank,self.ridge_alpha,self.n_neighbors=int(rank),float(alpha),int(k)
        self.anchor_vectors_=normalize(self.teacher_raw_[:,:self.rank],norm='l2',axis=1)
        target=self.anchor_vectors_
        if permutation_seed is not None:
            target=target[np.random.default_rng(permutation_seed).permutation(len(target))]
        reg=Ridge(alpha=self.ridge_alpha,fit_intercept=True).fit(self.anchor_genres_,target)
        self.coef_=reg.coef_.copy();self.intercept_=np.asarray(reg.intercept_).copy()
        self.permutation_seed_=permutation_seed
        self.bound_namespace_=None
        return self

    def candidate(self,rank,alpha,k,permutation_seed=None):
        return copy(self).configure(rank,alpha,k,permutation_seed)

    def score_catalog(self,catalog,ks=None,mode='distilled',namespace='book'):
        if namespace not in ['book','movie']:raise ValueError('명시적 movie/book namespace 필요')
        if mode not in ['distilled','genre','global_mean']:raise ValueError('예측 방식 오류')
        ks=list(ks or [self.n_neighbors])
        if any(not isinstance(k,(int,np.integer)) or k<1 for k in ks):raise ValueError('K 오류')
        x=self.transform_genres(catalog)
        mapped=np.linalg.norm(x,axis=1)>0
        if mode=='distilled':
            queries=normalize(x@self.coef_.T+self.intercept_,norm='l2',axis=1)
            anchors=self.anchor_vectors_
        else:queries,anchors=x,self.anchor_genres_
        scores={k:np.full(len(x),self.global_mean_) for k in ks}
        contributors={k:np.zeros(len(x),dtype=int) for k in ks}
        if mode!='global_mean':
            for start in range(0,len(x),256):
                end=min(start+256,len(x))
                similarity=np.maximum(queries[start:end]@anchors.T,0.)
                if namespace=='movie':
                    loc=pd.Index(self.anchor_ids_).get_indexer(catalog.ids[start:end])
                    known=loc>=0
                    similarity[np.flatnonzero(known),loc[known]]=0.
                similarity[~mapped[start:end]]=0.
                order=np.argsort(-similarity,axis=1,kind='stable')[:,:min(max(ks),len(anchors))]
                weights=np.take_along_axis(similarity,order,axis=1)
                weights[weights<=1e-12]=0.
                values=self.anchor_means_[order]
                for k in ks:
                    w=weights[:,:k];denominator=w.sum(1)
                    positive=denominator>1e-12
                    rows=np.flatnonzero(positive)+start
                    scores[k][rows]=(w*values[:,:k]).sum(1)[positive]/denominator[positive]
                    contributors[k][start:end]=(w>0).sum(1)
        low=1. if namespace=='book' else .5
        for k in ks:scores[k]=np.clip(scores[k],low,5.)
        return scores,{'mapped_genre':mapped,'contributors':contributors,'namespace':namespace}

    def bind_target_content(self,catalog,namespace='book'):
        if namespace!='book':raise ValueError('타깃 bind는 book namespace만 허용')
        predictions,details=self.score_catalog(catalog,namespace='book')
        self.bound_ids_=catalog.ids.copy();self.bound_scores_=predictions[self.n_neighbors]
        self.bound_index_=pd.Index(self.bound_ids_);self.bound_namespace_=namespace
        self.bound_details_=details
        return self

    def predict(self,user_id,item_id):
        return float(self.predict_many([user_id],[item_id])[0])

    def predict_many(self,user_ids,item_ids):
        if self.bound_namespace_!='book':raise ValueError('bind_target_content를 먼저 호출')
        if len(user_ids)!=len(item_ids):raise ValueError('ID 길이 불일치')
        # user_id 값은 의도적으로 조회하지 않음
        loc=self.bound_index_.get_indexer(item_ids)
        output=np.full(len(loc),np.clip(self.global_mean_,1.,5.))
        valid=loc>=0;output[valid]=self.bound_scores_[loc[valid]]
        return output

    def recommend(self,user_id,top_n=10):
        if self.bound_namespace_!='book':raise ValueError('타깃 bind 필요')
        if not isinstance(top_n,int) or top_n<1:raise ValueError('top_n 오류')
        return pd.DataFrame({'item_id':self.bound_ids_,'predicted_rating':self.bound_scores_}).sort_values(
            ['predicted_rating','item_id'],ascending=[False,True]).head(top_n).reset_index(drop=True)

    def save(self,path):
        metadata={k:getattr(self,k) for k in ['rank','ridge_alpha','n_neighbors','teacher_rank','random_state','global_mean_','permutation_seed_','teacher_shape_','teacher_nnz_']}
        with Path(path).open('xb') as handle:
            np.savez_compressed(handle,metadata=np.asarray(json.dumps(metadata)),genre_names=np.asarray(self.genre_names_),
                anchor_ids=self.anchor_ids_,anchor_genres=self.anchor_genres_,anchor_means=self.anchor_means_,anchor_counts=self.anchor_counts_,
                teacher_raw=self.teacher_raw_,coef=self.coef_,intercept=self.intercept_,singular_values=self.singular_values_,
                explained_variance_ratio=self.explained_variance_ratio_,audit_indices=self.teacher_audit_indices_,audit_cosine=self.teacher_audit_cosine_)

    @classmethod
    def load(cls,path):
        with np.load(path,allow_pickle=False) as z:
            meta=json.loads(str(z['metadata']))
            model=cls(meta['rank'],meta['ridge_alpha'],meta['n_neighbors'],meta['teacher_rank'],meta['random_state'])
            for k,v in meta.items():setattr(model,k,v)
            for target,key in [('anchor_ids_','anchor_ids'),('anchor_genres_','anchor_genres'),('anchor_means_','anchor_means'),('anchor_counts_','anchor_counts'),
                ('teacher_raw_','teacher_raw'),('coef_','coef'),('intercept_','intercept'),('singular_values_','singular_values'),
                ('explained_variance_ratio_','explained_variance_ratio'),('teacher_audit_indices_','audit_indices'),('teacher_audit_cosine_','audit_cosine')]:setattr(model,target,z[key].copy())
            model.genre_names_=z['genre_names'].tolist()
        model.anchor_vectors_=normalize(model.teacher_raw_[:,:model.rank],norm='l2',axis=1)
        model.rating_range_=(.5,5.);model.bound_namespace_=None
        return model
