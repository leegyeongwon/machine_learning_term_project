"""NPZ 전용 분할·매핑·Train 통계·저장 프로필 감사."""
import hashlib
import zipfile
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
from src.common_item_data import restore_csr


def sha256(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def npz_schema(path):
    result={}
    with zipfile.ZipFile(path) as z:
        for name in z.namelist():
            with z.open(name) as h:
                version=np.lib.format.read_magic(h)
                shape,order,dtype=(np.lib.format.read_array_header_1_0(h) if version==(1,0) else np.lib.format.read_array_header_2_0(h))
            result[name[:-4]]={'shape':list(shape),'dtype':str(dtype)}
    return result


def sampled_array(path,key,rows):
    """대형 압축 배열을 스트리밍하며 일부 행만 선택."""
    pieces=[]
    with zipfile.ZipFile(path) as z,z.open(key+'.npy') as f:
        version=np.lib.format.read_magic(f)
        shape,order,dtype=(np.lib.format.read_array_header_1_0(f) if version==(1,0) else np.lib.format.read_array_header_2_0(f))
        if order or len(shape)!=2:raise ValueError('행 우선 2차원 배열 필요')
        for start in range(0,shape[0],4096):
            count=min(4096,shape[0]-start)
            block=np.frombuffer(f.read(count*shape[1]*dtype.itemsize),dtype=dtype).reshape(count,shape[1])
            keep=rows[(rows>=start)&(rows<start+count)]-start
            if len(keep):pieces.append(block[keep].copy())
    return np.concatenate(pieces)


def validate_processed(directory,parts,content,domain='movie'):
    """두 모델 NPZ의 분할은 dtype 차이를 허용하고 값·순서를 비교."""
    cf_path=directory/(domain+'_item_user_data.npz')
    profile_path=directory/(domain+'_user_profiles.npz')
    content_path=directory/(domain+'_content.npz')
    ids,names,g,years=content[:4]
    with np.load(cf_path,allow_pickle=False) as cf,np.load(profile_path,allow_pickle=False) as profiles,np.load(content_path,allow_pickle=False) as meta:
        if int(cf['random_state'])!=42 or int(profiles['random_state'])!=42:raise ValueError('seed 불일치')
        for s,frame in parts.items():
            for key,col in [('user_ids','user_id'),(domain+'_ids','item_id'),('ratings','rating')]:
                np.testing.assert_array_equal(frame[col].to_numpy(),profiles[s+'_'+key])
            if domain=='movie':
                movie_index=pd.Index(meta['movie_ids'])
                loc=movie_index.get_indexer(frame.item_id)
                if (loc<0).any():raise ValueError('MovieLens 콘텐츠 ID 누락')
                np.testing.assert_array_equal(meta['tmdb_ids'][loc],cf[s+'_tmdb_ids'])
                np.testing.assert_array_equal(cf[s+'_tmdb_ids'],profiles[s+'_tmdb_ids'])
        users=pd.Index(cf['user_ids']);items=pd.Index(cf[domain+'_ids'])
        np.testing.assert_array_equal(items.to_numpy(),ids)
        np.testing.assert_array_equal(users.to_numpy(),profiles['user_ids'])
        np.testing.assert_array_equal(names,profiles['genre_names'])
        if domain=='movie':np.testing.assert_array_equal(cf['tmdb_ids'],meta['tmdb_ids'])
        train=parts['train'];r=users.get_indexer(train.user_id);c=items.get_indexer(train.item_id)
        if (r<0).any() or (c<0).any():raise ValueError('Train ID 누락')
        values=train.rating.to_numpy(dtype=float)
        counts=np.bincount(r,minlength=len(users))
        means=np.divide(np.bincount(r,weights=values,minlength=len(users)),counts,out=np.zeros(len(users)),where=counts>0)
        centered=values-means[r]
        stds=np.sqrt(np.divide(np.bincount(r,weights=centered**2,minlength=len(users)),counts,out=np.zeros(len(users)),where=counts>0))
        stds[stds==0]=1
        np.testing.assert_allclose(cf['user_means'],means,atol=1e-8,rtol=1e-8)
        np.testing.assert_allclose(cf['user_stds'],stds,atol=1e-8,rtol=1e-8)
        mask=restore_csr(cf,'mask');normalized=restore_csr(cf,'rating')
        if mask.shape!=(len(users),len(items)) or mask.nnz!=len(train) or not np.all(mask.data==1):raise ValueError('Train 마스크 오류')
        max_error=0.
        for start in range(0,len(r),200000):
            sl=slice(start,start+200000)
            if not np.asarray(mask[r[sl],c[sl]]).all():raise ValueError('Train 마스크 누락')
            actual=np.asarray(normalized[r[sl],c[sl]]).ravel()
            expected=(centered[sl]/stds[r[sl]]).astype(np.float32)
            np.testing.assert_allclose(actual,expected,atol=1e-6,rtol=1e-5)
            max_error=max(max_error,float(np.abs(actual-expected).max()))
        if (normalized-normalized.multiply(mask)).nnz:raise ValueError('Train 외 정규화 값')
        for s in ['validation','test']:
            rr=users.get_indexer(parts[s].user_id);cc=items.get_indexer(parts[s].item_id)
            if (rr<0).any() or (cc<0).any() or np.asarray(mask[rr,cc]).any():raise ValueError('Holdout 마스크 오류')
        # 팀 원문의 저장 프로필은 (rating-3) 및 max(rating-3,0) 사용
        adjusted=csr_matrix(((values-3).astype(np.float32),(r,c)),shape=mask.shape)
        preference=adjusted.maximum(0).tocsr()
        genres=restore_csr(meta,'genres')
        expected_genre=normalize(preference@genres,norm='l2',axis=1).tocsr()
        stored_genre=restore_csr(profiles,'user_genre_profiles')
        difference=expected_genre-stored_genre
        genre_error=float(np.abs(difference.data).max()) if difference.nnz else 0.
        if genre_error>2e-6:raise ValueError('저장 장르 프로필이 Train 계산과 다름')
        numeric=meta['numeric_features'];numeric_expected=np.zeros((len(users),2),dtype=np.float32)
        first_missing='runtime_missing' if domain=='movie' else 'pages_missing'
        for j,missing_key in enumerate([first_missing,'year_missing']):
            observed=~meta[missing_key].astype(bool)
            numerator=np.asarray(preference@(numeric[:,j]*observed)).ravel()
            denominator=np.asarray(preference@observed.astype(np.float32)).ravel()
            numeric_expected[:,j]=np.divide(numerator,denominator,out=np.full(len(users),.5,dtype=np.float32),where=denominator>0)
        numeric_error=float(np.max(np.abs(numeric_expected-profiles['numeric_profiles'])))
        # float32 희소 누적 순서·공통 연도 재정규화의 반올림 오차 허용
        np.testing.assert_allclose(numeric_expected,profiles['numeric_profiles'],atol=1e-5,rtol=1e-5)
        sample_rows=np.unique(np.linspace(0,len(users)-1,min(64,len(users)),dtype=int))
        expected_plot=normalize(adjusted[sample_rows]@meta['plot_embeddings'],norm='l2',axis=1)
        sampled_plot=sampled_array(profile_path,'plot_profiles',sample_rows)
        np.testing.assert_allclose(expected_plot,sampled_plot,atol=2e-5,rtol=2e-4)
        tmdb_unique=int(len(np.unique(meta['tmdb_ids']))) if domain=='movie' else None
    report={'dataset_origin':domain,'matched_ratings':sum(map(len,parts.values())),'users':len(users),'items':len(items),
            'rating_arrays_identical_across_models':True,'rating_dtype_difference_allowed':True,
            'source_raw_ratings_count':None,'raw_source_status':'원본 CSV 없음: 원본 총수·외부 링크 매핑은 독립 검증 불가',
            'npz_movie_tmdb_mapping_consistent':domain=='movie','unique_tmdb_ids':tmdb_unique,
            'train_only_means_stds_mask_zscore_verified':True,'zscore_max_abs_error':max_error,
            'stored_genre_profiles_train_only_verified':True,'stored_numeric_profiles_train_only_verified':True,
            'stored_plot_profiles_train_only_sample_rows':len(sample_rows),
            'stored_profiles_used_for_model':False,'genre_profile_max_abs_error':genre_error,
            'numeric_profile_max_abs_error':numeric_error,'numeric_profile_atol':1e-5,
            'year_min':float(np.nanmin(years)),'year_max':float(np.nanmax(years)),
            'missing_years':int((~np.isfinite(years)).sum()),
            'abnormal_years':[{'item_id':int(i),'year':float(y)} for i,y in zip(ids,years) if np.isfinite(y) and (y>2026 or y<=0)]}
    if domain=='movie':
        # 영화 원문의 사용자·영화 정렬 순서를 NPZ 쌍 집합에서 복원
        radix=int(ids.max())+1
        keys={s:f.user_id.to_numpy(dtype=np.uint64)*np.uint64(radix)+f.item_id.to_numpy(dtype=np.uint64) for s,f in parts.items()}
        sorted_keys=np.sort(np.concatenate(list(keys.values())))
        expected_train,temp=train_test_split(sorted_keys,test_size=.2,random_state=42)
        expected_val,expected_test=train_test_split(temp,test_size=.5,random_state=42)
        for s,expected in zip(parts,[expected_train,expected_val,expected_test]):np.testing.assert_array_equal(keys[s],expected)
        report['seed42_split_replayed_from_npz_pairs']=True
    else:report['seed42_split_replay']='원본 도서 행 순서 부재로 미확인; 저장 seed와 두 모델 분할 일치 검증'
    return report
