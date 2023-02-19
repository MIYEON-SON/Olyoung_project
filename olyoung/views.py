from django.shortcuts import render
from sklearn.metrics.pairwise import cosine_similarity
#!pip install json
import json
import pandas as pd
import numpy as np
from scipy import stats

def index(request):
    return render(request, 'olyoung/index.html')

def ver1(request):
    username=request.GET.get('username')
    return render(request, 'olyoung/ver1.html',{'username':username})

def ver1_1(request):
    #경로 C:\Users\dkiso\Desktop\asiae_bigdata\final\myproject
    skin_df1=pd.read_csv('skin_df.csv',encoding='cp949',index_col=0)
    cat_df=pd.read_csv('cat_df.csv',index_col=0)
    
    topic=[['트러블','수분','세정력','보습','메이크업','건성','진정''저자극','세안','화장'],
    ['흡수','트러블','쿨링','진정','저자극','여드름','수분','보습','모공','지성'],
    ['피지','저자극','세정력','보습','거품','각질','향','트러블','냄새','건조'],
    ['흡수','트러블','진정','지성','저자극','유분','수분','보습','건조','피지'],
    ['피지','트러블','지성','저자극','여드름','약산성','세정력','모공','건조','거품']]  

    # 확인용
    #username=request.GET.get('username')
    
    # 선택값 받아오기
    skintype=request.GET.get('skintype_option')
    skinconcern=request.GET.getlist('skin_concern')
    category=request.GET.get('category')
 
    c1_score = (float((cat_df.loc[category,'군집1']))/cat_df.loc[category].values.sum())*(skin_df1.loc[skinconcern,'군집1'].sum()*0.5+float((skin_df1.loc[skintype,'군집1'])*0.5))
    c2_score = (float((cat_df.loc[category,'군집2']))/cat_df.loc[category].values.sum())*(skin_df1.loc[skinconcern,'군집2'].sum()*0.5+float((skin_df1.loc[skintype,'군집2'])*0.5))
    c3_score = (float((cat_df.loc[category,'군집3']))/cat_df.loc[category].values.sum())*(skin_df1.loc[skinconcern,'군집3'].sum()*0.5+float((skin_df1.loc[skintype,'군집3'])*0.5))
    c4_score = (float((cat_df.loc[category,'군집4']))/cat_df.loc[category].values.sum())*(skin_df1.loc[skinconcern,'군집4'].sum()*0.5+float((skin_df1.loc[skintype,'군집4'])*0.5))
    c5_score = (float((cat_df.loc[category,'군집5']))/cat_df.loc[category].values.sum())*(skin_df1.loc[skinconcern,'군집5'].sum()*0.5+float((skin_df1.loc[skintype,'군집5'])*0.5))
    scores=[c1_score,c2_score,c3_score,c4_score,c5_score]
    topic_num=scores.index(max(scores))
    keywords=topic[topic_num]
    
    
    #     # 넘겨줄 데이터 Json 변환
    #     targetdict = {
    #         'skintype1' : skintype,
    #     }
    #     targetJson = json.dumps(targetdict)

    #     return render(request, 'olyoung/ver1_1.html',{'skintype1': skintype,'targetJson':targetJson, 'username':username, 'username2':username2})
    
    # else:
    return render(request, 'olyoung/ver1_1.html', {'keywords':keywords,'skintype':skintype, 'skinconcern':skinconcern, 'category':category, 'topic_num':topic_num})

def ver1_result(request):

    tp1_keywords_df=pd.read_csv('topic1_제품별_키워드_빈도수표.csv',encoding='cp949',index_col='product_id')
    tp2_keywords_df=pd.read_csv('topic2_제품별_키워드_빈도수표.csv',encoding='cp949',index_col='product_id')
    tp3_keywords_df=pd.read_csv('topic3_제품별_키워드_빈도수표.csv',encoding='cp949',index_col='product_id')
    tp4_keywords_df=pd.read_csv('topic4_제품별_키워드_빈도수표.csv',encoding='cp949',index_col='product_id')
    tp5_keywords_df=pd.read_csv('topic5_제품별_키워드_빈도수표.csv',encoding='cp949',index_col='product_id')
    total_keywords_df=pd.concat([tp1_keywords_df,tp2_keywords_df,tp3_keywords_df,tp4_keywords_df,tp5_keywords_df],axis=0)
    total_keywords_df=total_keywords_df.fillna(0)
    total_keywords_df=total_keywords_df.replace('스킨케어','skincare').replace('클렌징','cleansing').replace('바디케어','bodycare').replace('마스크팩','maskpack').replace('선케어','suncare')
    total_keywords_df=total_keywords_df.iloc[:,1:]
    
    # 추천대상
    i_df = pd.read_csv('추천시스템전처리.csv', encoding='utf8', index_col=0)
    a=total_keywords_df.index
    b=i_df.loc[:,'product_id'].to_list()
    test=a.intersection(b)
    total_keywords_df=total_keywords_df.loc[test]

    # 선택값 받아오기
    keywords2=request.GET.getlist('keywords')
    skintype=request.GET.get('skintype')
    skinconcern1=request.GET.get('skinconcern1')
    skinconcern2=request.GET.get('skinconcern2')
    category=request.GET.get('category')
    topic_num=request.GET.get('topic_num')

    # 카테고리 필터링
    df_f1=total_keywords_df[total_keywords_df.loc[:,'속성_num']==int(topic_num)+1]
    df_f1=df_f1[df_f1.loc[:,'카테고리']==category]

    # 키워드 필터링 
    # 키워드 둘중 하나만 값이 높은 경우 뽑히게 되는 문제-> 한키워드의 비율이 90% z값 이상인 제품은 제외됨
    df_f1_r=df_f1[keywords2].div(df_f1[keywords2].sum(axis=1), axis=0).max(axis=1)
    # 90% z-value 
    z90_value=stats.norm(df_f1_r.mean(),df_f1_r.std()).ppf(0.90)
    df_f2=df_f1.loc[df_f1_r[df_f1_r<z90_value].index,:]
    df_f2_i=list(df_f2[keywords2].sum(axis=1).sort_values(ascending=False).index[0:10])
    df_f2=df_f1.loc[df_f2_i]
      
    
    # 스킨케어, 선케어, 클렌징 카테고리를 선택한 경우 피부타입 만족도 필터링
    if skintype=='건성':
     skintype_var='피부타입_건성에 좋아요'
    elif skintype=='복합성': skintype_var='피부타입_복합성에 좋아요'
    else: skintype_var='피부타입_지성에 좋아요'
    
    df_f3=df_f2[skintype_var].sort_values(ascending=False)
    df_f3_i=list(df_f3.index[0:5])
    df_f3=df_f2.loc[df_f3_i]
    
    # 평점 순 상위 3개 선정
    df_f4=df_f3.loc[:,'평점'].sort_values(ascending=False)
    i=list(df_f4.index[0:3])

    product_df=pd.read_csv('product_df2.csv',encoding='cp949',index_col='product_id')
    product_df=product_df.iloc[:,:8]

    keyword_df=pd.read_csv('total_keyword.csv',index_col='product_id')
    keyword_df=pd.DataFrame(keyword_df.loc[:,'keyword'])
    keyword_df=pd.DataFrame(keyword_df['keyword'].str.replace('[','').replace(']',''))
    keyword_df=pd.DataFrame(keyword_df['keyword'].str.replace(']',''))

    category_f=product_df.loc[i,'category2'].to_list()
    product_name_f=product_df.loc[i,'product_name'].to_list()
    price_f=product_df.loc[i,'prices'].to_list()
    imgpath_f=product_df.loc[i,'imgpaths'].to_list()
    keyword_f=keyword_df.loc[i,'keyword'].to_list()
    
    ## 제품별 속성 평점 부여
    
    # 제품별 사용고객 피부타입 비율
    # 해석1: 동일한 피부타입을 가진 고객들은 해당제품에 00% 만족했어요
    # 해석2: 해당 제품은 고객 피부타입에 00% 잘 맞아요
    p1_skin_rate=total_keywords_df.loc[i[0],skintype_var]
    p2_skin_rate=total_keywords_df.loc[i[1],skintype_var]
    p3_skin_rate=total_keywords_df.loc[i[2],skintype_var]
    
    p1_skin_rate2='%d'%(round(p1_skin_rate*100))
    p2_skin_rate2='%d'%(round(p2_skin_rate*100))
    p3_skin_rate2='%d'%(round(p3_skin_rate*100))
    p_skin_rate=[p1_skin_rate2,p2_skin_rate2,p3_skin_rate2]
 
    # 제품별 리뷰 개수
    revcnt_df = pd.read_csv('rev_cnt.csv', encoding='utf8', index_col=0)
    total_keywords_df=pd.merge(total_keywords_df,revcnt_df, how='left',left_index=True, right_index=True)
    p_rev_cnt=list(total_keywords_df.loc[i,'review'].values)
    
    # 제품별 리뷰에서 키워드 빈도수 비율
    # 해석1: 해당 제품 리뷰에서 키워드A 가 언급된 비율은 00% 에요
    p1_kw_fr=total_keywords_df.loc[i[0],keywords2].values
    p1_kw_fr=p1_kw_fr/p_rev_cnt[0]
    p2_kw_fr=total_keywords_df.loc[i[1],keywords2].values
    p2_kw_fr=p2_kw_fr/p_rev_cnt[1]
    p3_kw_fr=total_keywords_df.loc[i[2],keywords2].values
    p3_kw_fr=p3_kw_fr/p_rev_cnt[2]
    p_kw_fr=['%d'%(round(p1_kw_fr[0]*100)),'%d'%(round(p1_kw_fr[1]*100)),'%d'%(round(p2_kw_fr[0]*100)),'%d'%(round(p2_kw_fr[1]*100)),'%d'%(round(p3_kw_fr[0]*100)),'%d'%(round(p3_kw_fr[1]*100))]

    # *군집별* 피부고민 비율
    # 해석1: 해당 제품군(군집)에서 동일한 피부고민을 가진 고객의 비율은 00% 이에요 
    # 해석2: 동일한 피부고민을 가진 고객들은 해당 제품군(군집)을 00% 구매했어요
    topic_num2='군집%d'%(int(topic_num)+1)
    skin_df1=pd.read_csv('skin_df3.csv',encoding='cp949',index_col=0)
    p_con_rate1=skin_df1.loc[skinconcern1,topic_num2]
    if p_con_rate1<0.001:
     p_con_rate1_1=0
    else: p_con_rate1_1=p_con_rate1*100
    
    p_con_rate2=skin_df1.loc[skinconcern2,topic_num2]
    if p_con_rate2<0.001:
     p_con_rate2_1=0
    else: p_con_rate2_1=p_con_rate2*100
    p_con_rate=['%d'%p_con_rate1_1,'%d'%p_con_rate2_1]

    # 제품별 그래프 수치 리스트
    p1_gr_value=[]
    p2_gr_value=[]
    p3_gr_value=[]
    p1_gr_value.extend([p1_skin_rate,p_con_rate1,p_con_rate2,p1_kw_fr[0],p1_kw_fr[1]])
    p2_gr_value.extend([p2_skin_rate,p_con_rate1,p_con_rate2,p2_kw_fr[0],p2_kw_fr[1]])
    p3_gr_value.extend([p3_skin_rate,p_con_rate1,p_con_rate2,p3_kw_fr[0],p3_kw_fr[1]])
    
    # 제품별 평점
    p_score=list(total_keywords_df.loc[i,'평점'].values)

    # 넘겨줄 데이터 Json 변환
    targetdict = {
        'product1':p1_gr_value,
        'product2':p2_gr_value,
        'product3':p3_gr_value
    }

    targetJson = json.dumps(targetdict)

    return render(request, 'olyoung/ver1_result.html',{'targetJson':targetJson,'p_kw_fr':p_kw_fr,'p_con_rate':p_con_rate,'p_skin_rate':p_skin_rate,'keywords2': keywords2,'skintype':skintype,'skinconcern1':skinconcern1,'skinconcern2':skinconcern2,'category':category,'topic_num':topic_num,'category_f':category_f,'product_name_f':product_name_f,'price_f':price_f,'imgpath_f':imgpath_f,'keyword_f':keyword_f,'p_score':p_score})
    
# 가중 평균 함수
def weight_average(df, pct):
    pct = pct
    m = df['review_cnt'].quantile(pct)
    c = df['rev_score'].mean()
    v = df['review_cnt']
    r = df['rev_score']
    weight_average = (v / (v+m)) * r + (m / (v+m)) * c
        
    return weight_average
    
# 가중 평균 기반 추천 시스템 함수
def weight_vote_avg(df, sorted_sim, title = '', num=5):
    title_item = df[df['product_name'] == title]
    title_item_idx = title_item.index.values

    sim_idx = sorted_sim[title_item_idx, :(num)]
    sim_idx = sim_idx.reshape(-1)
    sim_idx = sim_idx[sim_idx != title_item_idx]
    similar_item = df.iloc[sim_idx]
    
    return similar_item.sort_values(by='weight_average', ascending=False)

def ver2_result1(request):
    # 선택값 받아오기
    product_name_f2_1=request.GET.get('product_name_f1')
    
    df = pd.read_csv('추천시스템전처리.csv', encoding='utf8', index_col=0)
    a_df = pd.read_csv('product_df2.csv',encoding='cp949')
    df = pd.merge(df, a_df[['product_id','rev_score','imgpaths','prices']], how='inner', on='product_id')
    df = df.drop_duplicates(subset='product_id', keep='first').reset_index()
    df['keyword'] = df['keyword'].apply(lambda x: str(x).replace("'","").replace('[','').replace(']',''))   
    
    # 전체 키워드 리스트
    keyword = []
    for i in df['keyword']:
        i = i.split(',')
        for j in i:
            keyword.append(j.strip())
    keyword = list(set(keyword))
    
    # 키워드 딕셔너리
    keyword_dic_li = []
    for i, j in zip(range(len(keyword)), keyword):
        keyword_dic_li.append({'id': i, 'name': j})
    
    # 제품별 키워드 딕셔너리
    keyword_list = []
    for names in df['keyword']:
        li = []
        for val in keyword_dic_li: 
            if val['name'] in names:
                li.append(val)
            else:
                continue
        keyword_list.append(li)
    
    df['keyword'] = keyword_list
    df['keyword'] = df['keyword'].apply(lambda x: [i['name'] for i in x])
    
    # 키워드 컬럼 펼치기
    keywords_list = []
    for keyword in df['keyword']:
        keywords_list.extend(keyword)
    keywords_list = np.unique(keywords_list)
    
    # 원핫인코딩 매트릭스 만들기
    zero_array = np.zeros(shape=(df.shape[0], len(keywords_list)))
    zero_df = pd.DataFrame(zero_array, columns=keywords_list)
    
    for idx, keyword in enumerate(df['keyword']):
        indices = zero_df.columns.get_indexer(keyword)
        zero_df.iloc[idx, indices] = 1
    
    # 유사도 구하기
    key_df = zero_df.copy()
    key_sim = cosine_similarity(key_df, key_df)
    
    # 정렬
    sorted_key_sim = key_sim.argsort()[:,::-1]
    
    df['weight_average'] = weight_average(df, 0.6)
    
    sim_item1 = weight_vote_avg(df, sorted_key_sim, product_name_f2_1)   
    imgpath_f1=sim_item1.loc[:,'imgpaths'].to_list()
    price_f1=sim_item1.loc[:,'prices'].to_list()
    product_f1=sim_item1.loc[:,'product_name'].to_list()
    
    return render(request, 'olyoung/ver2_result1.html',{'product_f1':product_f1,'imgpath_f1':imgpath_f1,'price_f1':price_f1})

def ver2_result2(request):
    # 선택값 받아오기
    product_name_f2_2=request.GET.get('product_name_f2')
    
    df = pd.read_csv('추천시스템전처리.csv', encoding='utf8', index_col=0)
    a_df = pd.read_csv('product_df2.csv',encoding='cp949')
    df = pd.merge(df, a_df[['product_id','rev_score','imgpaths','prices']], how='inner', on='product_id')
    df = df.drop_duplicates(subset='product_id', keep='first').reset_index()
    df['keyword'] = df['keyword'].apply(lambda x: str(x).replace("'","").replace('[','').replace(']',''))   

    # 전체 키워드 리스트
    keyword = []
    for i in df['keyword']:
        i = i.split(',')
        for j in i:
            keyword.append(j.strip())
    keyword = list(set(keyword))

    # 키워드 딕셔너리
    keyword_dic_li = []
    for i, j in zip(range(len(keyword)), keyword):
        keyword_dic_li.append({'id': i, 'name': j})
    
    # 제품별 키워드 딕셔너리
    keyword_list = []
    for names in df['keyword']:
        li = []
        for val in keyword_dic_li: 
            if val['name'] in names:
                li.append(val)
            else:
                continue
        keyword_list.append(li)
    
    df['keyword'] = keyword_list
    df['keyword'] = df['keyword'].apply(lambda x: [i['name'] for i in x])
    
    # 키워드 컬럼 펼치기
    keywords_list = []
    for keyword in df['keyword']:
        keywords_list.extend(keyword)
    keywords_list = np.unique(keywords_list)

    # 원핫인코딩 매트릭스 만들기
    zero_array = np.zeros(shape=(df.shape[0], len(keywords_list)))
    zero_df = pd.DataFrame(zero_array, columns=keywords_list)
    
    for idx, keyword in enumerate(df['keyword']):
        indices = zero_df.columns.get_indexer(keyword)
        zero_df.iloc[idx, indices] = 1
    
    # 유사도 구하기
    key_df = zero_df.copy()
    key_sim = cosine_similarity(key_df, key_df)
    
    # 정렬
    sorted_key_sim = key_sim.argsort()[:,::-1]
    
    # 가중 평점으로 추천
    # 가중평점 = (v / (v+m)) * r + (m / (v+m)) * c
    # v = 아이팀별 평점 투표 횟수
    # m = 평점 부여를 위한 최소 투표 횟수
    # r = 아이템별 평균 평점
    # c = 전체 아이템의 평균 평점
    
    df['weight_average'] = weight_average(df, 0.6)
    #df[['product_name', 'rev_score', 'weight_average', 'review_cnt']].sort_values(by='weight_average', ascending=False)
    
    sim_item2 = weight_vote_avg(df, sorted_key_sim, product_name_f2_2)
    imgpath_f2=sim_item2.loc[:,'imgpaths'].to_list()
    price_f2=sim_item2.loc[:,'prices'].to_list()
    product_f2=sim_item2.loc[:,'product_name'].to_list()
    
    return render(request, 'olyoung/ver2_result2.html',{'product_f2':product_f2,'imgpath_f2':imgpath_f2,'price_f2':price_f2})

def ver2_result3(request):
    # 선택값 받아오기
    product_name_f2_3=request.GET.get('product_name_f3')
    
    df = pd.read_csv('추천시스템전처리.csv', encoding='utf8', index_col=0)
    a_df = pd.read_csv('product_df2.csv',encoding='cp949')
    df = pd.merge(df, a_df[['product_id','rev_score','imgpaths','prices']], how='inner', on='product_id')
    df = df.drop_duplicates(subset='product_id', keep='first').reset_index()
    df['keyword'] = df['keyword'].apply(lambda x: str(x).replace("'","").replace('[','').replace(']',''))   

    # 전체 키워드 리스트
    keyword = []
    for i in df['keyword']:
        i = i.split(',')
        for j in i:
            keyword.append(j.strip())
    keyword = list(set(keyword))

    # 키워드 딕셔너리
    keyword_dic_li = []
    for i, j in zip(range(len(keyword)), keyword):
        keyword_dic_li.append({'id': i, 'name': j})
    
    # 제품별 키워드 딕셔너리
    keyword_list = []
    for names in df['keyword']:
        li = []
        for val in keyword_dic_li: 
            if val['name'] in names:
                li.append(val)
            else:
                continue
        keyword_list.append(li)
    
    df['keyword'] = keyword_list
    df['keyword'] = df['keyword'].apply(lambda x: [i['name'] for i in x])
    
    # 키워드 컬럼 펼치기
    keywords_list = []
    for keyword in df['keyword']:
        keywords_list.extend(keyword)
    keywords_list = np.unique(keywords_list)

    # 원핫인코딩 매트릭스 만들기
    zero_array = np.zeros(shape=(df.shape[0], len(keywords_list)))
    zero_df = pd.DataFrame(zero_array, columns=keywords_list)
    
    for idx, keyword in enumerate(df['keyword']):
        indices = zero_df.columns.get_indexer(keyword)
        zero_df.iloc[idx, indices] = 1
    
    # 유사도 구하기
    key_df = zero_df.copy()
    key_sim = cosine_similarity(key_df, key_df)
    
    # 정렬
    sorted_key_sim = key_sim.argsort()[:,::-1]
    
    # 가중 평점으로 추천
    # 가중평점 = (v / (v+m)) * r + (m / (v+m)) * c
    # v = 아이팀별 평점 투표 횟수
    # m = 평점 부여를 위한 최소 투표 횟수
    # r = 아이템별 평균 평점
    # c = 전체 아이템의 평균 평점
    
    df['weight_average'] = weight_average(df, 0.6)
    #df[['product_name', 'rev_score', 'weight_average', 'review_cnt']].sort_values(by='weight_average', ascending=False)
    
    sim_item3 = weight_vote_avg(df, sorted_key_sim, product_name_f2_3)
    imgpath_f3=sim_item3.loc[:,'imgpaths'].to_list()
    price_f3=sim_item3.loc[:,'prices'].to_list()
    product_f3=sim_item3.loc[:,'product_name'].to_list()
    
    return render(request, 'olyoung/ver2_result3.html',{'product_f3':product_f3,'imgpath_f3':imgpath_f3,'price_f3':price_f3})