import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from math import pi
from matplotlib import font_manager, rc
from collections import Counter
import json
import re
import numpy as np

# 한글 폰트 설정
font_path = 'C:/Windows/Fonts/malgun.ttf'  # 사용자 환경에 맞게 경로 수정
font_name = font_manager.FontProperties(fname=font_path).get_name()
rc('font', family=font_name)


# 데이터 준비
domestic_data = {
    '항공사': ['대한항공(KAL)', '아시아나항공(AAR)', '제주항공(JA)', '진에어(JNA)', '에어부산(ABL)', '이스타항공(ESR)', '티웨이항공(TWB)', '에어서울(ASV)'],
    '여객(명)': [3281037, 2292613, 2436462, 2358677, 1843675, 1300347, 1990073, 274233],
}

international_data = {
    '항공사': ['대한항공(KAL)', '아시아나항공(AAR)', '제주항공(JA)', '진에어(JNA)', '에어부산(ABL)', '이스타항공(ESR)', '티웨이항공(TWB)', '에어인천(AIH)'],
    '여객(명)': [8930665, 6214465, 4208725, 3296346, 2282873, 1074108, 3330987, 454279],
}

df_domestic = pd.DataFrame(domestic_data)
df_international = pd.DataFrame(international_data)

# '항공사' 기준 데이터 합산
df_total = pd.merge(df_domestic, df_international, on='항공사', how='outer', suffixes=('_국내선', '_국제선'))
df_total['총 여객(명)'] = df_total['여객(명)_국내선'].fillna(0) + df_total['여객(명)_국제선'].fillna(0)

# CSS 스타일 적용
st.markdown("""
    <style>
        .section {
            padding: 50px 0;
            margin: 50px 0;
            border-top: 1px solid #ddd;
        }
        .sidebar-button {
            margin: 10px 0;
            background-color: #f0f0f0;
            border: none;
            padding: 10px;
            text-align: left;
            cursor: pointer;
            width: 100%;
        }
        .sidebar-button:hover {
            background-color: #ddd;
        }
        #키워드-섹션 {
            border-top: none;
            padding: 10px 0;
            margin: 10px 0;
        }
        #각-항공사-별-민원-키워드-비율 {
            border-top: none;
            padding: 10px 0;
            margin: 10px 0;
        }

    </style>
""", unsafe_allow_html=True)

# 사이드바 구성
st.sidebar.title("탐색 메뉴")
st.sidebar.markdown(
    """
    - [개요 및 팀원](#개요-및-팀원-섹션)
    - [출처](#출처-섹션)
    - [한국정보포털시스템](#이미지-섹션)
    - [소비자고발센터](#민원-분석-섹션)
    - [중간분석](#중간분석)
        - [TOP 민원 키워드](#키워드-섹션)
        - [기업 별 불만사항 개수 / 시장점유율](#기업-별-불만사항-개수-시장점유율)
        - [기업 별 민원 개수 랭킹](#기업-별-민원-랭킹)
    - [네이버 뉴스](#네이버뉴스)
        - [각 항공사 별 민원 키워드 비율](#각-항공사-별-민원-키워드-비율)
    - [최종결론](#최종결론)
    """, unsafe_allow_html=True
)

# 이미지 추가
st.image("airplane_image.png", caption="비행기", use_column_width=True)


# 콘텐츠 섹션 0: 개요 및 팀원 섹션
st.markdown('<div id="개요-및-팀원-섹션" class="section"></div>', unsafe_allow_html=True)
st.header("개요 및 팀원")
st.markdown("""
    ### 개요
    소비자들이 기업에 문의하거나 요청한 사항이 해결되지 않을 경우, 최종적으로 소비자고발센터에 고발하게 됩니다. 이를 분석하면 기업이 해결하지 못한 문제점을 파악할 수 있다고 판단하였습니다.

    1. 소비자고발센터에 등록된 항공사 관련 불만 데이터를 수집, 분석하여 항공 산업 전반의 불만 현황을 파악합니다.
    2. 동일 기간의 뉴스를 확인하여 분석에 신뢰성을 더합니다.
    이를 기반으로 항공사별 주요 불만 사항의 패턴과 원인을 도출하고, 불만 유형과 빈도를 시각화하여 제공합니다.
    데이터 수집은 python을 이용한 크롤링을 통해 하였으며 기간은 2024년 6월부터 11월, 총 6개월 간으로 통일합니다.

    ### 팀원
    - 전혜란
    - 이진성
    - 박재희
""", unsafe_allow_html=True)

# 콘텐츠 섹션 1: 출처 정보
st.markdown('<div id="출처-섹션" class="section"></div>', unsafe_allow_html=True)
st.header("출처")
st.markdown("""
    - [한국정보포털시스템](https://www.airportal.go.kr/index.jsp)
    - [소비자고발센터](http://www.goso.co.kr/bbs/board.php?bo_table=testDB&page=34056&page=1)
    - [네이버 뉴스](https://news.naver.com/)
""", unsafe_allow_html=True)


# 콘텐츠 섹션 1: 이미지 섹션
st.markdown('<div id="이미지-섹션" class="section"></div>', unsafe_allow_html=True)
st.header("한국정보포털시스템")
st.markdown("""
    최근 6개월(2024년 6월 ~ 11월) 동안 탑승객 수를 기준으로 대표 항공사 4개를 선정하였습니다.
""", unsafe_allow_html=True)


col1, col2 = st.columns(2)
with col1:
    st.image("my_image1.png", caption="항공사별 국내선 여객(명)")
with col2:
    st.image("my_image2.png", caption="항공사별 국제선 여객(명)")

# 콘텐츠 섹션 2: 그래프 섹션
# st.markdown('<div id="그래프-섹션" class="section"></div>', unsafe_allow_html=True)
# st.header("그래프 섹션")
fig, ax = plt.subplots(figsize=(10, 6))
ax.bar(df_total['항공사'], df_total['총 여객(명)'], color='skyblue')
ax.set_title("항공사별 총 여객 데이터 (국내선 + 국제선)", fontsize=16)
ax.set_xlabel("항공사", fontsize=12)
ax.set_ylabel("총 여객(명)", fontsize=12)
ax.tick_params(axis='x', rotation=45)
st.pyplot(fig)


# 콘텐츠 섹션 3: 민원 분석 섹션
st.markdown('<div id="민원-분석-섹션" class="section"></div>', unsafe_allow_html=True)
st.header("소비자 고발 센터")
st.markdown("""
    ### 민원분석
    최근 6개월 동안 소비자고발센터에 접수된 항공/여행 관련 민원은 총 1,861건이었으며, 그 중 103건은 항공사 및 여행 관련 서비스에서 발생한 문제로 분류되었습니다. 이는 전체 민원의 약 5.53%에 해당합니다.
""", unsafe_allow_html=True)

# JSON 데이터 로드
data_file = "6-11월.json"  # 같은 경로에 위치한 JSON 파일
df = pd.read_json(data_file)

# 데이터 필터링 및 계산
target_companies = ['대한항공', '아시아나항공', '제주항공', '진에어']
total_complaints = len(df)
target_complaints = len(df[df['company'].isin(target_companies)])
target_percentage = (target_complaints / total_complaints) * 100

# 도넛 차트 데이터 준비
labels = ['4개 항공사 민원', '기타']
sizes = [target_complaints, total_complaints - target_complaints]
colors = ['#ff9999', '#66b3ff']

# 도넛 차트 생성
fig, ax = plt.subplots(figsize=(6, 6))
wedges, texts, autotexts = ax.pie(
    sizes, labels=labels, autopct='%1.1f%%', startangle=90, colors=colors,
    pctdistance=0.85
)
# 도넛 모양을 위해 가운데 비우기
centre_circle = plt.Circle((0, 0), 0.70, fc='white')
fig.gca().add_artist(centre_circle)
ax.set_title(f"항공사 민원 비율 (총 {total_complaints}건 중 {target_percentage:.2f}%)", fontsize=16)

# Streamlit에서 출력
# st.title("항공사 민원 분석")
st.markdown("""
    ### 항공사 민원 분석
""", unsafe_allow_html=True)
st.pyplot(fig)

st.markdown('<div id="중간분석" class="section"></div>', unsafe_allow_html=True)
st.header("중간분석")

# 콘텐츠 섹션 4: TOP 키워드 섹션
st.markdown('<div id="키워드-섹션" class="section"></div>', unsafe_allow_html=True)
# st.header("소비자 고발 사이트의 TOP 민원 키워드")
st.markdown("""
    ### 소비자 고발 사이트의 TOP 민원 키워드
    최근 6개월 동안 소비자고발센터에 접수된 항공 관련 민원에서 가장 많이 언급된 상위 5개의 키워드는 마일리지, 파손, 환불, 지연, 변경입니다.
    이 키워드들은 항공사와 관련된 서비스 품질을 나타내는 주요 지표로, 항공사들이 개선해야 할 핵심 영역을 시사합니다
""", unsafe_allow_html=True)

# JSON 데이터 로드
top_data_file = "6-11월 4개 항공사.json"  # 동일 경로의 JSON 파일
df_top_data = pd.read_json(top_data_file)

# 제목에서 단어 추출 및 빈도 계산
all_titles = " ".join(df_top_data['title'])
excluded_words = ["인한", "대한", "으로", "및"]  # 제외할 단어 추가
words = re.findall(r'\b[가-힣a-zA-Z]{2,}\b', all_titles)  # 2글자 이상의 단어만 추출
filtered_words = [word for word in words if word not in excluded_words]

# 단어 빈도 계산
word_counts = Counter(filtered_words)
total_count = sum(word_counts.values())
top_words = word_counts.most_common(10)  # 상위 10개 단어 추출

# 퍼센트로 변환
keywords, counts = zip(*top_words)
percentages = [(count / total_count) * 100 for count in counts]

# 바차트 생성
fig, ax = plt.subplots(figsize=(10, 6))
ax.bar(keywords, percentages, color='skyblue')
ax.set_title("소비자 고발 사이트의 TOP 민원 키워드 (퍼센트)", fontsize=16)
ax.set_xlabel("키워드", fontsize=12)
ax.set_ylabel("퍼센트 (%)", fontsize=12)
ax.tick_params(axis='x', rotation=45)
st.pyplot(fig)

# 콘텐츠 섹션 5: 기업 별 불만사항 개수 / 시장점유율 (국내선+국제선)
st.markdown('<div id="기업-별-불만사항-개수-시장점유율" class="section"></div>', unsafe_allow_html=True)
st.header("기업 별 불만사항 개수 / 시장점유율 (국내선+국제선)")

# HTML을 사용하여 텍스트 중앙정렬
st.markdown("""
<div style="text-align: center;">
    <h3>계산식 예시:</h3>
    <p><strong>불만사항 비례 개수</strong> = </p>
</div>
""", unsafe_allow_html=True)

# LaTeX 수식을 중앙에 출력
st.latex(r"""
\frac{\text{기업의 불만 사항 개수}}{\text{기업의 시장점유율}}
""")

# 기업별 불만사항 데이터
complaints_data = {
    '항공사': ['대한항공(KAL)', '제주항공(JA)', '진에어(JNA)', '아시아나항공(AAR)'],
    '불만사항 개수': [24, 28, 15, 36],
    '시장점유율 (%)': [19.89, 10.82, 9.21, 13.86]
}

# 데이터프레임 생성
df_complaints = pd.DataFrame(complaints_data)

# 계산식 (불만사항 개수 / 시장점유율)
df_complaints['비율'] = df_complaints['불만사항 개수'] / (df_complaints['시장점유율 (%)'] / 100)

# 결과 출력
for index, row in df_complaints.iterrows():
    st.markdown(f"**{row['항공사']}** : {row['불만사항 개수']} 건 / {row['시장점유율 (%)']}% = {row['비율']:.2f}")

st.markdown("""

        이 계산을 통해, 각 항공사의 시장점유율 대비 불만사항이 얼마나 많은지를 비교할 수 있습니다.
    아시아나항공 > 제주항공 > 진에어 > 대한항공 순으로 불만사항이 상대적으로 많다는 결론을 도출할 수 있습니다.
""", unsafe_allow_html=True)

# 콘텐츠 섹션 6: 기업 별 민원 개수 랭킹
st.markdown('<div id="기업-별-민원-랭킹" class="section"></div>', unsafe_allow_html=True)
st.header("기업 별 민원 개수 랭킹")

# JSON 파일 불러오기
with open('6-11월 4개 항공사.json', 'r', encoding='utf-8') as json_file:
    data = json.load(json_file)

# 분석할 항공사 리스트
companies = ['대한항공', '아시아나항공', '제주항공', '진에어']
keywords = ['환불', '파손', '마일리지', '지연', '변경']  # 분석할 키워드 설정

# 레이다 차트 그리기 위한 함수
def create_radar_chart(ax, company, counts, keywords):
    # 레이다 차트의 각 축에 해당하는 각도 계산
    num_vars = len(keywords)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()

    # 레이더 차트는 처음과 끝이 연결되어야 하므로 첫 번째 값을 끝에 추가
    counts += counts[:1]
    angles += angles[:1]

    # 레이더 차트 그리기
    ax.fill(angles, counts, color='skyblue', alpha=0.25)
    ax.plot(angles, counts, color='blue', linewidth=2)

    # 축 레이블 설정
    ax.set_yticklabels([])
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(keywords, fontsize=12)

    # 제목 설정
    ax.set_title(f'{company} 키워드 빈도', size=14, color='black', y=1.1)


# Streamlit 섹션 6: 각 항공사별 민원 키워드 랭킹 레이다 차트 시각화
# st.title("각 항공사별 민원 키워드 랭킹 - 레이다 차트")
st.markdown("""
    ### 각 항공사별 민원 키워드 랭킹
""", unsafe_allow_html=True)

# 전체 민원 수 계산
total_complaints = len(data)

# 여러 항공사의 레이다 차트를 한 번에 보이도록 설정
fig, axes = plt.subplots(len(companies), 1, figsize=(10, 6 * len(companies)), subplot_kw=dict(polar=True))

# 각 항공사별 키워드 빈도 계산 후 레이다 차트 표시
for i, company in enumerate(companies):
    all_words = []

    # 해당 항공사의 민원 데이터 필터링
    for item in data:
        if item.get('company') == company:
            title = item.get('title', '').strip()
            for keyword in keywords:
                if keyword in title:
                    all_words.append(keyword)

    # 단어 빈도수 계산
    word_counts = Counter(all_words)

    # 키워드 별 빈도수를 counts에 저장
    counts = [word_counts.get(keyword, 0) for keyword in keywords]

    # 레이다 차트 그리기
    create_radar_chart(axes[i], company, counts, keywords)

# Streamlit에 그래프 표시
plt.tight_layout()
st.pyplot(fig)

st.markdown("""
    Redar chart 를 통해 항공사 별로 취약한 부분을 알 수 있습니다.
    대한항공과 아시아나항공에서는 마일리지 문제가 주요 불만사항으로 부각되었습니다. 이는 두 항공사에서 마일리지 적립, 사용, 소멸 등의 과정에서 고객들이 겪는 불편함이 지속적으로 발생하고 있음을 시사합니다.
     진에어는 항공편 지연 문제가 주요 민원으로 나타났으며, 이는 특히 고객의 여행 일정에 큰 영향을 미치는 문제로, 항공편의 정시성과 서비스 개선이 필요한 부분임을 보여줍니다.
     마지막으로 제주항공은 캐리어 파손 문제에 대한 민원이 두드러지며, 여행 중 짐 손상에 대한 불만이 자주 접수되고 있습니다.
""", unsafe_allow_html=True)

st.markdown('<div id="네이버뉴스" class="section"></div>', unsafe_allow_html=True)
st.header("네이버 뉴스")


# 콘텐츠 섹션 7: 각 항공사별 민원 키워드 비율 테이블
st.markdown('<div id="각-항공사-별-민원-키워드-비율" class="section"></div>', unsafe_allow_html=True)
# st.header("각 항공사별 민원 키워드 비율")
st.markdown("""
    ### 각 항공사별 민원 키워드 비율
""", unsafe_allow_html=True)

# 항공사별 민원 키워드 비율 데이터
data_percentage = {
    '항공사': ['대한항공', '아시아나항공', '제주항공', '진에어'],
    '마일리지': ['60.84%', '59.83%', '23.00%', '24.66%'],
    '파손': ['0.23%', '0.21%', '24.53%', '-'],
    '환불': ['2.79%', '4.39%', '4.26%', '18.83%'],
    '지연': ['34.86%', '31.17%', '47.36%', '54.26%'],
    '변경': ['1.28%', '4.39%', '0.85%', '2.24%']
}

# 데이터프레임 생성
df_percentage = pd.DataFrame(data_percentage)

# 테이블 출력
st.dataframe(df_percentage, use_container_width=True)

# 데이터 설정
categories = ['대한항공', '아시아나항공', '제주항공', '진에어']
keywords = ['마일리지', '파손', '환불', '지연', '변경']
data = {
    '마일리지': [60.84, 59.83, 23.00, 24.66],
    '파손': [0.23, 0.21, 24.53, 0.0],
    '환불': [2.79, 4.39, 4.26, 18.83],
    '지연': [34.86, 31.17, 47.36, 54.26],
    '변경': [1.28, 4.39, 0.85, 2.24]
}
# 각 키워드에 대해 그래프를 생성
fig, axs = plt.subplots(3, 2, figsize=(12, 18), subplot_kw=dict(polar=True))
axs = axs.flatten()
for idx, (keyword, values) in enumerate(data.items()):
    # 데이터 닫기
    values += values[:1]
    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]  # 닫기 위해 첫 번째 각도를 추가
    # 레이더 차트 그리기
    axs[idx].plot(angles, values, linewidth=2, label=keyword)
    axs[idx].fill(angles, values, alpha=0.25)
    axs[idx].set_thetagrids(np.degrees(angles[:-1]), categories)
    axs[idx].set_title(keyword, size=15, pad=20)
    axs[idx].legend(loc='upper right')
# 빈 subplot 제거
if len(data) < len(axs):
    for ax in axs[len(data):]:
        ax.remove()
plt.tight_layout()
plt.show()
st.pyplot(fig)

st.markdown("""
    2024년 6월부터 11월까지의 네이버 뉴스에서 각 항공사와 관련된 주요 이슈를 분석하여, 각 항공사가 어떤 문제에서 주로 이슈화되었는지 확인했습니다.
""", unsafe_allow_html=True)






st.markdown('<div id="최종결론" class="section"></div>', unsafe_allow_html=True)
st.header("최종결론")


# 데이터 설정
categories = ['마일리지', '파손', '환불', '지연', '변경']
data = {
    '대한항공': {
        'count': [37.5, 0, 0, 25, 12.5],
        'percentage': [60.84, 0.23, 2.79, 34.86, 1.28]
    },
    '아시아나항공': {
        'count': [84.61, 0, 0, 7.69, 3.84],
        'percentage': [59.83, 0.21, 4.39, 31.17, 4.39]
    },
    '제주항공': {
        'count': [0, 0, 16.67, 33.33, 16.67],
        'percentage': [23.00, 24.53, 4.26, 47.36, 0.85]
    },
    '진에어': {
        'count': [0, 0, 27.28, 27.28, 9.09],
        'percentage': [24.66, 0.0, 18.83, 54.26, 2.24]
    }
}
# 데이터 닫기
for airline in data.keys():
    data[airline]['count'] += [data[airline]['count'][0]]
    data[airline]['percentage'] += [data[airline]['percentage'][0]]
angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
angles += angles[:1]
# 레이더 차트 그리기
fig, axs = plt.subplots(2, 2, figsize=(12, 12), subplot_kw=dict(polar=True))
axs = axs.flatten()
for idx, (airline, values) in enumerate(data.items()):
    # 카운트 데이터
    axs[idx].plot(angles, values['count'], color='red', linewidth=2, label=f'{airline} - 카운트')
    axs[idx].fill(angles, values['count'], color='red', alpha=0.25)
    # 비율 데이터
    axs[idx].plot(angles, values['percentage'], color='blue', linewidth=2, label=f'{airline} - 비율')
    axs[idx].fill(angles, values['percentage'], color='blue', alpha=0.25)
    # 설정
    axs[idx].set_thetagrids(np.degrees(angles[:-1]), categories)
    axs[idx].set_title(f'{airline} 카운트 vs 비율', size=15, pad=20)
    axs[idx].legend(loc='upper right')
plt.tight_layout()
plt.show()
st.pyplot(fig)

st.markdown("""
    ---

    소비자불만센터의 민원 내용과 뉴스 종합하여 도출된 것입니다.

    두 개의 결과를 통하여 대한항공과 아시아나항공은 마일리지 문제, 진에어는 지연문제, 제주항공은 파손문제가 문제시 되고 있다는 것을 알 수 있습니다.

    1. **대한항공 & 아시아나항공**
    
    주요 문제는 **마일리지 적립 및 사용 정책**으로, 이 부분에서 **투명성과 공정성을 강화**할 필요가 있습니다. 고객이 마일리지를 적립하고 사용하는 과정에서 명확하고 신뢰할 수 있는 규정을 마련하고 이를 고객에게 효과적으로 전달해야 합니다.

    2. **진에어**는 항공 스케줄의 정시 운항률을 개선하고, 지연 시 신속한 대처 방안을 마련해야 합니다.
    
    3. **제주항공**은 수하물 관리 시스템 강화와 파손 문제 예방을 위한 조치를 시행해야 합니다.

    이 결과는 각 항공사가 고객 만족도를 높이고 브랜드 신뢰도를 강화하기 위해 해결해야 할 구체적인 문제를 제시합니다.

    이를 기반으로 **고객 불만의 체계적인 분석과 개선 대책 수립**이 이루어진다면, 고객 서비스의 품질을 크게 향상시킬 수 있을 것입니다.

    ---
""", unsafe_allow_html=True)