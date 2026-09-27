import streamlit as st
import google.generativeai as genai
import pandas as pd
from PIL import Image
import PyPDF2
import plotly.express as px
import requests
from bs4 import BeautifulSoup
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# 1. 화면 기본 설정 및 세션 상태(메모리) 초기화
st.set_page_config(page_title="CEO 통합 경영 대시보드", layout="wide")
st.title("📊 CEO 경영 의사결정 및 K-IFRS 회계 통합 워크스테이션")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  
if "extracted_data" not in st.session_state:
    st.session_state.extracted_data = ""  
if "current_df" not in st.session_state:
    st.session_state.current_df = None  
if "ceo_dashboard_result" not in st.session_state:
    st.session_state.ceo_dashboard_result = ""

# 2. 사이드바 설정 (API 및 SMTP 메일 보안 터널 연동)
api_key = st.sidebar.text_input("Gemini API Key", type="password")
st.sidebar.markdown("---")
st.sidebar.subheader("📧 사내 메일 송신 설정 (보안 패치 완료)")
smtp_server = st.sidebar.text_input("SMTP 서버 (예: ://naver.com)", "://naver.com")
smtp_port = st.sidebar.number_input("SMTP 포트", value=587)
sender_email = st.sidebar.text_input("보내는 사람 메일 주소")
sender_password = st.sidebar.text_input("보내는 사람 보안 토큰 (앱 비밀번호)", type="password")
receiver_email = st.sidebar.text_input("받는 사람 메일 주소 (사장님 보고용)")

if api_key:
    genai.configure(api_key=api_key)
    try:
        model = genai.GenerativeModel(model_name="gemini-3.6-flash")
    except Exception as e:
        st.sidebar.error(f"모델 로드 실패: {e}")
        model = None

    # 3. 다중 리소스 입력 파트
    st.subheader("📥 다중 경영 리소스 입력 파트")
    col_f1, col_f2 = st.columns(2)
    
    is_image = False
    image_obj = None
    
    with col_f1:
        uploaded_file = st.file_uploader("엑셀 장부, 회계 PDF, 제품 사진 등을 업로드하세요", type=["png", "jpg", "jpeg", "csv", "xlsx", "pdf", "txt"])
        if uploaded_file:
            if uploaded_file.name.endswith(('png', 'jpg', 'jpeg')):
                image_obj = Image.open(uploaded_file)
                st.image(image_obj, caption="업로드된 이미지 프리뷰", use_container_width=True)
                is_image = True
            elif uploaded_file.name.endswith(('csv', 'xlsx')):
                try:
                    raw_df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('csv') else pd.read_excel(uploaded_file, engine='openpyxl')
                    
                    # [Arrow 호환성 안전 패치] 문자열 캐스팅으로 화면 크래시 차단
                    preview_df = raw_df.astype(str)
                    st.session_state.current_df = raw_df 
                    
                    st.write("📊 데이터 상위 5행 미리보기 (Arrow 호환성 패치 완료)")
                    st.dataframe(preview_df.head(5)) 
                    
                    # 🌟 [429 무료 쿼터 차단벽 우회: 리스크 가중치 우선순위 정렬 필터 모듈]
                    if len(raw_df) > 50:
                        st.sidebar.info("🔍 대용량 데이터 감지: AI 쿼터 보호를 위해 리스크 가중치 정렬 분석을 가동합니다.")
                        sort_df = raw_df.copy()
                        if 'Status' in sort_df.columns:
                            # Cancelled 및 Pending 데이터를 최상단으로 정렬하여 의사결정 효율 극대화
                            sort_df['__risk_score__'] = sort_df['Status'].apply(lambda x: 0 if str(x).strip().lower() in ['cancelled', 'pending'] else 1)
                            sort_df = sort_df.sort_values(by='__risk_score__').drop(columns=['__risk_score__'])
                        elif 'Total' in sort_df.columns:
                            # 비정상 수치 및 마이너스 매출을 최상단으로 우선 정렬
                            sort_df = sort_df.sort_values(by='Total')
                        
                        sliced_df = sort_df.head(50)
                        st.sidebar.warning(f"⚠️ 429 방어 완료: 가장 치명적인 리스크 데이터 50개 행을 추출해 AI에게 전달했습니다. (전체 행: {len(raw_df)})")
                    else:
                        sliced_df = raw_df
                    
                    st.session_state.extracted_data += f"\n[엑셀 데이터 (핵심 리스크 50행 분석)]\n{sliced_df.to_markdown(index=False)}"
                except Exception as e:
                    st.error(f"데이터 파싱 에러: {e}")
            elif uploaded_file.name.endswith('pdf'):
                try:
                    pdf_reader = PyPDF2.PdfReader(uploaded_file)
                    text = "".join([page.extract_text() for page in pdf_reader.pages if page.extract_text()])
                    st.session_state.extracted_data += f"\n[PDF 데이터]\n{text}"
                    st.success("PDF 텍스트 추출 완료")
                except Exception as e:
                    st.error(f"PDF 파싱 에러: {e}")

    with col_f2:
        web_url = st.text_input("시장 동향 분석용 뉴스 URL을 입력하세요:")
        if st.button("🌐 뉴스 데이터 스크랩") and web_url:
            try:
                res = requests.get(web_url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=10)
                if res.status_code == 200:
                    soup = BeautifulSoup(res.content, 'html.parser')
                    for t in soup(["script", "style", "nav", "footer", "header"]): t.decompose()
                    st.session_state.extracted_data += f"\n[뉴스 데이터]\n{soup.get_text(separator=' ', strip=True)[:3000]}"
                    st.success("뉴스 데이터 결합 완료")
            except Exception as e:
                st.error(f"크롤링 에러: {e}")
    # --- 4. 데이터 자동 동적 시각화 모듈 (고급 비즈니스 차트군) ---
    if st.session_state.current_df is not None:
        st.markdown("---")
        st.subheader("📊 실시간 데이터 동적 시각화 패널")
        cols = st.session_state.current_df.columns.tolist()
        
        col1, col2, col3 = st.columns(3)
        with col1:
            x_axis = st.selectbox("X축 (또는 그룹화 기준) 선택", cols)
        with col2:
            y_axis = st.selectbox("Y축 (수치형 데이터) 선택", cols)
        with col3:
            chart_type = st.selectbox("차트 종류", [
                "막대 그래프 (Bar)", 
                "누적 막대 그래프 (Stacked Bar)", 
                "선 그래프 (Line)", 
                "영역 차트 (Area)",
                "산점도 (Scatter)", 
                "원형 차트 (Pie)", 
                "도넛 차트 (Donut)"
            ])
            
        if st.button("📈 인터랙티브 차트 생성"):
            try:
                if chart_type == "막대 그래프 (Bar)":
                    fig = px.bar(st.session_state.current_df, x=x_axis, y=y_axis, title=f"{x_axis}별 {y_axis} 실적")
                elif chart_type == "누적 막대 그래프 (Stacked Bar)":
                    fig = px.bar(st.session_state.current_df, x=x_axis, y=y_axis, color=x_axis, title=f"{x_axis}별 {y_axis} 누적 구성")
                elif chart_type == "선 그래프 (Line)":
                    fig = px.line(st.session_state.current_df, x=x_axis, y=y_axis, title=f"{x_axis}에 따른 {y_axis} 추이")
                elif chart_type == "영역 차트 (Area)":
                    fig = px.area(st.session_state.current_df, x=x_axis, y=y_axis, title=f"{x_axis} 기준 {y_axis} 누적 볼륨")
                elif chart_type == "산점도 (Scatter)":
                    fig = px.scatter(st.session_state.current_df, x=x_axis, y=y_axis, title=f"{x_axis}와 {y_axis}의 상관 분포")
                elif chart_type == "원형 차트 (Pie)":
                    fig = px.pie(st.session_state.current_df, names=x_axis, values=y_axis, title=f"{x_axis} 기준 {y_axis} 마켓 셰어")
                elif chart_type == "도넛 차트 (Donut)":
                    fig = px.pie(st.session_state.current_df, names=x_axis, values=y_axis, hole=0.4, title=f"{x_axis} 기준 {y_axis} 점유율 (도넛형)")
                
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"차트 생성 오류: {e}")

    # --- 5. 사장님 보고용 전략 수립 엔진 ---
    st.markdown("---")
    st.subheader("🎯 사장님 보고용 대시보드 및 전략 수립 엔진")
    
    if st.button("🚀 한 장의 통합 현황판(Dashboard) 생성"):
        if model and (st.session_state.extracted_data or is_image):
            with st.spinner("AI가 실무 엑셀 장부 입체 분석 및 K-IFRS 검증 중..."):
                prompt = (
                    f"당신은 사내 최고 권위의 데이터 과학자이자 수석 회계감사관입니다. 제공된 모든 리소스는 《회사에서 바로 통하는 실무 엑셀》 기반의 핵심 기업 데이터입니다.\n"
                    f"데이터를 쪼갤 때 행과 열의 텍스트와 숫자를 단 한 줄도 누락하거나 흐리지 마시고, 정밀하게 연산하여 최고경영진(CEO) 보고 규격으로 작성해 주세요.\n\n"
                    f"필수 출력 항목:\n"
                    f"1. [경영 분석 현황판]: 매출 소계와 실적 데이터의 이상 징후(마이너스 매출, NaN 공백 등)를 정확히 짚어 요약해 주세요.\n"
                    f"2. [K-IFRS 회계 검증]: 발견된 회계적 결함이나 장부 기록이 K-IFRS 몇 단원, 몇 절(예: 제1115호, 제1016호 등)의 문항에 위배되거나 적용되어야 하는지 명확히 감사 판단을 내려주세요.\n"
                    f"3. [경영자 제언]: 자금 유동성 확보 및 운영 효율화를 위한 비상 자문 의견을 주십시오.\n"
                    f"4. [경쟁사 분석 및 경쟁 전략]: 당사 부실 품목을 방어하고 시장 점유율을 독점할 핵심 차별화 전략을 제안해 주세요.\n\n"
                    f"[입력된 통합 자원 데이터]:\n{st.session_state.extracted_data}"
                )
                
                try:
                    if is_image:
                        response = model.generate_content([prompt, image_obj])
                    else:
                        response = model.generate_content(prompt)
                    st.session_state.ceo_dashboard_result = response.text
                except Exception as e:
                    st.error(f"AI 연산 중 에러: {e}")
        else:
            st.error("Gemini API 키가 없거나 분석 대상 데이터가 로드되지 않았습니다.")

    # 6. 대시보드 결과 출력 및 이메일 전송 [🌟 finally 세션 클로즈 디펜스 이식]
    if st.session_state.ceo_dashboard_result:
        st.info("📋 생성된 사장님 보고용 실시간 대시보드 결과")
        st.markdown(st.session_state.ceo_dashboard_result)
        
        st.markdown("---")
        st.subheader("📨 이메일 원클릭 사내 전송")
        if st.button("📧 이메일로 이 대시보드 즉시 송신"):
            if sender_email and sender_password and receiver_email:
                with st.spinner("사내 메일 터널 전송 중..."):
                    server = None  # finally 자원반환 스캔용 선언
                    try:
                        msg = MIMEMultipart()
                        msg['From'] = sender_email
                        msg['To'] = receiver_email
                        msg['Subject'] = "🚨 [경영 보고] AI 기반 통합 경영 대시보드 및 K-IFRS 검증 보고서"
                        msg.attach(MIMEText(st.session_state.ceo_dashboard_result, 'plain', 'utf-8'))
                        
                        # 포트 규격별 동적 보안 연결 수립
                        if int(smtp_port) == 465:
                            server = smtplib.SMTP_SSL(smtp_server, int(smtp_port), timeout=15)
                        else:
                            server = smtplib.SMTP(smtp_server, int(smtp_port), timeout=15)
                            server.ehlo()
                            server.starttls()  # 최신 STARTTLS 보안 프로토콜 주입
                            server.ehlo()
                        
                        server.login(sender_email, sender_password)
                        server.sendmail(sender_email, receiver_email, msg.as_string())
                        st.success(f"✅ 사장님 메일({receiver_email})로 보고서 전송이 완료되었습니다!")
                    except smtplib.SMTPAuthenticationError:
                        st.error("❌ 메일 전송 실패: 로그인 인증 실패. 네이버 보안 토큰(앱 비밀번호) 입력을 확인하세요.")
                    except Exception as e:
                        st.error(f"메일 발송 오류: {e}")
                    finally:
                        # 🌟 연산 성공/실패 여부와 관계없이 메일 서버 소켓 자원을 100% 해제합니다.
                        if server is not None:
                            try:
                                server.quit()
                            except:
                                pass
            else:
                st.warning("사이드바에 메일 송신자/수신자 정보 및 앱 비밀번호를 모두 입력해 주세요.")
else:
    st.info("왼쪽 사이드바에 Gemini API Key를 입력하면 장난감이 작동하기 시작합니다.")



