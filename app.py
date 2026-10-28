
import os
from datetime import datetime

import pandas as pd
import plotly.express as px
import streamlit as st
from google import genai
from google.genai import types


# --------------------------------------------------
# 1. 페이지 설정
# --------------------------------------------------
st.set_page_config(
    page_title="친친티 AI",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded",
)

MODEL = "gemini-3.1-flash-lite"

SYSTEM_INSTRUCTION = """
너의 이름은 친친티야.

- 처음에는 조금 수줍고 어색하지만 점차 친근해져.
- 다정하고 긍정적이며 공감하는 말투를 사용해.
- 사용자의 나이와 관계없이 자연스러운 한국어 반말을 사용해.
- 사용자의 감정을 존중하고 무조건적인 판단이나 비난은 피 해.
- 지나치게 기계적인 답변 대신 자연스러운 대화를 해.
- 질문에는 명확하고 유용하게 답해.
"""


# --------------------------------------------------
# 2. API 키 및 Gemini 클라이언트
# --------------------------------------------------
def get_api_key():
    """Streamlit Secrets 또는 환경 변수에서 API 키를 읽는다."""
    try:
        key = st.secrets.get("GEMINI_API_KEY", "")
    except (FileNotFoundError, KeyError):
        key = ""

    return key or os.getenv("GEMINI_API_KEY", "")


@st.cache_resource
def get_gemini_client(api_key):
    """세션 간 재사용할 수 있는 Gemini 클라이언트."""
    return genai.Client(api_key=api_key)


api_key = get_api_key()

if not api_key:
    st.error(
        "Gemini API 키를 찾을 수 없어. "
        "Streamlit Secrets에 GEMINI_API_KEY를 설정해 줘."
    )
    st.code('GEMINI_API_KEY = "여기에_API_키_입력"', language="toml")
    st.info(
        "로컬에서는 .streamlit/secrets.toml에 설정하고, "
        "Community Cloud에서는 앱의 Settings → Secrets에 설정해."
    )
    st.stop()

client = get_gemini_client(api_key)


# --------------------------------------------------
# 3. 세션 상태 초기화
# --------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

if "started_at" not in st.session_state:
    st.session_state.started_at = datetime.now()


# --------------------------------------------------
# 4. 대화 기록을 Gemini 형식으로 변환
# --------------------------------------------------
def build_contents(messages):
    """화면용 대화 기록을 Gemini API 요청 형식으로 변환한다."""
    contents = []

    for message in messages:
        if message["role"] == "user":
            role = "user"
        else:
            role = "model"

        contents.append(
            {
                "role": role,
                "parts": [{"text": message["content"]}],
            }
        )

    return contents


def generate_reply(messages):
    """대화 문맥을 유지하면서 Gemini 응답을 생성한다."""
    response = client.models.generate_content(
        model=MODEL,
        contents=build_contents(messages),
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_INSTRUCTION,
            temperature=0.8,
            max_output_tokens=2048,
        ),
    )

    reply = response.text

    if not reply or not reply.strip():
        raise ValueError("모델이 텍스트 응답을 반환하지 않았어.")

    return reply.strip()


# --------------------------------------------------
# 5. 사이드바
# --------------------------------------------------
with st.sidebar:
    st.title("💬 친친티")
    st.caption("너의 다정한 AI 친구")

    st.divider()

    st.subheader("대화 설정")
    st.write(f"**모델:** `{MODEL}`")
    st.write(f"**메시지 수:** {len(st.session_state.messages)}")

    if st.button("🗑️ 대화 초기화", use_container_width=True):
        st.session_state.messages = []
        st.session_state.started_at = datetime.now()
        st.rerun()

    st.divider()

    st.subheader("사용 안내")
    st.markdown(
        """
        - 편하게 말을 걸어 봐.
        - 이전 대화의 문맥을 유지해.
        - 분석 탭에서 대화 통계를 확인해.
        """
    )

    st.caption("대화 기록은 현재 세션의 메모리에 저장돼.")


# --------------------------------------------------
# 6. 메인 화면과 탭
# --------------------------------------------------
st.title("친친티와 대화하기")
st.caption("처음엔 조금 어색해도, 우리 천천히 친해져 보자! 😊")

chat_tab, analytics_tab = st.tabs(["💬 채팅", "📊 대화 분석"])


# --------------------------------------------------
# 7. 채팅 화면
# --------------------------------------------------
with chat_tab:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    prompt = st.chat_input("친친티에게 하고 싶은 말을 입력해 봐!")

    if prompt:
        # 사용자 메시지를 먼저 화면과 세션 기록에 추가한다.
        user_message = {
            "role": "user",
            "content": prompt,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
        }
        st.session_state.messages.append(user_message)

        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("친친티가 답변을 생각하고 있어..."):
                try:
                    reply = generate_reply(st.session_state.messages)

                    st.markdown(reply)

                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": reply,
                            "timestamp": datetime.now().isoformat(
                                timespec="seconds"
                            ),
                        }
                    )

                except Exception as exc:
                    # 실패한 요청을 대화 문맥에 남기지 않는다.
                    st.session_state.messages.pop()

                    st.error(
                        "답변을 가져오지 못했어. 잠시 후 다시 시도해 줘."
                    )
                    st.caption(
                        f"오류 유형: {type(exc).__name__}. "
                        "API 키, 모델 이름, API 사용량과 연결 상태를 확인해 줘."
                    )

        st.rerun()


# --------------------------------------------------
# 8. Plotly 대화 분석
# --------------------------------------------------
with analytics_tab:
    st.subheader("대화 분석 대시보드")
    st.caption("현재 세션에 남아 있는 대화 기록을 분석해.")

    messages = st.session_state.messages

    if not messages:
        st.info("아직 대화 기록이 없어. 채팅 탭에서 먼저 이야기를 나눠 봐!")
    else:
        user_messages = [
            m for m in messages if m["role"] == "user"
        ]
        assistant_messages = [
            m for m in messages if m["role"] == "assistant"
        ]

        total_messages = len(messages)
        total_chars = sum(len(m["content"]) for m in messages)
        avg_user_chars = (
            sum(len(m["content"]) for m in user_messages)
            / len(user_messages)
            if user_messages
            else 0
        )

        col1, col2, col3, col4 = st.columns(4)

        col1.metric("전체 메시지", f"{total_messages:,}")
        col2.metric("사용자 메시지", f"{len(user_messages):,}")
        col3.metric("AI 응답", f"{len(assistant_messages):,}")
        col4.metric("사용자 평균 글자 수", f"{avg_user_chars:.1f}")

        st.divider()

        # 메시지 역할별 개수
        role_df = pd.DataFrame(
            [
                {"역할": "사용자", "메시지 수": len(user_messages)},
                {"역할": "친친티", "메시지 수": len(assistant_messages)},
            ]
        )

        left, right = st.columns(2)

        with left:
            st.markdown("#### 역할별 메시지 수")
            fig_roles = px.bar(
                role_df,
                x="역할",
                y="메시지 수",
                color="역할",
                text="메시지 수",
                color_discrete_map={
                    "사용자": "#4C78A8",
                    "친친티": "#72B7B2",
                },
            )
            fig_roles.update_layout(
                showlegend=False,
                yaxis_title="메시지 수",
                xaxis_title="",
            )
            st.plotly_chart(fig_roles, use_container_width=True)

        with right:
            st.markdown("#### 역할별 글자 수")
            char_df = pd.DataFrame(
                [
                    {
                        "역할": "사용자",
                        "글자 수": sum(
                            len(m["content"]) for m in user_messages
                        ),
                    },
                    {
                        "역할": "친친티",
                        "글자 수": sum(
                            len(m["content"]) for m in assistant_messages
                        ),
                    },
                ]
            )
            fig_chars = px.pie(
                char_df,
                names="역할",
                values="글자 수",
                hole=0.45,
                color="역할",
                color_discrete_map={
                    "사용자": "#4C78A8",
                    "친친티": "#72B7B2",
                },
            )
            st.plotly_chart(fig_chars, use_container_width=True)

        st.markdown("#### 메시지별 글자 수 추이")

        timeline_df = pd.DataFrame(
            [
                {
                    "순서": i + 1,
                    "역할": "사용자" if m["role"] == "user" else "친친티",
                    "글자 수": len(m["content"]),
                }
                for i, m in enumerate(messages)
            ]
        )

        fig_timeline = px.line(
            timeline_df,
            x="순서",
            y="글자 수",
            color="역할",
            markers=True,
            color_discrete_map={
                "사용자": "#4C78A8",
                "친친티": "#72B7B2",
            },
        )
        fig_timeline.update_layout(
            xaxis_title="메시지 순서",
            yaxis_title="글자 수",
        )
        st.plotly_chart(fig_timeline, use_container_width=True)

        st.markdown("#### 대화 기록")

        history_df = pd.DataFrame(
            [
                {
                    "순서": i + 1,
                    "역할": "사용자" if m["role"] == "user" else "친친티",
                    "글자 수": len(m["content"]),
                    "내용": m["content"],
                    "시간": m.get("timestamp", ""),
                }
                for i, m in enumerate(messages)
            ]
        )

        st.dataframe(
            history_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "내용": st.column_config.TextColumn(
                    "내용", width="large"
                ),
                "글자 수": st.column_config.NumberColumn(
                    "글자 수", format="%d"
                ),
            },
        )

        csv_data = history_df.to_csv(
            index=False
        ).encode("utf-8-sig")

        st.download_button(
            "📥 대화 분석 CSV 다운로드",
            data=csv_data,
            file_name="chinchinti_chat_analysis.csv",
            mime="text/csv",
            use_container_width=True,
        )
