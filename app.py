# 필요 패키지 설치
# pip install gradio transformers sentence-transformers faiss-cpu

import gradio as gr
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# 1️⃣ 법령 데이터 예시 (민법, 상법 일부)
documents = [
    {"title": "민법 제1조", "content": "민법은 사람과 사람 사이의 권리와 의무를 규정한다."},
    {"title": "민법 제2조", "content": "권리와 의무는 법률행위에 의하여 발생한다."},
    {"title": "상법 제1조", "content": "상법은 상인의 영업 및 상행위에 관한 규정을 다룬다."},
    {"title": "상법 제2조", "content": "상행위는 상인이 행하는 거래행위를 말한다."}
]

# 2️⃣ 임베딩 모델
embedding_model = SentenceTransformer("jhgan/ko-sroberta-multitask")

# 3️⃣ 벡터화 및 FAISS 인덱스 생성
corpus = [doc["content"] for doc in documents]
corpus_embeddings = embedding_model.encode(corpus, convert_to_numpy=True)

dim = corpus_embeddings.shape[1]
index = faiss.IndexFlatL2(dim)
index.add(corpus_embeddings)

# 4️⃣ LLM 로드 (KoAlpaca)
llm_model_name = "Beomi/KoAlpaca-Polyglot"
tokenizer = AutoTokenizer.from_pretrained(llm_model_name)
model = AutoModelForCausalLM.from_pretrained(llm_model_name, device_map="auto")

# 5️⃣ 질문 처리 함수
def legal_qa(question, top_k=2):
    # 5-1. 질문 임베딩
    q_embedding = embedding_model.encode([question], convert_to_numpy=True)
    
    # 5-2. 벡터 검색 (FAISS)
    D, I = index.search(q_embedding, top_k)
    relevant_docs = [documents[i]["content"] for i in I[0]]
    
    # 5-3. LLM 프롬프트 생성
    context = "\n".join(relevant_docs)
    prompt = f"아래 법령 내용을 참고하여 질문에 답해주세요:\n\n법령 내용:\n{context}\n\n질문: {question}\n답변:"
    
    # 5-4. 답변 생성
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=256, temperature=0.7, top_p=0.9)
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # 5-5. 프롬프트 제거 후 디스클레이머 추가
    if "답변:" in answer:
        answer = answer.split("답변:")[-1].strip()
    answer += "\n\n⚠️ 이 답변은 참고용이며, 실제 법적 판단은 변호사 상담이 필요합니다."
    
    return answer

# 6️⃣ Gradio 인터페이스
iface = gr.Interface(
    fn=legal_qa,
    inputs=gr.Textbox(label="법률 질문 입력", placeholder="예: 월세 계약을 해지하려면 어떻게 해야 하나요?"),
    outputs=gr.Textbox(label="답변"),
    title="한국어 법률 Q&A 챗봇",
    description="KoAlpaca + 임베딩 기반 법령 검색 후 LLM이 요약 답변"
)

if __name__ == "__main__":
    iface.launch()
