터미널에서

1. 가상환경 비활성화
   deactivate

2. 기존 가상환경 삭제
   rm -rf ./env

3. 3.11. 버전으로 가상환경 생성
   python3.11 -m venv ./env

4. 가상환경 활성화
   source ./env/bin/activate

5. 다시 pyhon 버전 확인
   python --version

huggingface 모델 주소:

- 일반 모델 -
  https://huggingface.co/mistralai/Mistral-7B-v0.1

- instruct 모델 (instruction[지시]를 따르도록 세밀하게 조정됨)-
  https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.1

Mistral AI document 주소:
https://docs.mistral.ai/category/large-language-models
https://docs.mistral.ai/models/#chat-template

#RAG

Prompt
Question : {q}
Context : {docs}
