from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline

tokenizer = AutoTokenizer.from_pretrained("vitus9988/klue-roberta-small-ner-identified")
model = AutoModelForTokenClassification.from_pretrained("vitus9988/klue-roberta-small-ner-identified")

nlp = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple")
example = """
저는 김철수입니다. 집은 서울특별시 강남대로이고 전화번호는 010-1234-5678, 주민등록번호는 123456-1234567입니다. 메일주소는 hugging@face.com입니다. 저는 10월 25일에 출국할 예정입니다.
"""
print(f"원본 {example}")
ner_results = nlp(example)

indices = [(res['start'], res['end']) for res in ner_results]

indices.sort(reverse=True)

for start, end in indices:
    example = example[:start] + '***' + example[end:]
  
print(f"후처리 {example}")


# 원본 
# 저는 김철수입니다. 집은 서울특별시 강남대로이고 전화번호는 010-1234-5678, 주민등록번호는 123456-1234567입니다. 메일주소는 hugging@face.com입니다. 저는 10월 25일에 출국할 예정입니다.

# 후처리 
# 저는 ***입니다. 집은 ***이고 전화번호는 ***, 주민등록번호는 ***입니다. 메일주소는 ***입니다. 저는 ***에 출국할 예정입니다.
