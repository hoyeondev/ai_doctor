from transformers import pipeline

sentiment_model = pipeline(model="WhitePeak/bert-base-cased-Korean-sentiment")
# sentiment_model("매우 좋아")

result = sentiment_model("매우 좋아")
print(result)