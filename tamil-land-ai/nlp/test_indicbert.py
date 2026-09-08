import torch
from transformers import AutoTokenizer, AutoModel

MODEL_NAME = "ai4bharat/IndicBERTv2-MLM-only"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Device:", device)

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
else:
    print("GPU: None")

print("\nLoading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

print("Loading IndicBERT...")
model = AutoModel.from_pretrained(MODEL_NAME)
model = model.to(device)
model.eval()

text = "சுப்பிரமணியன் மகன் ராமசாமி சர்வே எண் 124/3"

inputs = tokenizer(
    text,
    return_tensors="pt",
    truncation=True
)

inputs = {
    key: value.to(device)
    for key, value in inputs.items()
}

print("\nTamil text:")
print(text)

print("\nTokens:")
print(tokenizer.convert_ids_to_tokens(inputs["input_ids"][0]))

with torch.no_grad():
    outputs = model(**inputs)

print("\nOutput shape:")
print(outputs.last_hidden_state.shape)

print("\nModel is running on:")
print(outputs.last_hidden_state.device)

print("\n✓ IndicBERT + CUDA working!")