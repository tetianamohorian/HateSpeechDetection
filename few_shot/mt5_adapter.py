from lm_eval.base import BaseLM
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

class MT5Adapter(BaseLM):
    def __init__(self, model_name="google/mt5-small", device="cuda" if torch.cuda.is_available() else "cpu"):
        super().__init__()
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2).to(self.device)

    def loglikelihood(self, requests):
        # Not needed for classification tasks
        raise NotImplementedError

    def fewshot_predict(self, inputs):
        self.model.eval()
        with torch.no_grad():
            encoded_inputs = self.tokenizer(inputs, return_tensors="pt", padding=True, truncation=True, max_length=128).to(self.device)
            logits = self.model(**encoded_inputs).logits
            probs = torch.nn.functional.softmax(logits, dim=-1)
            predictions = torch.argmax(probs, dim=-1).cpu().numpy()
        return predictions
