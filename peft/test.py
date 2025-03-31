from transformers import AutoTokenizer, AutoModelForSequenceClassification
from lm_eval import evaluator

# Model name
model_name = "ApoTro/slovak-t5-small"

# Load tokenizer and model
try:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    print("Model and tokenizer loaded successfully.")
except Exception as e:
    print(f"Error loading model or tokenizer: {e}")
    exit(1)

# Function to evaluate model
def evaluate_model():
    try:
        # Define task
        task_name = "hate_speech"  # Adjust based on available tasks

        # Get the task dictionary with required argument
        task_dict = evaluator.get_task_dict([task_name])

        # Ensure task exists
        if task_name not in task_dict:
            print(f"Task '{task_name}' not found in the task dictionary!")
            return

        # Run evaluation
        results = evaluator.simple_evaluate(
            model="hf-causal",
            model_args=f"pretrained={model_name}",
            tasks=[task_name],
            batch_size=8  # Adjust based on resources
        )

        # Print evaluation results
        print("Evaluation Results:")
        print(results)

        # Display common metrics
        precision = results.get("precision")
        recall = results.get("recall")
        f1 = results.get("f1")

        if precision and recall and f1:
            print(f"Precision: {precision:.4f}")
            print(f"Recall: {recall:.4f}")
            print(f"F1-Score: {f1:.4f}")
        else:
            print("Metrics not available. Check evaluation setup.")

    except Exception as e:
        print(f"Error during evaluation: {e}")

# Run evaluation
if __name__ == "__main__":
    evaluate_model()
