from transformers import TFGPT2LMHeadModel, GPT2Tokenizer
import tensorflow as tf

# Load model + tokenizer
model = TFGPT2LMHeadModel.from_pretrained("distilgpt2")
tokenizer = GPT2Tokenizer.from_pretrained("distilgpt2")

# Freeze model for export
@tf.function(input_signature=[tf.TensorSpec([1, None], tf.int32, name="input_ids")])
def model_fn(input_ids):
    return model(input_ids).logits

# Convert
concrete_func = model_fn.get_concrete_function()
converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]  # no flex ops
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

# Save it
with open("gpt2_model.tflite", "wb") as f:
    f.write(tflite_model)

print("✅ GPT-2 TFLite model saved successfully.")
