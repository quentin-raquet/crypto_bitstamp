import onnxruntime
import numpy as np

# Load the ONNX model
ort_session = onnxruntime.InferenceSession("model.onnx")

# Prepare the input data
input_data = np.array([[0.5, 0.6], [0.7, 0.8]], dtype=np.float32) # Example input data
input_name = ort_session.get_inputs()[0].name

# Run the inference
output_name = ort_session.get_outputs()[0].name
output_data = ort_session.run([output_name], {input_name: input_data})[0]

# Print the output
print(output_data)