from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
import joblib


# pipeline is the model
# loading pretrained model
model2 = joblib.load("../output/model2.pkl")

print(model2)

# convert the datatype
initial_types = [('float_input', FloatTensorType([None,4]))]
onx = convert_sklearn(model2, initial_types=initial_types)

with open("../output/model2.onnx", "wb") as f:
    f.write(onx.SerializeToString())