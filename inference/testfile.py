import json
import base64
with open('15.jpg', 'rb') as f:
    content = f.read()
    encoded2 = base64.b64encode(content).decode("utf-8")
with open('2.jpg', 'rb') as f:
    content = f.read()
    encoded3 = base64.b64encode(content).decode("utf-8")
with open('3.jpg', 'rb') as f:
    content = f.read()
    encoded = base64.b64encode(content).decode("utf-8")
    
request_dict = {'img': [encoded2,encoded,encoded3]}

from predict import predict
output = predict('request_dict')
print('point')
print(output)

