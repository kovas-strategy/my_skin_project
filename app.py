from flask import Flask, render_template, request
import torch
import pandas as pd
from PIL import Image
import torchvision.transforms as transforms
from torchvision import models
import torch.nn as nn
import os
import numpy as np
from io import BytesIO

# Flask 앱 설정
app = Flask(__name__)

# 모델 로딩
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = models.resnet50(weights=None)  # pretrained=False 대신 weights=None 사용
model.fc = nn.Linear(model.fc.in_features, 81)

# Load model state_dict
model.load_state_dict(torch.load("best_skin_model.pth", map_location=device))

model = model.to(device)
model.eval()  # 평가 모드

# 이미지 전처리
transform = transforms.Compose([
    transforms.Resize((400, 600)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# 피부 상태 예측 함수
def predict_skin_condition(image, age, gender):
    image = transform(image).unsqueeze(0).to(device)  # 배치 차원 추가
    
    with torch.no_grad():
        outputs = model(image)  # 모델 예측
    
    predicted_skin = outputs.squeeze(0).cpu().numpy()
    
    # 피부 상태 분석 (모든 항목 포함)
    skin_analysis = {
        "수분_이마": predicted_skin[0],
        "수분_오른쪽볼": predicted_skin[1],
        "수분_왼쪽볼": predicted_skin[2],
        "수분_턱": predicted_skin[3],
        "탄력_턱_R0": predicted_skin[4],
        "탄력_턱_R1": predicted_skin[5],
        "탄력_턱_R2": predicted_skin[6],
        "탄력_턱_R3": predicted_skin[7],
        "탄력_턱_R4": predicted_skin[8],
        "탄력_턱_R5": predicted_skin[9],
        "탄력_턱_R6": predicted_skin[10],
        "탄력_턱_R7": predicted_skin[11],
        "탄력_턱_R8": predicted_skin[12],
        "탄력_턱_R9": predicted_skin[13],
        "탄력_턱_Q0": predicted_skin[14],
        "탄력_턱_Q1": predicted_skin[15],
        "탄력_턱_Q2": predicted_skin[16],
        "탄력_턱_Q3": predicted_skin[17],
        "탄력_왼쪽볼_R0": predicted_skin[18],
        "탄력_왼쪽볼_R1": predicted_skin[19],
        "탄력_왼쪽볼_R2": predicted_skin[20],
        "탄력_왼쪽볼_R3": predicted_skin[21],
        "탄력_왼쪽볼_R4": predicted_skin[22],
        "탄력_왼쪽볼_R5": predicted_skin[23],
        "탄력_왼쪽볼_R6": predicted_skin[24],
        "탄력_왼쪽볼_R7": predicted_skin[25],
        "탄력_왼쪽볼_R8": predicted_skin[26],
        "탄력_왼쪽볼_R9": predicted_skin[27],
        "탄력_왼쪽볼_Q0": predicted_skin[28],
        "탄력_왼쪽볼_Q1": predicted_skin[29],
        "탄력_왼쪽볼_Q2": predicted_skin[30],
        "탄력_왼쪽볼_Q3": predicted_skin[31],
        "탄력_오른쪽볼_R0": predicted_skin[32],
        "탄력_오른쪽볼_R1": predicted_skin[33],
        "탄력_오른쪽볼_R2": predicted_skin[34],
        "탄력_오른쪽볼_R3": predicted_skin[35],
        "탄력_오른쪽볼_R4": predicted_skin[36],
        "탄력_오른쪽볼_R5": predicted_skin[37],
        "탄력_오른쪽볼_R6": predicted_skin[38],
        "탄력_오른쪽볼_R7": predicted_skin[39],
        "탄력_오른쪽볼_R8": predicted_skin[40],
        "탄력_오른쪽볼_R9": predicted_skin[41],
        "탄력_오른쪽볼_Q0": predicted_skin[42],
        "탄력_오른쪽볼_Q1": predicted_skin[43],
        "탄력_오른쪽볼_Q2": predicted_skin[44],
        "탄력_오른쪽볼_Q3": predicted_skin[45],
        "탄력_이마_R0": predicted_skin[46],
        "탄력_이마_R1": predicted_skin[47],
        "탄력_이마_R2": predicted_skin[48],
        "탄력_이마_R3": predicted_skin[49],
        "탄력_이마_R4": predicted_skin[50],
        "탄력_이마_R5": predicted_skin[51],
        "탄력_이마_R6": predicted_skin[52],
        "탄력_이마_R7": predicted_skin[53],
        "탄력_이마_R8": predicted_skin[54],
        "탄력_이마_R9": predicted_skin[55],
        "탄력_이마_Q0": predicted_skin[56],
        "탄력_이마_Q1": predicted_skin[57],
        "탄력_이마_Q2": predicted_skin[58],
        "탄력_이마_Q3": predicted_skin[59],
        "주름_왼쪽눈가_Ra": predicted_skin[60],
        "주름_왼쪽눈가_Rq": predicted_skin[61],
        "주름_왼쪽눈가_Rmax": predicted_skin[62],
        "주름_왼쪽눈가_R3z": predicted_skin[63],
        "주름_왼쪽눈가_Rt": predicted_skin[64],
        "주름_왼쪽눈가_Rz=Rtm": predicted_skin[65],
        "주름_왼쪽눈가_Rp": predicted_skin[66],
        "주름_왼쪽눈가_Rv": predicted_skin[67],
        "주름_오른쪽눈가_Ra": predicted_skin[68],
        "주름_오른쪽눈가_Rq": predicted_skin[69],
        "주름_오른쪽눈가_Rmax": predicted_skin[70],
        "주름_오른쪽눈가_R3z": predicted_skin[71],
        "주름_오른쪽눈가_Rt": predicted_skin[72],
        "주름_오른쪽눈가_Rz=Rtm": predicted_skin[73],
        "주름_오른쪽눈가_Rp": predicted_skin[74],
        "주름_오른쪽눈가_Rv": predicted_skin[75],
        "스팟개수_정면": predicted_skin[76],
        "모공개수_오른쪽볼": predicted_skin[77],
        "모공개수_왼쪽볼": predicted_skin[78]
    }

    return skin_analysis

# Peer 그룹 분석 함수
def peer_group_analysis(age, gender, user_skin_analysis):
    min_age = age - 3
    max_age = age + 3
    all_data = pd.read_excel('https://raw.githubusercontent.com/kovas-strategy/my_skin_project/master/all_data.xlsx') # GitHub에서 불러오기
    peer_group = all_data[(all_data['성별'] == gender) & (all_data['나이'] >= min_age) & (all_data['나이'] <= max_age)]
    peer_group_skin = peer_group.iloc[:, 1:]
    peer_group_mean = peer_group_skin.mean(axis=0)
    feedback = {}

    def compare_with_peer_group(features, category, is_better_low=True):
        user_values = [user_skin_analysis[feature] for feature in features]
        peer_mean = peer_group_mean[features].mean()
        user_avg = np.mean(user_values)

        if is_better_low:
            if user_avg < peer_mean:
                feedback[category] = f"Your {category} is better than average."
            elif user_avg > peer_mean:
                feedback[category] = f"Your {category} is above average. You may want to focus on improving this area."
            else:
                feedback[category] = f"Your {category} is right around average."
        else:
            if user_avg > peer_mean:
                feedback[category] = f"Your {category} is better than average."
            elif user_avg < peer_mean:
                feedback[category] = f"Your {category} is below average. You may want to focus on improving this area."
            else:
                feedback[category] = f"Your {category} is right around average."

    # 수분, 탄력, 주름 등 여러 항목을 분석
    moisture_features = ['수분_이마', '수분_오른쪽볼', '수분_왼쪽볼', '수분_턱']
    compare_with_peer_group(moisture_features, "수분", is_better_low=True)

    elasticity_features = ['탄력_턱_R0', '탄력_턱_R1', '탄력_턱_R2', '탄력_턱_R3']
    compare_with_peer_group(elasticity_features, "탄력_턱", is_better_low=False)

    wrinkle_features = ['주름_왼쪽눈가_Ra', '주름_왼쪽눈가_Rq', '주름_오른쪽눈가_Ra', '주름_오른쪽눈가_Rq']
    compare_with_peer_group(wrinkle_features, "주름", is_better_low=True)

    return feedback

# 웹페이지 라우팅
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return 'No file part'
        
        file = request.files['file']
        
        if file and allowed_file(file.filename):
            image = Image.open(BytesIO(file.read())).convert('RGB')
            age = int(request.form['age'])
            gender = int(request.form['gender'])

            skin_analysis = predict_skin_condition(image, age, gender)
            feedback = peer_group_analysis(age, gender, skin_analysis)

            return render_template('result.html', skin_analysis=skin_analysis, feedback=feedback)

    return render_template('index.html')

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'png', 'jpg', 'jpeg'}

# Flask 애플리케이션을 실행하는 부분 제거 (Gunicorn에서 실행)
if __name__ == '__main__':
    pass  # 이제 gunicorn이 이 앱을 실행합니다
