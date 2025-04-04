import os
import torch
import torch.nn as nn
import torchvision.models as models
from flask import Flask, render_template, request
from PIL import Image
import torchvision.transforms as transforms
from io import BytesIO
import numpy as np
import pandas as pd

app = Flask(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet50(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, 81)

# 환경변수에서 모델 경로를 받아오거나 기본값 사용
model_path = os.getenv('MODEL_PATH', 'best_skin_model.pth')
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def allowed_file(filename):
    """Check if the file has an allowed extension."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'png', 'jpg', 'jpeg'}


# Prediction function
def predict_skin_condition(image, age, gender):
    image = transform(image).unsqueeze(0).to(device)  # Add batch dimension
    with torch.no_grad():
        outputs = model(image)  # Model prediction
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

# 각 항목을 비교할 때 전체 항목의 평균을 구해 피드백을 하나만 제공
def peer_group_analysis(age, gender, user_skin_analysis):
    # 나이 범위 설정 (예: 나이 ±3)
    min_age = age - 3
    max_age = age + 3
    
    # 예시 URL은 GitHub 저장소의 Raw 주소를 사용해야 함
    url = "https://raw.githubusercontent.com/kovas-strategy/my_skin_project/master/all_data.xlsx"
    all_data = pd.read_excel(url)
    
    # 나이 범위와 성별에 맞는 그룹 필터링
    peer_group = all_data[(all_data['성별'] == gender) & (all_data['나이'] >= min_age) & (all_data['나이'] <= max_age)]
    
    # 피어 그룹의 피부 상태 항목 추출 (모든 피부 상태 항목을 포함하도록 수정)
    peer_group_skin = peer_group.iloc[:, 1:]  # 데이터의 모든 컬럼을 가져오도록 수정 (1열부터 끝까지)
    
    # 각 항목별로 peer 그룹 평균을 계산
    peer_group_mean = peer_group_skin.mean(axis=0)
    
    # 사용자 피부 상태와 피어 그룹 평균 비교하여 피드백 제공
    feedback = {}

    # 각 항목을 비교할 때 전체 항목의 평균을 구해 피드백을 하나만 제공
    def compare_with_peer_group(features, category, is_better_low=True):
        user_values = [user_skin_analysis[feature] for feature in features]
        peer_mean = peer_group_mean[features].mean()
        user_avg = np.mean(user_values)
        
        if is_better_low:  # 값이 낮을수록 좋은 경우
            if user_avg < peer_mean:
                feedback[category] = f"Your {category} is better than average."
            elif user_avg > peer_mean:
                feedback[category] = f"Your {category} is below average. You may want to focus on improving this area."
            else:
                feedback[category] = f"Your {category} is right around average."
        else:  # 값이 높을수록 좋은 경우
            if user_avg > peer_mean:
                feedback[category] = f"Your {category} is better than average."
            elif user_avg < peer_mean:
                feedback[category] = f"Your {category} is below average. You may want to focus on improving this area."
            else:
                feedback[category] = f"Your {category} is right around average."

    # 수분 관련 항목 (평균값이 낮으면 좋음)
    moisture_features = [
        '수분_이마', '수분_오른쪽볼', '수분_왼쪽볼', '수분_턱'
    ]
    compare_with_peer_group(moisture_features, "수분", is_better_low=True)

    # 탄력 관련 항목 (평균값이 높으면 좋음)
    elasticity_chin_features = [
        '탄력_턱_R0', '탄력_턱_R1', '탄력_턱_R2', '탄력_턱_R3',
        '탄력_턱_R4', '탄력_턱_R5', '탄력_턱_R6', '탄력_턱_R7',
        '탄력_턱_R8', '탄력_턱_R9', '탄력_턱_Q0', '탄력_턱_Q1',
        '탄력_턱_Q2', '탄력_턱_Q3'
    ]
    compare_with_peer_group(elasticity_chin_features, "탄력_턱", is_better_low=False)

    # 왼쪽 볼 탄력 관련 항목 (평균값이 높으면 좋음)
    elasticity_cheek_left_features = [
        '탄력_왼쪽볼_R0', '탄력_왼쪽볼_R1', '탄력_왼쪽볼_R2', 
        '탄력_왼쪽볼_R3', '탄력_왼쪽볼_R4', '탄력_왼쪽볼_R5',
        '탄력_왼쪽볼_R6', '탄력_왼쪽볼_R7', '탄력_왼쪽볼_R8',
        '탄력_왼쪽볼_R9', '탄력_왼쪽볼_Q0', '탄력_왼쪽볼_Q1',
        '탄력_왼쪽볼_Q2', '탄력_왼쪽볼_Q3'
    ]
    compare_with_peer_group(elasticity_cheek_left_features, "탄력_왼쪽볼", is_better_low=False)

    # 오른쪽 볼 탄력 관련 항목 (평균값이 높으면 좋음)
    elasticity_cheek_right_features = [
        '탄력_오른쪽볼_R0', '탄력_오른쪽볼_R1', '탄력_오른쪽볼_R2',
        '탄력_오른쪽볼_R3', '탄력_오른쪽볼_R4', '탄력_오른쪽볼_R5',
        '탄력_오른쪽볼_R6', '탄력_오른쪽볼_R7', '탄력_오른쪽볼_R8',
        '탄력_오른쪽볼_R9', '탄력_오른쪽볼_Q0', '탄력_오른쪽볼_Q1',
        '탄력_오른쪽볼_Q2', '탄력_오른쪽볼_Q3'
    ]
    compare_with_peer_group(elasticity_cheek_right_features, "탄력_오른쪽볼", is_better_low=False)

    # 이마 탄력 관련 항목 (평균값이 높으면 좋음)
    elasticity_forehead_features = [
        '탄력_이마_R0', '탄력_이마_R1', '탄력_이마_R2', 
        '탄력_이마_R3', '탄력_이마_R4', '탄력_이마_R5',
        '탄력_이마_R6', '탄력_이마_R7', '탄력_이마_R8',
        '탄력_이마_R9', '탄력_이마_Q0', '탄력_이마_Q1',
        '탄력_이마_Q2', '탄력_이마_Q3'
    ]
    compare_with_peer_group(elasticity_forehead_features, "탄력_이마", is_better_low=False)

    # 주름 관련 항목 (평균값이 낮으면 좋음)
    wrinkle_eye_left_features = [
        '주름_왼쪽눈가_Ra', '주름_왼쪽눈가_Rq', '주름_왼쪽눈가_Rmax', '주름_왼쪽눈가_R3z',
        '주름_왼쪽눈가_Rt', '주름_왼쪽눈가_Rz=Rtm', '주름_왼쪽눈가_Rp', '주름_왼쪽눈가_Rv'
    ]
    compare_with_peer_group(wrinkle_eye_left_features, "주름_왼쪽눈가", is_better_low=True)

    # 주름 관련 항목 (평균값이 낮으면 좋음)
    wrinkle_eye_right_features = [
        '주름_오른쪽눈가_Ra', '주름_오른쪽눈가_Rq', '주름_오른쪽눈가_Rmax', '주름_오른쪽눈가_R3z',
        '주름_오른쪽눈가_Rt', '주름_오른쪽눈가_Rz=Rtm', '주름_오른쪽눈가_Rp', '주름_오른쪽눈가_Rv'
    ]
    compare_with_peer_group(wrinkle_eye_right_features, "주름_오른쪽눈가", is_better_low=True)

    # 스팟개수, 모공개수 (평균값이 낮으면 좋음)
    other_features = [
        '스팟개수_정면', '모공개수_오른쪽볼', '모공개수_왼쪽볼'
    ]
    compare_with_peer_group(other_features, "스팟개수/모공개수", is_better_low=True)

    return feedback


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files.get('file')
        age = request.form.get('age')
        gender = request.form.get('gender')

        if file and allowed_file(file.filename) and age is not None and gender is not None:
            try:
                image = Image.open(BytesIO(file.read())).convert('RGB')
                age = int(age)
                gender = int(gender)
                skin_analysis = predict_skin_condition(image, age, gender)
                feedback = peer_group_analysis(age, gender, skin_analysis)  # 생성한 분석 결과를 feedback 변수에 저장
                return render_template('result.html', feedback=feedback)  # feedback 변수를 템플릿으로 전달
            except Exception as e:
                return render_template('index.html', error=f"Error processing the image or data: {str(e)}")
        else:
            return render_template('index.html', error="Please ensure all fields are correctly filled and the file is an image.")
    return render_template('index.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
   