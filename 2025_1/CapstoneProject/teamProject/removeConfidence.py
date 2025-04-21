import os
import json

# 폴더 경로
json_dir = r"D:\Data\FinalData_Without_Confidence"

# JSON 파일 순회
for file in os.listdir(json_dir):
    if file.endswith(".json"):
        file_path = os.path.join(json_dir, file)

        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        for item in data:
            keypoints = item.get("keypoints", [])
            # 3개씩 묶여있는 [x, y, confidence] 중 x, y만 추출
            xy_only = []
            for i in range(0, len(keypoints), 3):
                xy_only.extend(keypoints[i:i+2])  # x, y만
            item["keypoints"] = xy_only

        # 덮어쓰기 저장
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

        print(f"✅ confidence 제거 완료: {file}")
