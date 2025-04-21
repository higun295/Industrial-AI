import os
import json
import re

# 대상 경로
json_dir = r"D:\Data\newFinalData"

# 모든 json 파일 순회
for filename in os.listdir(json_dir):
    if filename.endswith(".json"):
        match = re.search(r"Activity(\d+)", filename)
        if not match:
            print(f"⚠️ Activity 번호 없음 → 스킵: {filename}")
            continue

        activity_num = int(match.group(1))
        new_label = 0 if activity_num <= 5 else 1

        file_path = os.path.join(json_dir, filename)

        # JSON 로드
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # label 수정
        for item in data:
            item["label"] = new_label

        # 덮어쓰기 저장
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

        print(f"✅ 라벨 변경 완료: {filename} → label={new_label}")
