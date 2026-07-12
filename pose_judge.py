"""
SeatNow 포즈 판정 테스트
YOLOv8-pose의 17개 keypoint로 앉음/서있음 판정 (Seatify 논문 방식)

COCO keypoint 인덱스:
  5 left_shoulder   6 right_shoulder
  11 left_hip       12 right_hip
  13 left_knee      14 right_knee
  15 left_ankle     16 right_ankle

판정 (3가지 테스트의 OR 조합):
  Test1. Hip-Knee-Ankle 각도 < 110°       → 앉음
  Test2. Torso(Shoulder-Hip-Knee) 각도 < 110° → 앉음
  Test3. Hip-Knee 수직거리가 작음          → 앉음

사용법:
  python3 pose_judge.py "/경로/이미지.png"
"""

import sys
import math
from ultralytics import YOLO

# keypoint 인덱스
L_SHO, R_SHO = 5, 6
L_HIP, R_HIP = 11, 12
L_KNE, R_KNE = 13, 14
L_ANK, R_ANK = 15, 16

CONF_TH = 0.3  # keypoint 신뢰도 임계값 (이하면 '안 보임'으로 처리)


def angle(a, b, c):
    """점 b를 꼭짓점으로 하는 a-b-c 각도(도). 좌표 없으면 None"""
    if a is None or b is None or c is None:
        return None
    ba = (a[0] - b[0], a[1] - b[1])
    bc = (c[0] - b[0], c[1] - b[1])
    dot = ba[0] * bc[0] + ba[1] * bc[1]
    mag = math.hypot(*ba) * math.hypot(*bc)
    if mag == 0:
        return None
    cosang = max(-1, min(1, dot / mag))
    return math.degrees(math.acos(cosang))


def get_kp(kpts, idx):
    """신뢰도 충분한 keypoint만 (x,y) 반환, 아니면 None"""
    x, y, c = kpts[idx]
    return (x, y) if c >= CONF_TH else None


def judge_person(kpts):
    """한 사람의 keypoint로 앉음 여부 판정. (앉음bool, 통과한 테스트 리스트)"""
    passed = []

    # 좌우 중 더 잘 보이는 쪽 사용 (양쪽 다 시도)
    sides = [
        (L_SHO, L_HIP, L_KNE, L_ANK),
        (R_SHO, R_HIP, R_KNE, R_ANK),
    ]

    sitting = False
    for sho_i, hip_i, kne_i, ank_i in sides:
        sho = get_kp(kpts, sho_i)
        hip = get_kp(kpts, hip_i)
        kne = get_kp(kpts, kne_i)
        ank = get_kp(kpts, ank_i)

        # Test1. Hip-Knee-Ankle 각도
        hka = angle(hip, kne, ank)
        if hka is not None and hka < 110:
            sitting = True
            passed.append(f"HKA={hka:.0f}°<110")

        # Test2. Torso(Shoulder-Hip-Knee) 각도
        torso = angle(sho, hip, kne)
        if torso is not None and torso < 110:
            sitting = True
            passed.append(f"Torso={torso:.0f}°<110")

        # Test3. Hip-Knee 수직거리 → 오판이 잦아 제거
        # (서서 상체를 숙이면 이 비율도 작아져서 서있는 사람을 앉음으로 오판함.
        #  각도 테스트 2개가 더 신뢰도 높아 그것만 사용)

    return sitting, passed


def main(image_path):
    model = YOLO('yolov8x-pose.pt')
    results = model(image_path, verbose=False)
    r = results[0]

    if r.keypoints is None or len(r.keypoints) == 0:
        print("사람이 탐지되지 않음")
        return

    kpts_all = r.keypoints.data  # (N, 17, 3)
    boxes = r.boxes

    print(f"\n탐지된 사람: {len(kpts_all)}명")
    print("=" * 50)

    for i in range(len(kpts_all)):
        kpts = kpts_all[i].tolist()
        conf = float(boxes.conf[i]) if boxes is not None else 0
        sitting, passed = judge_person(kpts)
        status = "🪑 앉음" if sitting else "🚶 서있음/기타"
        print(f"\n사람 #{i+1} (탐지 {conf:.2f}) → {status}")
        if passed:
            print(f"   통과한 테스트: {', '.join(passed)}")
        else:
            print(f"   (앉음 조건 충족 안 됨)")

    out = 'pose_judge_result.jpg'
    r.save(filename=out)
    print(f"\n스켈레톤 이미지 저장됨: {out}")


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('사용법: python3 pose_judge.py "/경로/이미지.png"')
        sys.exit(1)
    main(sys.argv[1])
