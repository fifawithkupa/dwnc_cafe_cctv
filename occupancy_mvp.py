"""
SeatNow 점유 판정 MVP
- 테이블 ROI 내부에 person이 아닌 객체가 있거나
- 테이블 ROI 내부에 사람이 있으면
→ 점유로 판정

사용법:
  python3 occupancy_mvp.py /경로/이미지.png
"""

import sys
from ultralytics import YOLO


# 점유 신호로 치지 않는 클래스 (가구/고정 집기 = 좌석 자체이거나 항상 있는 것)
EXCLUDE = {
    'chair', 'couch', 'dining table', 'bench',   # 좌석/테이블 자체
    'potted plant', 'tv', 'refrigerator',        # 고정 집기
    'cat', 'dog',                                 # 동물 (점유로 보기 애매)
}


def center(box):
    """bounding box의 중심 좌표 (x, y)"""
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2, (y1 + y2) / 2)


def point_in_box(point, box, margin=0.0):
    """점이 box 안에 있는지 (margin만큼 box를 확장해서 판정)"""
    px, py = point
    x1, y1, x2, y2 = box
    w, h = (x2 - x1), (y2 - y1)
    x1 -= w * margin
    x2 += w * margin
    y1 -= h * margin
    y2 += h * margin
    return x1 <= px <= x2 and y1 <= py <= y2


def main(image_path):
    model = YOLO('yolov8x.pt')
    names = model.names

    results = model(image_path, conf=0.25, verbose=False)
    boxes = results[0].boxes

    # 객체들을 테이블 / 사람 / 기타물체로 분류
    tables = []   # dining table
    persons = []  # person
    objects = []  # 그 외 모든 것 (cup, phone, bag 등)

    for b in boxes:
        cls_name = names[int(b.cls)]
        xyxy = b.xyxy[0].tolist()
        conf = float(b.conf)
        item = {'name': cls_name, 'box': xyxy, 'conf': conf, 'center': center(xyxy)}

        if cls_name == 'dining table':
            tables.append(item)
        elif cls_name == 'person':
            persons.append(item)
        elif cls_name in EXCLUDE:
            continue  # 의자/소파/화분 등은 점유 신호로 안 침
        else:
            objects.append(item)

    print(f"\n탐지 요약: 테이블 {len(tables)}개, 사람 {len(persons)}명, 기타물체 {len(objects)}개")
    print("=" * 50)

    if not tables:
        print("테이블이 탐지되지 않음. (ROI 기준이 없어 판정 불가)")
        return

    # 각 테이블마다 점유 판정
    for i, table in enumerate(tables, 1):
        # ROI를 테이블보다 약간 넓게 잡음 (의자/물체가 테두리에 걸칠 수 있어서)
        margin = 0.3

        # ROI 안의 기타 물체
        objs_in = [o for o in objects if point_in_box(o['center'], table['box'], margin)]
        # ROI 안의 사람
        persons_in = [p for p in persons if point_in_box(p['center'], table['box'], margin)]

        occupied = len(objs_in) > 0 or len(persons_in) > 0
        status = "🔴 점유" if occupied else "🟢 비어있음"

        print(f"\n테이블 #{i} (신뢰도 {table['conf']:.2f}) → {status}")
        if objs_in:
            obj_names = [f"{o['name']}({o['conf']:.2f})" for o in objs_in]
            print(f"   물체: {', '.join(obj_names)}")
        if persons_in:
            print(f"   사람: {len(persons_in)}명")
        if not occupied:
            print(f"   (ROI 안에 사람도 물체도 없음)")

    # 결과 이미지 저장
    out = 'occupancy_result.jpg'
    results[0].save(filename=out)
    print(f"\n탐지 이미지 저장됨: {out}")


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("사용법: python3 occupancy_mvp.py /경로/이미지.png")
        sys.exit(1)
    main(sys.argv[1])
