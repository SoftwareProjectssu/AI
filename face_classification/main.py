import argparse
import os
import json
import numpy as np
import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from safetensors.torch import save_file

# 모델 클래스 임포트
from model import FaceShapeViT, FACE_SHAPE_CLASSES


def preprocess_image(image_path, device=None):
    """이미지 전처리 함수
    
    Args:
        image_path (str): 이미지 파일 경로
        device (torch.device): 텐서를 이동시킬 디바이스
    
    Returns:
        tuple: (원본 이미지, 전처리된 텐서)
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    # ViT 모델에 맞는 전처리
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    # 이미지 로드
    img = Image.open(image_path).convert('RGB')
    original_img = img.copy()
    
    # 텐서로 변환 및 배치 차원 추가
    img_tensor = transform(img).unsqueeze(0).to(device)
    
    return original_img, img_tensor


def predict_face_shape(model, image_path, classes=None, visualize=True, save_json=False):
    """얼굴형 예측 및 시각화 함수
    
    Args:
        model (nn.Module): 얼굴형 분류 모델
        image_path (str): 이미지 파일 경로
        classes (list): 클래스 이름 목록
        visualize (bool): 시각화 여부
        save_json (bool): JSON으로 결과 저장 여부
    
    Returns:
        tuple: (예측 클래스, 확률 배열)
    """
    if classes is None:
        classes = FACE_SHAPE_CLASSES
        
    device = next(model.parameters()).device
    
    # 이미지 전처리
    img, img_tensor = preprocess_image(image_path, device)
    
    # 모델 예측
    with torch.no_grad():
        outputs = model(img_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
        _, predicted_idx = torch.max(outputs, 1)
        predicted_class = classes[predicted_idx.item()]
    
    # 결과를 numpy 배열로 변환
    prob_array = probabilities.cpu().numpy()
    
    # 결과 시각화
    if visualize:
        plt.figure(figsize=(12, 5))
        
        # 원본 이미지 표시
        plt.subplot(1, 2, 1)
        plt.imshow(img)
        plt.title(f'예측된 얼굴형: {predicted_class}', fontsize=14)
        plt.axis('off')
        
        # 확률 분포 그래프
        plt.subplot(1, 2, 2)
        bars = plt.bar(classes, prob_array)
        plt.title('클래스별 확률', fontsize=14)
        plt.xticks(rotation=45, ha='right')
        plt.ylim(0, 1.0)
        
        # 확률값 표시
        for bar, prob in zip(bars, prob_array):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                    f'{prob:.2f}', ha='center', fontsize=10)
        
        plt.tight_layout()
        plt.show()
    
    # 결과 출력
    print(f"\n예측 결과: {predicted_class}\n")
    print("클래스별 확률:")
    for i, (face_shape, prob) in enumerate(zip(classes, prob_array)):
        print(f"{i+1}. {face_shape}: {prob:.4f}")
    
    # JSON으로 결과 저장
    if save_json:
        result = {
            "prediction": predicted_class,
            "probabilities": {cls: float(prob) for cls, prob in zip(classes, prob_array)}
        }
        
        # 이미지 파일 이름에서 JSON 파일 이름 생성
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        json_path = f"{base_name}_prediction.json"
        
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        
        print(f"예측 결과가 {json_path}에 저장되었습니다.")
        
    return predicted_class, prob_array


def batch_predict(model, image_dir, output_dir=None, classes=None):
    """디렉토리 내 모든 이미지에 대해 일괄 예측
    
    Args:
        model (nn.Module): 얼굴형 분류 모델
        image_dir (str): 이미지 디렉토리 경로
        output_dir (str): 결과 저장 디렉토리 경로
        classes (list): 클래스 이름 목록
    """
    if classes is None:
        classes = FACE_SHAPE_CLASSES
        
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
    
    # 지원되는 이미지 확장자
    valid_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    
    # 이미지 파일 목록
    image_files = [f for f in os.listdir(image_dir) 
                  if os.path.splitext(f.lower())[1] in valid_extensions]
    
    if not image_files:
        print(f"경고: {image_dir}에 이미지 파일이 없습니다.")
        return
    
    results = {}
    
    # 각 이미지에 대해 예측
    for img_file in image_files:
        img_path = os.path.join(image_dir, img_file)
        print(f"\n처리 중: {img_file}")
        
        try:
            predicted_class, probabilities = predict_face_shape(
                model, img_path, classes, visualize=False, save_json=False
            )
            
            # 결과 저장
            results[img_file] = {
                "prediction": predicted_class,
                "probabilities": {cls: float(prob) for cls, prob in zip(classes, probabilities)}
            }
            
        except Exception as e:
            print(f"오류 발생: {img_file} - {e}")
    
    # 결과 저장
    if output_dir is not None:
        result_path = os.path.join(output_dir, "batch_predictions.json")
        with open(result_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        print(f"\n모든 예측 결과가 {result_path}에 저장되었습니다.")
    
    return results


def convert_model(input_path, output_path, model_class=FaceShapeViT, num_classes=5):
    """모델 파일 형식 변환 (.pth <-> .safetensors)
    
    Args:
        input_path (str): 입력 모델 파일 경로
        output_path (str): 출력 모델 파일 경로
        model_class (nn.Module): 모델 클래스
        num_classes (int): 클래스 수
    """
    # 모델 로드
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model_class.from_pretrained(input_path, num_classes=num_classes, device=device)
    
    # 상태 딕셔너리 가져오기
    state_dict = model.state_dict()
    
    # 출력 형식에 따라 저장
    if output_path.endswith('.safetensors'):
        # .pth -> .safetensors
        save_file(state_dict, output_path)
    else:
        # .safetensors -> .pth
        torch.save({'state_dict': state_dict}, output_path)
    
    print(f"모델이 {output_path}로 변환되었습니다.")


def main():
    # 인자 파서 설정
    parser = argparse.ArgumentParser(description='얼굴형 분류 모델')
    
    # 명령 서브파서 설정
    subparsers = parser.add_subparsers(dest='command', help='명령')
    
    # 단일 이미지 예측 명령
    predict_parser = subparsers.add_parser('predict', help='단일 이미지 예측')
    predict_parser.add_argument('--model', type=str, required=True, help='모델 파일 경로')
    predict_parser.add_argument('--image', type=str, required=True, help='이미지 파일 경로')
    predict_parser.add_argument('--json', action='store_true', help='결과를 JSON으로 저장')
    
    # 배치 예측 명령
    batch_parser = subparsers.add_parser('batch', help='디렉토리 내 이미지 일괄 예측')
    batch_parser.add_argument('--model', type=str, required=True, help='모델 파일 경로')
    batch_parser.add_argument('--dir', type=str, required=True, help='이미지 디렉토리 경로')
    batch_parser.add_argument('--output', type=str, default='results', help='결과 저장 디렉토리')
    
    # 모델 변환 명령
    convert_parser = subparsers.add_parser('convert', help='모델 파일 형식 변환')
    convert_parser.add_argument('--input', type=str, required=True, help='입력 모델 파일 경로')
    convert_parser.add_argument('--output', type=str, required=True, help='출력 모델 파일 경로')
    
    # 인자 파싱
    args = parser.parse_args()
    
    # 명령에 따라 처리
    if args.command == 'predict':
        # 모델 로드
        model = FaceShapeViT.from_pretrained(args.model)
        
        # 예측 수행
        predict_face_shape(model, args.image, visualize=True, save_json=args.json)
        
    elif args.command == 'batch':
        # 모델 로드
        model = FaceShapeViT.from_pretrained(args.model)
        
        # 배치 예측 수행
        batch_predict(model, args.dir, args.output)
        
    elif args.command == 'convert':
        # 모델 변환
        convert_model(args.input, args.output)
        
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
