import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as transforms
import cv2
from matplotlib import cm
import json

def get_device():
    """사용 가능한 장치(CPU/GPU) 반환"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')

def visualize_attention(model, image_path, output_path=None, alpha=0.6):
    """ViT 모델의 어텐션 맵 시각화
    
    Args:
        model (nn.Module): ViT 모델
        image_path (str): 입력 이미지 경로
        output_path (str): 출력 이미지 경로 (None이면 화면에 표시)
        alpha (float): 어텐션 맵 투명도
    """
    device = next(model.parameters()).device
    
    # 이미지 로드 및 전처리
    img = Image.open(image_path).convert('RGB')
    img_tensor = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])(img).unsqueeze(0).to(device)
    
    # 어텐션 가중치 추출 (모델에 따라 구현이 달라질 수 있음)
    model.eval()
    with torch.no_grad():
        # 여기서는 예시로 마지막 블록의 첫 번째 어텐션 헤드 사용
        # 실제 구현은 모델 구조에 맞게 조정 필요
        attentions = []
        
        def hook_fn(module, input, output):
            attentions.append(output[1])  # 어텐션 가중치
        
        # 어텐션 블록에 훅 등록 (timm 모델에 맞게 조정 필요)
        if hasattr(model.vit, 'blocks'):
            last_attn = model.vit.blocks[-1].attn
            if hasattr(last_attn, 'register_forward_hook'):
                hook = last_attn.register_forward_hook(hook_fn)
                
                # 전방 전파
                _ = model(img_tensor)
                
                # 훅 제거
                hook.remove()
            else:
                print("경고: 어텐션 레이어에 훅을 등록할 수 없습니다.")
                return
        else:
            print("경고: 모델 구조에서 어텐션 블록을 찾을 수 없습니다.")
            return
    
    if not attentions:
        print("경고: 어텐션 맵을 추출할 수 없습니다.")
        return
    
    # 첫 번째 어텐션 헤드 선택
    attn_weights = attentions[0][0, 0, 1:, 1:]  # CLS 토큰 제외
    
    # 어텐션 맵을 이미지 크기로 변환
    attn_map = attn_weights.mean(dim=0)
    size = int(attn_map.size(0) ** 0.5)
    attn_map = attn_map.reshape(size, size).cpu().numpy()
    
    # 어텐션 맵 정규화
    attn_map = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min())
    
    # 어텐션 맵 리사이즈
    attn_map = cv2.resize(attn_map, (img.width, img.height))
    
    # 시각화
    plt.figure(figsize=(12, 5))
    
    # 원본 이미지
    plt.subplot(1, 3, 1)
    plt.imshow(img)
    plt.title("원본 이미지")
    plt.axis('off')
    
    # 어텐션 맵
    plt.subplot(1, 3, 2)
    plt.imshow(attn_map, cmap='jet')
    plt.title("어텐션 맵")
    plt.axis('off')
    
    # 오버레이
    plt.subplot(1, 3, 3)
    img_np = np.array(img) / 255.0
    heatmap = cm.jet(attn_map)[:, :, :3]
    overlay = (1 - alpha) * img_np + alpha * heatmap
    plt.imshow(overlay)
    plt.title("오버레이 결과")
    plt.axis('off')
    
    plt.tight_layout()
    
    # 저장 또는 표시
    if output_path:
        plt.savefig(output_path, bbox_inches='tight')
        print(f"시각화 결과가 {output_path}에 저장되었습니다.")
    else:
        plt.show()


def export_results_to_csv(results, filename='predictions.csv'):
    """예측 결과를 CSV 파일로 내보내기
    
    Args:
        results (dict): 이미지 파일명을 키로 하고 예측 결과를 값으로 하는 딕셔너리
        filename (str): 출력 CSV 파일 이름
    """
    import csv
    
    with open(filename, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        
        # 헤더 작성
        first_result = next(iter(results.values()))
        classes = list(first_result['probabilities'].keys())
        header = ['이미지', '예측 클래스'] + classes
        writer.writerow(header)
        
        # 각 이미지의 예측 결과 작성
        for img_file, result in results.items():
            row = [img_file, result['prediction']]
            probabilities = [result['probabilities'][cls] for cls in classes]
            row.extend(probabilities)
            writer.writerow(row)
    
    print(f"결과가 {filename}에 저장되었습니다.")


# 사용 예시
if __name__ == "__main__":
    print("얼굴형 분류 모델 유틸리티 모듈입니다.")
