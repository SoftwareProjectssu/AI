import torch
import torch.nn as nn
import timm
from safetensors.torch import load_file

class FaceShapeViT(nn.Module):
    """Vision Transformer 기반 얼굴형 분류 모델"""
    
    def __init__(self, num_classes=5):
        """
        Args:
            num_classes (int): 분류할 얼굴형 클래스 수
        """
        super(FaceShapeViT, self).__init__()
        
        # timm 라이브러리를 사용한 ViT 모델
        self.vit = timm.create_model(
            'vit_base_patch16_224',
            pretrained=False,
            num_classes=0  # 분류 헤드 제거
        )
        
        # 별도의 classifier 레이어 추가
        self.classifier = nn.Linear(768, num_classes)
    
    def forward(self, x):
        """
        Args:
            x (torch.Tensor): 입력 이미지 텐서, 형태: [batch_size, 3, 224, 224]
        
        Returns:
            torch.Tensor: 클래스별 로짓 값, 형태: [batch_size, num_classes]
        """
        # ViT 특성 추출
        x = self.vit(x)
        
        # 분류
        x = self.classifier(x)
        return x

    @classmethod
    def from_pretrained(cls, model_path, num_classes=5, device=None):
        """사전 학습된 모델 가중치로부터 모델 인스턴스 생성
        
        Args:
            model_path (str): 모델 가중치 경로 (.pth 또는 .safetensors)
            num_classes (int): 분류할 얼굴형 클래스 수
            device (torch.device): 모델을 로드할 디바이스
            
        Returns:
            FaceShapeViT: 로드된 모델 인스턴스
        """
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
        # 모델 인스턴스 생성
        model = cls(num_classes=num_classes)
        
        # 가중치 로드
        print(f"모델 로드 중: {model_path}")
        
        if model_path.endswith('.safetensors'):
            # safetensors 파일에서 가중치 로드
            state_dict = load_file(model_path)
        else:
            # .pth 파일에서 가중치 로드
            state_dict = torch.load(model_path, map_location=device)
            if 'state_dict' in state_dict:
                state_dict = state_dict['state_dict']
                
        # 키 매핑 (모델 구조에 맞게 조정)
        new_state_dict = {}
        for k, v in state_dict.items():
            # 키 이름 매핑 (모델 구조에 맞게 조정)
            new_k = k.replace('vit.embeddings', 'vit') \
                    .replace('vit.encoder.layer', 'vit.blocks') \
                    .replace('.attention.attention', '.attn') \
                    .replace('intermediate', 'mlp') \
                    .replace('layernorm_before', 'norm1') \
                    .replace('layernorm_after', 'norm2') \
                    .replace('patch_embeddings.projection', 'patch_embed.proj') \
                    .replace('position_embeddings', 'pos_embed')
            new_state_dict[new_k] = v
            
        # 가중치 로드 (missing key는 무시)
        model.load_state_dict(new_state_dict, strict=False)
        model = model.to(device)
        model.eval()
        
        print(f"모델이 {device}로 로드되었습니다.")
        return model


# 표준 클래스 이름 정의 
FACE_SHAPE_CLASSES = [
    '타원형(Oval)',
    '하트형(Heart)',
    '둥근형(Round)',
    '각진형(Square)',
    '긴형(Oblong)'
]


# 사용 예시
if __name__ == "__main__":
    # 간단한 테스트
    model = FaceShapeViT(num_classes=5)
    print(model)
    
    # 임의의 입력 데이터로 테스트
    x = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        output = model(x)
    
    print(f"출력 형태: {output.shape}")  # [1, 5]
