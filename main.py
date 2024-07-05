import torch
import hydra
from omegaconf import DictConfig
from torch.utils.data import DataLoader
import random
import numpy as np
# from src.models.evflownet import EVFlowNet
from src.models.evflownet import RAFTlikeNet
from src.datasets import DatasetProvider
from enum import Enum, auto
from src.datasets import train_collate
from tqdm import tqdm
from pathlib import Path
from typing import Dict, Any
import os
import time
from transformers import get_cosine_schedule_with_warmup
import torchvision.transforms as transforms


class RepresentationType(Enum):
    VOXEL = auto()
    STEPAN = auto()

def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)

def compute_epe_error(pred_flow: torch.Tensor, gt_flow: torch.Tensor):
    '''
    end-point-error (ground truthと予測値の二乗誤差)を計算
    pred_flow: torch.Tensor, Shape: torch.Size([B, 2, 480, 640]) => 予測したオプティカルフローデータ
    gt_flow: torch.Tensor, Shape: torch.Size([B, 2, 480, 640]) => 正解のオプティカルフローデータ
    '''
    epe = torch.mean(torch.mean(torch.norm(pred_flow - gt_flow, p=2, dim=1), dim=(1, 2)), dim=0)
    return epe

def save_optical_flow_to_npy(flow: torch.Tensor, file_name: str):
    '''
    optical flowをnpyファイルに保存
    flow: torch.Tensor, Shape: torch.Size([2, 480, 640]) => オプティカルフローデータ
    file_name: str => ファイル名
    '''
    np.save(f"{file_name}.npy", flow.cpu().numpy())

# def compute_multiscale_loss(predictions, ground_truth):
#   total_loss = 0.0
#   for prediction in predictions:
#     loss = compute_epe_error(prediction, ground_truth)
#     total_loss += loss
#   return total_loss

@hydra.main(version_base=None, config_path="configs", config_name="base")
def main(args: DictConfig):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    '''
        ディレクトリ構造:

        data
        ├─test
        |  ├─test_city
        |  |    ├─events_left
        |  |    |   ├─events.h5
        |  |    |   └─rectify_map.h5
        |  |    └─forward_timestamps.txt
        └─train
            ├─zurich_city_11_a
            |    ├─events_left
            |    |       ├─ events.h5
            |    |       └─ rectify_map.h5
            |    ├─ flow_forward
            |    |       ├─ 000134.png
            |    |       |.....
            |    └─ forward_timestamps.txt
            ├─zurich_city_11_b
            └─zurich_city_11_c
        '''
    
    # ------------------
    #    Dataloader
    # ------------------
    loader = DatasetProvider(
        dataset_path=Path(args.dataset_path),
        representation_type=RepresentationType.VOXEL,
        delta_t_ms=100,
        num_bins=4
    )
    train_set = loader.get_train_dataset()
    test_set = loader.get_test_dataset()
    collate_fn = train_collate
    # transform = transforms.Compose([transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4,hue=0.5/np.pi),transforms.ToTensor()])
    train_data = DataLoader(train_set,
                                 batch_size=args.data_loader.train.batch_size,
                                 shuffle=args.data_loader.train.shuffle,
                                 collate_fn=collate_fn,
                                 drop_last=False)
    test_data = DataLoader(test_set,
                                 batch_size=args.data_loader.test.batch_size,
                                 shuffle=args.data_loader.test.shuffle,
                                 collate_fn=collate_fn,
                                 drop_last=False)

    '''
    train data:
        Type of batch: Dict
        Key: seq_name, Type: list
        Key: event_volume, Type: torch.Tensor, Shape: torch.Size([Batch, 4, 480, 640]) => イベントデータのバッチ
        Key: flow_gt, Type: torch.Tensor, Shape: torch.Size([Batch, 2, 480, 640]) => オプティカルフローデータのバッチ
        Key: flow_gt_valid_mask, Type: torch.Tensor, Shape: torch.Size([Batch, 1, 480, 640]) => オプティカルフローデータのvalid. ベースラインでは使わない
    
    test data:
        Type of batch: Dict
        Key: seq_name, Type: list
        Key: event_volume, Type: torch.Tensor, Shape: torch.Size([Batch, 4, 480, 640]) => イベントデータのバッチ
    '''
    # ------------------
    #       Model
    # ------------------
    # model = EVFlowNet(args.train).to(device)
    model = RAFTlikeNet(args.train).to(device)
    
    start_epoch = 0
    again = 0
    if again == 1:
        # ロードするモデルのパス 書き換え忘れずに
        model_path = '/content/drive/MyDrive/Colab Notebooks/DLBasics2023_colab/final report/dl_lecture_competition_pub/checkpoints/noskipmodel2_epoch_3_20240704084716.pth'
        
        # 前回の学習状態をロード（オプション）
        # checkpoint = torch.load(model_path)
        checkpoint = torch.load(model_path, map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        # print(checkpoint)
        # 不要なキーを除外する
        state_dict = {k: v for k, v in checkpoint.items() if k in model.state_dict()}

        model.load_state_dict(state_dict, strict=False)
        
        # モデルの重みをロード
        # model.load_state_dict(torch.load(model_path))
        # model.load_state_dict(torch.load(model_path, map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu")))
        
        if 'epoch' in checkpoint:
            start_epoch = checkpoint['epoch']

    # ------------------
    #   optimizer
    # ------------------
    optimizer = torch.optim.Adam(model.parameters(), lr=args.train.initial_learning_rate, weight_decay=args.train.weight_decay)
    if again==1:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    # 総ステップ数を計算
    total_steps = len(train_data) * args.train.epochs

    # ウォームアップのステップ数を設定
    warmup_steps = total_steps // 10
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
    
    if again==1:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    # ------------------
    #   Start training
    # ------------------
    model.train()
    for epoch in range(start_epoch, args.train.epochs):
        total_loss = 0
        print("on epoch: {}".format(epoch+1))
        for i, batch in enumerate(tqdm(train_data)):
            batch: Dict[str, Any]
            # event_image = batch["event_volume"].to(device) # [B, 4, 480, 640]
            event_image_1 = batch["event_volume"].to(device)  # [B, 4, 480, 640]
            event_image_2 = batch["event_volume_next"].to(device)  # [B, 4, 480, 640]
            ground_truth_flow = batch["flow_gt"].to(device) # [B, 2, 480, 640]
            flow = model(event_image_1,event_image_2) # [B, 2, 480, 640]
            loss: torch.Tensor = compute_epe_error(flow, ground_truth_flow)
            # # 各スケールの出力を取得
            # flows = model(event_image)
            # # マルチスケールロスを計算
            # loss = compute_multiscale_loss(flows, ground_truth_flow)
            print(f"batch {i} loss: {loss.item()}")
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
        #     # 定期的にモデルを保存
        # if (epoch + 1) % 1 == 0:
        #     if not os.path.exists('checkpoints'):
        #         os.makedirs('checkpoints')
        
        #     current_time = time.strftime("%Y%m%d%H%M%S")
        #     model_path = f"checkpoints/noskipmodel_epoch_{epoch+1}_{current_time}.pth"
        #     torch.save(model.state_dict(), model_path)
        #     print(f"Model saved to {model_path}")
            
        # # 定期的にモデルを保存
        # if (epoch + 1) % 1 == 0:
        #     if not os.path.exists('checkpoints'):
        #         os.makedirs('checkpoints')

        #     current_time = time.strftime("%Y%m%d%H%M%S")
        #     model_path = f"checkpoints/noskipmodel2_epoch_{epoch + 1}_{current_time}.pth"
        #     checkpoint = {
        #         'epoch': epoch + 1,
        #         'model_state_dict': model.state_dict(),
        #         'optimizer_state_dict': optimizer.state_dict(),
        #         'scheduler_state_dict': scheduler.state_dict(),
        #     }
        #     torch.save(checkpoint, model_path)
        #     print(f"Model saved to {model_path}")

        print(f'Epoch {epoch+1}, Loss: {total_loss / len(train_data)}')

    # Create the directory if it doesn't exist
    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')
    
    current_time = time.strftime("%Y%m%d%H%M%S")
    model_path = f"checkpoints/model_{current_time}.pth"
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

    # ------------------
    #   Start predicting
    # ------------------
    # model_path = f"checkpoints/model_20240704183548.pth"
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    flow: torch.Tensor = torch.tensor([]).to(device)
    with torch.no_grad():
        print("start test")
        for batch in tqdm(test_data):
            batch: Dict[str, Any]
            # event_image = batch["event_volume"].to(device)
            event_image_1 = batch["event_volume"].to(device)  # [B, 4, 480, 640]
            event_image_2 = batch["event_volume_next"].to(device)  # [B, 4, 480, 640]
            batch_flow = model(event_image_1,event_image_2) # [1, 2, 480, 640]
            flow = torch.cat((flow, batch_flow), dim=0)  # [N, 2, 480, 640]
        print("test done")
    # ------------------
    #  save submission
    # ------------------
    file_name = "submission2"
    save_optical_flow_to_npy(flow, file_name)

if __name__ == "__main__":
    main()
