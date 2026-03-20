# CUDA互換性のトラブルシューティング

## 問題: CUDA capability sm_120がサポートされていない

NVIDIA RTX PRO 6000 Blackwell Max-Q（またはその他の新世代GPU）を使用している場合、以下のエラーが発生する可能性があります：

```
RuntimeError: CUDA error: no kernel image is available for execution on the device
```

これは、インストールされているPyTorchのバージョンが新しいGPUのCUDA capability（sm_120など）をサポートしていないためです。

## 解決方法

### 方法1: CPUで実行する（推奨・最も簡単）

`config/drum_config.yaml`を編集してCPUモードに変更：

```yaml
training:
  device: 'cpu'
```

または、コマンドライン引数で指定：

```bash
python train_drum.py --config config/drum_config.yaml --device cpu
```

**メリット:**
- 追加のインストールが不要
- 安定した動作

**デメリット:**
- GPUより遅い（小規模な実験には十分）

### 方法2: PyTorchを最新版にアップグレードする

新しいGPUをサポートするPyTorchの最新版をインストールします。

#### CUDA 12.x の場合:
```bash
pip install --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

#### CUDA 11.x の場合:
```bash
pip install --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

#### バージョン確認:
```bash
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
```

### 方法3: PyTorchをソースからビルドする（上級者向け）

最新のCUDA capabilityをサポートするために、PyTorchをソースからビルドします。

```bash
git clone --recursive https://github.com/pytorch/pytorch
cd pytorch
# CUDAアーキテクチャを指定
export TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0;12.0"
python setup.py install
```

**注意:** ビルドには1時間以上かかる場合があります。

### 方法4: Docker環境を使用する

CUDA対応のDockerコンテナを使用：

```bash
docker pull pytorch/pytorch:latest
docker run --gpus all -it -v /mnt/kiso-qnap5/obara:/workspace pytorch/pytorch:latest
```

## 現在のシステム情報の確認

```bash
# CUDA バージョン
nvcc --version

# GPU 情報
nvidia-smi

# PyTorch と CUDA の状態
python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    cap = torch.cuda.get_device_capability(0)
    print(f'CUDA capability: sm_{cap[0]}{cap[1]}')
"
```

## その他のCUDAエラー

### メモリ不足エラー
```
RuntimeError: CUDA out of memory
```

**解決方法:**
- バッチサイズを小さくする: `batch_size: 2` または `1`
- 系列長を短くする: `dec_seqlen: 640`, `enc_seqlen: 64`
- モデルを小さくする: `d_model: 256`, `d_ff: 1024`

### CUDAとドライバーのバージョンミスマッチ
```
CUDA driver version is insufficient for CUDA runtime version
```

**解決方法:**
- NVIDIAドライバーを最新版に更新
- または、古いCUDAツールキットに対応するPyTorchをインストール

## 推奨設定

### 小規模実験（CPU）
```yaml
training:
  device: 'cpu'
  batch_size: 2
  max_epochs: 10
```

### 大規模学習（GPU）
```yaml
training:
  device: 'cuda'
  batch_size: 8
  max_epochs: 100
```

PyTorchが新しいGPUをサポートしていることを確認してください。

## 参考リンク

- [PyTorch公式インストールガイド](https://pytorch.org/get-started/locally/)
- [CUDA Toolkit ダウンロード](https://developer.nvidia.com/cuda-downloads)
- [PyTorch CUDA互換性表](https://pytorch.org/get-started/previous-versions/)
