# CUDA互換性エラーの修正サマリー

## 修正内容

NVIDIA RTX PRO 6000 Blackwell Max-Q（CUDA capability sm_120）がPyTorchでサポートされていないために発生するCUDAエラーを解決しました。

## 変更されたファイル

### 1. `train_drum.py`
**変更点:**
- デバイス選択の自動フォールバック機能を追加
- CUDAが利用できない、またはエラーが発生する場合、自動的にCPUにフォールバック
- コマンドライン引数 `--device` を追加（設定ファイルを上書き可能）
- モデルのロード時にデバイスを考慮（`map_location='cpu'`）
- より詳細なエラーメッセージとログ出力

**使用例:**
```bash
# CPUで実行
python train_drum.py --config config/drum_config.yaml --device cpu

# CUDAで実行（自動フォールバック付き）
python train_drum.py --config config/drum_config.yaml --device cuda
```

### 2. `config/drum_config.yaml`
**変更点:**
- デフォルトデバイスを `cuda` から `cpu` に変更
- `num_workers` を 8 から 0 に変更（デバッグ時の安定性向上）

**変更前:**
```yaml
training:
  device: 'cuda'
data:
  num_workers: 8
```

**変更後:**
```yaml
training:
  device: 'cpu'
data:
  num_workers: 0
```

### 3. 新規ファイル

#### `CUDA_TROUBLESHOOTING.md`
CUDA互換性問題の詳細なトラブルシューティングガイド：
- 問題の原因説明
- 4つの解決方法（CPU実行、PyTorchアップグレード、ソースビルド、Docker）
- システム情報の確認方法
- その他のCUDAエラーの対処法

#### `quickstart_cpu.sh`
CPUで即座に学習を開始できるスクリプト：
- 設定ファイルとデータセットの自動確認
- 依存関係のチェック
- CPUモードでの学習実行

使用方法:
```bash
bash quickstart_cpu.sh
```

### 4. `README_drum.md`
**追加内容:**
- クイックスタートセクション（CPUでの実行方法）
- CUDAエラーのトラブルシューティングセクション
- `CUDA_TROUBLESHOOTING.md` への参照リンク

## エラーの原因

元のエラー:
```
RuntimeError: CUDA error: no kernel image is available for execution on the device
```

**原因:**
- 使用しているGPU: NVIDIA RTX PRO 6000 Blackwell Max-Q
- CUDA capability: sm_120
- インストールされているPyTorchのサポート範囲: sm_50, sm_60, sm_70, sm_75, sm_80, sm_86, sm_90

Blackwell世代（sm_120）は、インストールされているPyTorchバージョンでサポートされていません。

## 解決方法

### 即座に実行できる方法

**方法1: CPUで実行（推奨）**
```bash
python train_drum.py --config config/drum_config.yaml --device cpu
```

または
```bash
bash quickstart_cpu.sh
```

### 長期的な解決方法

**方法2: PyTorchを最新版にアップグレード**
```bash
# CUDA 12.1の場合
pip install --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# または CUDA 11.8の場合
pip install --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

アップグレード後、CUDAが使用可能か確認:
```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

## その他の警告について

### TransformerのUserWarning
```
enable_nested_tensor is True, but self.use_nested_tensor is False because
encoder_layer.self_attn.batch_first was not True
```

**影響:** パフォーマンスに影響する可能性がありますが、機能的には問題ありません。

**対処:** この警告は無視しても構いません。MuseMorphoseモデル自体の変更が必要になるため、現時点では対応していません。

## 動作確認

修正後、以下のコマンドで動作を確認できます:

```bash
# システムテスト
python test_drum_system.py

# 学習テスト（CPU、1エポックのみ）
python train_drum.py --config config/drum_config.yaml --device cpu
```

## 推奨設定

### 小規模実験・デバッグ（CPU）
```yaml
training:
  device: 'cpu'
  max_epochs: 10
data:
  batch_size: 2
  num_workers: 0
```

### 本格的な学習（GPU、PyTorch更新後）
```yaml
training:
  device: 'cuda'
  max_epochs: 100
data:
  batch_size: 8
  num_workers: 4
```

## 参考リソース

- **詳細なトラブルシューティング**: `CUDA_TROUBLESHOOTING.md`
- **使用方法**: `README_drum.md`
- **PyTorch公式**: https://pytorch.org/get-started/locally/
