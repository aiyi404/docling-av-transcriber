# docling-av-transcriber

独立的 Docling 兼容音/视频转结构化文档库，复用 Docling 文档模型，同时把底层语音识别切换为阿里云百炼 ASR（Paraformer、SenseVoice 等）。

## 特性
- 支持 WAV/MP3/FLAC 等音频，以及 MP4/AVI/MOV 等视频（先抽取音轨）
- `AliyunBailianAsrClient` 统一封装百炼 ASR 接口，默认使用 `paraformer-v1`
- 输出 `DoclingDocument`，可无缝接入 Docling 生态
- 通过环境变量 `ALIYUN_BAILIAN_API_KEY` 管理密钥
- 结构化目录，易于扩展或替换模型

## 快速开始
```bash
pip install -e .  # 或普通 pip install
export ALIYUN_BAILIAN_API_KEY=sk-xxxx
python examples/basic_usage.py sample.mp4 --language zh
```

## 调试模式

```bash
python examples/debug_usage.py sample.mp4 --language zh --debug
```

调试日志将输出到控制台和 `debug_transcription.log` 文件中，包含详细的处理步骤和错误信息。

## 模块
- `docling_av_transcriber.api`：顶层 API (`transcribe_file`/`transcribe_bytes`)
- `docling_av_transcriber.models`：ASR 客户端抽象与百炼实现
- `docling_av_transcriber.media`：输入验证与音轨抽取
- `docling_av_transcriber.pipelines`：轻量 Pipeline 与 DoclingDocument 构建

## 获取抽取音轨与 Docling 文档
在 RAG / 检索场景中，通常需要同时拿到结构化文本与抽取后的 WAV 音轨以便上传 OSS 或 Supabase。\
自 v0.x 起可以直接调用新的 `*_with_artifacts` API：

```python
from docling_av_transcriber import transcribe_file_with_artifacts

result = transcribe_file_with_artifacts("sample.mp4", language="zh")
doc = result.document          # DoclingDocument，可直接入库或向量化
wav_path = result.audio_path   # 16kHz/单声道 WAV 临时文件

# 将 wav_path 上传到自定义存储后即可手动清理
# wav_path.unlink(missing_ok=True)
```

若是内存字节流，可调用 `transcribe_bytes_with_artifacts(data, filename="input.mp3")`，内部会先写入临时文件并统一转码到 16kHz/单声道 WAV。\
需要注意 `audio_path` 来自 `tempfile.mkstemp`，生命周期由调用方管理：将文件上传或复制后请记得删除，以避免 `/tmp` 目录堆积。

## 测试
```bash
pip install -e .[dev]
pytest -q
```

## 发布
1. `python -m build`
2. `twine check dist/*`
3. `twine upload dist/*`

