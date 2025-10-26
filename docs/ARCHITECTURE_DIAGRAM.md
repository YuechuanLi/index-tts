# IndexTTS2 Architecture Diagram

## High-Level System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         INDEXTTS2 INFERENCE SYSTEM                          │
└─────────────────────────────────────────────────────────────────────────────┘

                              USER INTERFACES
          ┌──────────────────┬──────────────────┬──────────────────┐
          │                  │                  │                  │
        WebUI              CLI                 Python API        Direct Class
      (webui.py)      (indextts.cli)     (IndexTTS2 class)    (UnifiedVoice)
          │                  │                  │                  │
          └──────────────────┴──────────────────┴──────────────────┘
                                    │
                                    ▼
                        ┌───────────────────────┐
                        │  IndexTTS2 Instance   │
                        │ (infer_v2.py)         │
                        └───────────────────────┘
                                    │
                    ┌───────────────┼───────────────┐
                    │               │               │
                    ▼               ▼               ▼
            Input Audio        Input Text      Emotion Control
         (spk_prompt)          (Synthesis)      (Optional)
            .wav file           string    ┌──────────────────┐
                                          │ emo_audio_prompt │
                                          │ emo_vector       │
                                          │ emo_text         │
                                          │ use_emo_text     │
                                          │ emo_alpha        │
                                          └──────────────────┘

```

## Detailed Processing Pipeline

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                      STEP 1: AUDIO FEATURE EXTRACTION                        │
└──────────────────────────────────────────────────────────────────────────────┘

Speaker Audio Input
        │
        ▼
┌─────────────────────────────────────────┐
│ SeamlessM4TFeatureExtractor             │  Extract audio features
│ (facebook/w2v-bert-2.0 preprocessor)    │
└─────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────┐
│ Wav2Vec2BertModel                       │  Get semantic embeddings
│ (facebook/w2v-bert-2.0)                 │  Output: 1024-dim vectors
│ • Input: audio features                 │
│ • Output: semantic embeddings           │
└─────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────┐
│ Normalization (Semantic Statistics)     │  Normalize to mean=0, std=1
│ • semantic_mean: learnable              │
│ • semantic_std: learnable               │
└─────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────┐
│ RepCodec (Semantic Codec)               │  Quantize to discrete codes
│ • Codebook size: 8192                   │
│ • Output: semantic tokens               │
└─────────────────────────────────────────┘
        │
        ├─────────────────────────────────────┐
        │                                     │
        ▼                                     ▼
  Semantic Codes                      CAMPPlus Speaker Embedding
  (8192 vocabulary)                   (192-dimensional)


┌──────────────────────────────────────────────────────────────────────────────┐
│                      STEP 2: TEXT PROCESSING                                 │
└──────────────────────────────────────────────────────────────────────────────┘

Input Text
    │
    ▼
┌─────────────────────────────────────────┐
│ TextNormalizer                          │  Normalize punctuation & chars
│ • Chinese: WeTextProcessing (Linux)     │  Preserve pinyin tones
│ • English: Standard normalization       │
│ • Mixed: Language detection             │
└─────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────┐
│ TextTokenizer                           │  BPE tokenization
│ • SentencePiece model (bpe.model)       │  Output: token sequence
│ • Vocab size: 12,000                    │
└─────────────────────────────────────────┘
    │
    ▼
  Text Tokens (0-12000)


┌──────────────────────────────────────────────────────────────────────────────┐
│                      STEP 3: EMOTION PROCESSING                              │
└──────────────────────────────────────────────────────────────────────────────┘

Emotion Input (3 options)
    │
    ├─ Option 1: emo_audio_prompt
    │       │
    │       ▼
    │   ┌─────────────────────────┐
    │   │ Feature extraction      │  Extract emotion features
    │   └─────────────────────────┘
    │       │
    │       ▼
    │   ┌─────────────────────────┐
    │   │ Emotion Encoder         │
    │   │ (ConformerEncoder)      │  1024-dim → 512-dim
    │   └─────────────────────────┘
    │       │
    │       ▼
    │   ┌─────────────────────────┐
    │   │ Emotion Perceiver       │  Resample to 1-dim latent
    │   │ (PerceiverResampler)    │  Output: emotion vector
    │   └─────────────────────────┘
    │
    ├─ Option 2: emo_vector
    │       │
    │       ▼
    │   Direct 8-dim vector
    │   [happy, angry, sad, afraid, disgusted, melancholic, surprised, calm]
    │
    └─ Option 3: emo_text
            │
            ▼
        ┌─────────────────────────┐
        │ QwenEmotion             │  Text-based emotion encoder
        │ (Qwen 0.6B LLM)         │  Output: 8-dim emotion vector
        │ • Input: emotion text   │
        │ • Output: emotion codes │
        └─────────────────────────┘

All paths converge to: Emotion Vector (1024-dim or 8-dim)


┌──────────────────────────────────────────────────────────────────────────────┐
│                      STEP 4: GPT GENERATION                                  │
└──────────────────────────────────────────────────────────────────────────────┘

Semantic Tokens + Speaker Embedding + Emotion Vector
    │
    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    UNIFIED VOICE (GPT Model)                        │
│                      indextts/gpt/model_v2.py                       │
│                         UnifiedVoice Class                          │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │ ConformerEncoder (Speaker Conditioning)                     │  │
│  │ • Input: Speaker embeddings (1024-dim)                      │  │
│  │ • Output: 512-dim speaker condition                         │  │
│  └──────────────────────────────────────────────────────────────┘  │
│          ▼                                                          │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │ PerceiverResampler (Conditioning Resampling)                │  │
│  │ • Input: 512-dim condition                                  │  │
│  │ • Latents: 32                                               │  │
│  │ • Output: Resampled condition (32 latents)                  │  │
│  └──────────────────────────────────────────────────────────────┘  │
│          ▼                                                          │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │ Emotion Processing                                          │  │
│  │ • ConformerEncoder for emotion conditioning                 │  │
│  │ • PerceiverResampler for emotion (1 latent)                 │  │
│  └──────────────────────────────────────────────────────────────┘  │
│          ▼                                                          │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │ GPT2 Transformer Stack (24 layers)                          │  │
│  │ • Hidden dimension: 1280                                    │  │
│  │ • Attention heads: 20                                       │  │
│  │ • Context: 1815 mel tokens + 600 text tokens               │  │
│  │ • Inputs: Speaker + Emotion + Text embeddings              │  │
│  │ • Output: Token logits                                      │  │
│  └──────────────────────────────────────────────────────────────┘  │
│          ▼                                                          │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │ Auto-Regressive Generation                                  │  │
│  │ • Generate semantic tokens step-by-step                     │  │
│  │ • Output range: 0-8191 (semantic codes)                     │  │
│  │ • Stop condition: [STOP] token or max length               │  │
│  │ • Sampling: Typical sampling or greedy                      │  │
│  └──────────────────────────────────────────────────────────────┘  │
│          ▼                                                          │
│    Semantic Token Sequence                                        │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘


┌──────────────────────────────────────────────────────────────────────────────┐
│                      STEP 5: SPEECH-TO-MEL SYNTHESIS                         │
└──────────────────────────────────────────────────────────────────────────────┘

Semantic Tokens
    │
    ▼
┌──────────────────────────────────────────────────────────────────┐
│                     MyModel (Speech-to-Mel)                      │
│              indextts/s2mel/modules/commons.py                   │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │ InterpolateRegulator (Duration Control)                   │ │
│  │ • Input: Semantic codes                                   │ │
│  │ • Process: Expand/compress to target duration             │ │
│  │ • Output: Duration-adjusted representations              │ │
│  └────────────────────────────────────────────────────────────┘ │
│          ▼                                                       │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │ CFM (Continuous Flow Matching)                            │ │
│  │ • DiT (Diffusion Transformer) backbone                    │ │
│  │ • Hidden: 512, Heads: 8, Depth: 13                        │ │
│  │ • Input: Duration-adjusted codes + style condition       │ │
│  │ • Output: Mel-spectrogram                                 │ │
│  │ • Target: 80 mel-bins, sample rate 22050 Hz               │ │
│  └────────────────────────────────────────────────────────────┘ │
│          ▼                                                       │
│    Mel-Spectrogram (80 mel-bins, variable length)             │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘


┌──────────────────────────────────────────────────────────────────────────────┐
│                      STEP 6: NEURAL VOCODING                                 │
└──────────────────────────────────────────────────────────────────────────────┘

Mel-Spectrogram
    │
    ▼
┌──────────────────────────────────────────────────────────────────┐
│                       BigVGAN Vocoder                            │
│         indextts/BigVGAN/bigvgan.py or                          │
│    indextts/s2mel/modules/bigvgan/bigvgan.py                    │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Input: Mel-Spectrogram (80 mel-bins)                          │
│      ▼                                                           │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │ Spectral Normalization Convolutions                       │ │
│  │ • Residual blocks with alias-free activations             │ │
│  │ • Anti-aliasing filters                                   │ │
│  └────────────────────────────────────────────────────────────┘ │
│      ▼                                                           │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │ Multi-Scale Discriminator (during training)               │ │
│  │ • For waveform quality assurance                           │ │
│  └────────────────────────────────────────────────────────────┘ │
│      ▼                                                           │
│   Waveform Output (24000 Hz, mono)                            │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘


┌──────────────────────────────────────────────────────────────────────────────┐
│                      STEP 7: OUTPUT SAVING                                   │
└──────────────────────────────────────────────────────────────────────────────┘

Waveform
    │
    ▼
┌──────────────────────────────────────────┐
│ Save to WAV File                         │
│ • Output path: user-specified            │
│ • Format: PCM, 24000 Hz, mono            │
│ • Quality: 16-bit or configurable        │
└──────────────────────────────────────────┘
    │
    ▼
Output Audio File (.wav)
```

## Module Dependency Graph

```
┌────────────────────────────────────────────────────────────────────────────┐
│                        IndexTTS2 (Main Interface)                          │
│                        indextts/infer_v2.py                                │
└────────────────────────────────────────────────────────────────────────────┘
         │
         ├─────────────┬──────────────┬──────────────┬──────────────┐
         │             │              │              │              │
         ▼             ▼              ▼              ▼              ▼
    UnifiedVoice   MyModel        TextNormalizer  QwenEmotion   Feature
    (gpt/)         (s2mel/)       (utils/)        (external)     Extractors
         │             │              │                           │
         ├─────┐       ├─────┐        │                          │
         │     │       │     │        │                          │
         ▼     ▼       ▼     ▼        ▼                          ▼
    ConformerEncoder  CFM    Length    TextTokenizer      SeamlessM4T
    PerceiverResampler      Regulator  (SentencePiece)    Wav2Vec2Bert
    GPT2InferenceModel      BigVGAN    WeTextProcessing   CAMPPlus
    GPT2 Transformer        CAMPPlus   jieba              (Speaker Embed)
                            DiffusionTransformer


┌────────────────────────────────────────────────────────────────────────────┐
│                        External Dependencies                               │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                            │
│  PyTorch Ecosystem:                                                       │
│  • torch.nn (Core neural network components)                             │
│  • torch.optim (Optimizers)                                              │
│  • torchaudio (Audio processing)                                         │
│                                                                            │
│  HuggingFace Ecosystem:                                                  │
│  • transformers.GPT2Config, GPT2PreTrainedModel                          │
│  • transformers.Wav2Vec2BertModel                                        │
│  • transformers.SeamlessM4TFeatureExtractor                              │
│  • modelscope.AutoModelForCausalLM (QwenEmotion)                        │
│                                                                            │
│  Audio Processing:                                                       │
│  • librosa (Audio manipulation)                                          │
│  • descript-audiotools (Audio codec)                                     │
│  • safetensors (Model serialization)                                     │
│                                                                            │
│  Configuration:                                                          │
│  • omegaconf (Config management)                                         │
│  • json5 (Config parsing)                                                │
│                                                                            │
│  Utilities:                                                              │
│  • sentencepiece (Tokenization)                                          │
│  • jieba (Chinese word segmentation)                                     │
│  • munch (Config dict wrapper)                                           │
│  • accelerate (Distributed training/inference)                           │
│  • deepspeed (Optional optimization)                                     │
│                                                                            │
└────────────────────────────────────────────────────────────────────────────┘
```

## Configuration Flow

```
checkpoints/config.yaml
         │
         ├─ dataset config ─────────────────► Audio preprocessing
         │
         ├─ gpt config ──────────────────────► UnifiedVoice initialization
         │  ├─ model_dim: 1280
         │  ├─ layers: 24
         │  ├─ heads: 20
         │  └─ condition_type: "conformer_perceiver"
         │
         ├─ semantic_codec config ─────────► RepCodec initialization
         │  └─ codebook_size: 8192
         │
         ├─ s2mel config ────────────────────► MyModel + CFM initialization
         │  └─ length_regulator config
         │
         ├─ vocoder config ──────────────────► BigVGAN initialization
         │
         └─ checkpoint paths ───────────────► Model loading
            ├─ gpt.pth (UnifiedVoice weights)
            ├─ s2mel.pth (MyModel weights)
            ├─ bpe.model (TextTokenizer)
            ├─ wav2vec2bert_stats.pt (Feature normalization)
            └─ qwen0.6bemo4-merge/ (QwenEmotion weights)
```

## Data Type Flow

```
Input Tensors:
├─ Speaker audio: (batch, samples) → float32
├─ Text tokens: (batch, seq_len) → int64
├─ Emotion vector: (batch, 8) → float32
└─ Speaker embeddings: (batch, 192) → float32

Processing:
├─ Semantic codes: (batch, seq_len) → int64 [0-8191]
├─ Text embeddings: (batch, seq_len, 1280) → float32
├─ Speaker condition: (batch, 32, 1280) → float32
├─ Emotion condition: (batch, 1, 1280) → float32
├─ Mel-spectrogram: (batch, 80, time) → float32
└─ GPT hidden: (batch, seq_len, 1280) → float32

Output:
└─ Audio waveform: (samples,) → float32 [normalized to [-1, 1]]
```

## Hardware Acceleration Support

```
Device Detection & Configuration:
│
├─ CUDA (NVIDIA GPUs)
│  └─ use_fp16 available
│     use_cuda_kernel available (BigVGAN optimization)
│     deepspeed available
│
├─ XPU (Intel Arc)
│  └─ use_fp16 available
│     deepspeed not available (CUDA only)
│
├─ MPS (Apple Silicon)
│  └─ use_fp16 not recommended (performance issue)
│     deepspeed not available
│
└─ CPU (Fallback)
   └─ use_fp16 disabled
      deepspeed disabled
      use_cuda_kernel disabled
```

This architecture diagram provides a comprehensive visual understanding of the IndexTTS2 system's structure, data flow, and component interactions.
