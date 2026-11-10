from asr_model import Qwen3ASRModel
import librosa

model = Qwen3ASRModel(
    model_id=r"D:\Program\Github\Qwen3-ASR-onnx\model",
    model_dir=r"D:\Program\Github\Qwen3-ASR-onnx\Qwen3-ASR-0.6B",
)
model.load()
wav, sr = librosa.load(r"D:\Program\Github\Qwen3-ASR-onnx\test_wavs\rag_math.wav")
print(model.transcribe([wav]))
