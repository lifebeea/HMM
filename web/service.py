import json
import time
import uuid
from pathlib import Path

from hmm_segmenter.preprocess import clean_text, split_sentences


MAX_UPLOAD_BYTES = 5 * 1024 * 1024
RESULT_TTL_SECONDS = 3600
MAX_PREVIEW_SENTENCES = 20
MAX_PREVIEW_CHARS = 2000


class ServiceError(Exception):
    def __init__(self, message: str, code: str, status_code: int = 400):
        super().__init__(message)
        self.message = message
        self.code = code
        self.status_code = status_code


class SegmentationService:
    def __init__(
        self,
        segmenter,
        result_dir: str | Path,
        max_upload_bytes: int = MAX_UPLOAD_BYTES,
        ttl_seconds: int = RESULT_TTL_SECONDS,
    ):
        self.segmenter = segmenter
        self.result_dir = Path(result_dir)
        self.max_upload_bytes = max_upload_bytes
        self.ttl_seconds = ttl_seconds
        self.result_dir.mkdir(parents=True, exist_ok=True)
        self.cleanup_expired()

    def cleanup_expired(self, now: float | None = None) -> int:
        threshold = (time.time() if now is None else now) - self.ttl_seconds
        removed = 0
        for path in self.result_dir.glob("*.txt"):
            if path.stat().st_mtime >= threshold:
                continue
            path.unlink(missing_ok=True)
            path.with_suffix(".json").unlink(missing_ok=True)
            removed += 1
        return removed

    def segment_upload(self, filename: str, data: bytes) -> dict:
        if not filename:
            raise ServiceError("请选择文本文件", "EMPTY_FILE")
        if Path(filename).suffix.lower() != ".txt":
            raise ServiceError("仅支持 .txt 文本文件", "INVALID_FILE_TYPE")
        if not data:
            raise ServiceError("文件内容为空", "EMPTY_FILE")
        if len(data) > self.max_upload_bytes:
            raise ServiceError("文件超过 5 MiB 限制", "FILE_TOO_LARGE", 413)
        try:
            text = read_text_file_bytes(data)
        except UnicodeError as exc:
            raise ServiceError(
                "无法识别文件编码，请使用 UTF-8、GBK 或 GB18030",
                "DECODE_ERROR",
            ) from exc

        sentences = split_sentences(clean_text(text))
        if not sentences:
            raise ServiceError("清洗后没有可分词文本", "EMPTY_TEXT")

        lines: list[str] = []
        word_count = 0
        for sentence in sentences:
            words = self.segmenter.cut(sentence)
            word_count += len(words)
            lines.append(" ".join(words))

        self.cleanup_expired()
        job_id = str(uuid.uuid4())
        safe_filename = filename.replace("\\", "/").rsplit("/", 1)[-1]
        safe_stem = Path(safe_filename).stem or "result"
        output_name = f"{safe_stem}_seg.txt"
        result_path = self.result_dir / f"{job_id}.txt"
        metadata_path = self.result_dir / f"{job_id}.json"
        result_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        metadata_path.write_text(
            json.dumps({"output_name": output_name, "created_at": time.time()}),
            encoding="utf-8",
        )
        return {
            "job_id": job_id,
            "input_name": filename,
            "output_name": output_name,
            "sentence_count": len(lines),
            "word_count": word_count,
            "preview": self._preview(lines),
            "download_url": f"/api/results/{job_id}/download",
        }

    @staticmethod
    def _preview(lines: list[str]) -> list[str]:
        preview: list[str] = []
        char_count = 0
        for line in lines[:MAX_PREVIEW_SENTENCES]:
            added = len(line) + (1 if preview else 0)
            if char_count + added > MAX_PREVIEW_CHARS:
                break
            preview.append(line)
            char_count += added
        return preview

    def resolve_result(self, job_id: str) -> tuple[Path, str]:
        try:
            canonical = str(uuid.UUID(job_id))
        except (ValueError, AttributeError) as exc:
            raise ServiceError("结果不存在", "RESULT_NOT_FOUND", 404) from exc
        result_path = self.result_dir / f"{canonical}.txt"
        metadata_path = self.result_dir / f"{canonical}.json"
        if not result_path.is_file() or not metadata_path.is_file():
            raise ServiceError("结果已过期或不存在", "RESULT_EXPIRED", 404)
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            output_name = metadata["output_name"]
        except (OSError, UnicodeError, json.JSONDecodeError, KeyError, TypeError) as exc:
            raise ServiceError("结果元数据不可用", "RESULT_EXPIRED", 404) from exc
        return result_path, output_name


def read_text_file_bytes(data: bytes) -> str:
    for encoding in ("utf-8", "gbk", "gb18030"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    raise UnicodeError("unsupported encoding")
