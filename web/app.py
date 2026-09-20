from pathlib import Path

from flask import Flask, jsonify, render_template, request, send_file
from werkzeug.exceptions import HTTPException

from web.service import MAX_UPLOAD_BYTES, SegmentationService, ServiceError


def create_app(segmenter=None, model_path=None, result_dir=None) -> Flask:
    app = Flask(__name__, template_folder="templates", static_folder="static")
    app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_BYTES
    app.config["MODEL_PATH"] = str(model_path) if model_path else None
    root = Path(__file__).resolve().parents[1]
    service = SegmentationService(
        segmenter,
        result_dir or root / "instance" / "results",
    )

    @app.get("/")
    def index():
        return render_template("index.html")

    @app.post("/api/segment")
    def segment():
        if service.segmenter is None:
            raise ServiceError("模型不可用", "MODEL_UNAVAILABLE", 503)
        upload = request.files.get("file")
        if upload is None:
            raise ServiceError("请选择文本文件", "EMPTY_FILE")
        return jsonify(service.segment_upload(upload.filename or "", upload.read()))

    @app.get("/api/results/<job_id>/download")
    def download(job_id: str):
        path, output_name = service.resolve_result(job_id)
        return send_file(
            path,
            as_attachment=True,
            download_name=output_name,
            mimetype="text/plain; charset=utf-8",
        )

    @app.errorhandler(ServiceError)
    def handle_service_error(error: ServiceError):
        return (
            jsonify({"error": {"code": error.code, "message": error.message}}),
            error.status_code,
        )

    @app.errorhandler(HTTPException)
    def handle_http_error(error: HTTPException):
        if not request.path.startswith("/api/"):
            return error

        error_codes = {
            404: "RESULT_NOT_FOUND",
            405: "METHOD_NOT_ALLOWED",
            413: "FILE_TOO_LARGE",
        }
        return (
            jsonify(
                {
                    "error": {
                        "code": error_codes.get(error.code, "HTTP_ERROR"),
                        "message": error.description,
                    }
                }
            ),
            error.code,
        )

    @app.errorhandler(Exception)
    def handle_internal_error(_error):
        return (
            jsonify(
                {
                    "error": {
                        "code": "INTERNAL_ERROR",
                        "message": "服务器处理失败",
                    }
                }
            ),
            500,
        )

    return app
