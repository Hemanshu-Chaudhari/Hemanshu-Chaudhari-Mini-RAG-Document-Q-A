import os
from pathlib import Path
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename

import config
from rag_engine import PDFRAGEngine

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = config.UPLOAD_DIR
app.config["MAX_CONTENT_LENGTH"] = 32 * 1024 * 1024  # 32 MB max

# Lazy-load engine instance
engine = None

def get_engine():
    global engine
    if engine is None:
        key = os.environ.get("GEMINI_API_KEY") or config.GEMINI_API_KEY
        if not key:
            raise ValueError(
                "Gemini API Key is missing. Please enter your API key in the sidebar or set GEMINI_API_KEY in .env."
            )
        engine = PDFRAGEngine(api_key=key)
    return engine


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/config", methods=["GET"])
def get_config():
    key = os.environ.get("GEMINI_API_KEY") or config.GEMINI_API_KEY
    return jsonify({"has_api_key": bool(key and key.strip())})


@app.route("/api/set-key", methods=["POST"])
def set_api_key():
    global engine
    data = request.get_json() or {}
    api_key = data.get("api_key", "").strip()
    if not api_key:
        return jsonify({"success": False, "error": "API Key cannot be empty"}), 400

    env_path = config.BASE_DIR / ".env"
    with open(env_path, "w", encoding="utf-8") as f:
        f.write(f"GEMINI_API_KEY={api_key}\n")

    os.environ["GEMINI_API_KEY"] = api_key
    config.GEMINI_API_KEY = api_key

    try:
        engine = PDFRAGEngine(api_key=api_key)
        return jsonify({"success": True, "message": "API key successfully configured!"})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400



@app.route("/api/stats", methods=["GET"])
def get_stats():
    try:
        eng = get_engine()
        stats = eng.vector_store.get_collection_stats()
        return jsonify({"success": True, "stats": stats})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/upload", methods=["POST"])
def upload_file():
    if "file" not in request.files:
        return jsonify({"success": False, "error": "No file part in the request"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"success": False, "error": "No file selected"}), 400

    if not file.filename.lower().endswith(".pdf"):
        return jsonify({"success": False, "error": "Only PDF files are supported"}), 400

    filename = secure_filename(file.filename)
    save_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(save_path)

    try:
        eng = get_engine()
        result = eng.ingest_pdf(save_path)
        stats = eng.vector_store.get_collection_stats()
        return jsonify({
            "success": True,
            "filename": filename,
            "chunks_created": result["chunks_created"],
            "chunks_stored": result["chunks_stored"],
            "stats": stats
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/chat", methods=["POST"])
def chat():
    data = request.get_json() or {}
    message = data.get("message", "").strip()
    use_reranking = data.get("use_reranking", True)
    use_memory = data.get("use_memory", True)

    if not message:
        return jsonify({"success": False, "error": "Message is required"}), 400

    try:
        eng = get_engine()
        response = eng.ask(
            question=message,
            use_reranking=use_reranking,
            use_memory=use_memory
        )
        return jsonify({
            "success": True,
            "answer": response.answer,
            "standalone_query": response.standalone_query,
            "sources": response.sources,
            "retrieved_chunks": response.retrieved_chunks
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/reset-memory", methods=["POST"])
def reset_memory():
    try:
        eng = get_engine()
        eng.reset_conversation()
        return jsonify({"success": True, "message": "Conversational memory reset"})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/clear-collection", methods=["POST"])
def clear_collection():
    try:
        eng = get_engine()
        eng.vector_store.clear_collection()
        eng.reset_conversation()
        return jsonify({"success": True, "message": "Collection cleared"})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


if __name__ == "__main__":
    print(f"Starting PDF RAG Web UI on http://127.0.0.1:5000")
    app.run(host="127.0.0.1", port=5000, debug=True)
