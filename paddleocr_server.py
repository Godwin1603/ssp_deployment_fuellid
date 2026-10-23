"""
PaddleOCR Microservice - runs on Python 3.12
Exposes a simple HTTP endpoint so the main app.py (Python 3.14) can use PaddleOCR.

Usage: py -3.12 paddleocr_server.py
Endpoint: POST http://127.0.0.1:5001/ocr  (body: multipart image OR raw bytes)
"""

import base64
import json
import logging
import traceback
from http.server import BaseHTTPRequestHandler, HTTPServer
import sys
import unittest.mock

# Mock modelscope to avoid loading torch DLLs with WinError 127
sys.modules["modelscope"] = unittest.mock.MagicMock()

import cv2
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s [OCR-Server] %(message)s', stream=sys.stdout)

# Custom handler to force flush
class FlushFileHandler(logging.StreamHandler):
    def emit(self, record):
        super().emit(record)
        self.flush()

logging.getLogger().handlers = [FlushFileHandler(sys.stdout)]

logging.info("Loading PaddleOCR model (this may take a moment)...")
from paddleocr import PaddleOCR
import paddle
has_gpu = paddle.device.is_compiled_with_cuda()
logging.info(f"GPU Acceleration Available: {has_gpu}")

# Initialize using exact parameters as original code to preserve logic
ocr = PaddleOCR(
    enable_mkldnn=False,
    lang='en'
)
logging.info("PaddleOCR ready.")


def run_ocr(image):
    """Run PaddleOCR and extract texts/scores."""
    result = ocr.ocr(image)
    results = []
    if result and isinstance(result, list):
        for res in result:
            if not res:
                continue
            # PaddleOCR 3.x dict format:
            if isinstance(res, dict):
                texts = res.get('rec_texts', [])
                scores = res.get('rec_scores', [])
                for t, s in zip(texts, scores):
                    if t:
                        results.append({"rec_text": str(t).strip(), "rec_score": float(s)})
                continue
            for line in res:
                # 2.x format: [[box], ("text", score)]
                if isinstance(line, list) and len(line) == 2 and isinstance(line[1], tuple):
                    results.append({
                        "rec_text": line[1][0],
                        "rec_score": float(line[1][1])
                    })
                elif isinstance(line, dict):
                    texts = line.get('rec_texts', [])
                    scores = line.get('rec_scores', [])
                    for t, s in zip(texts, scores):
                        if t:
                            results.append({"rec_text": str(t).strip(), "rec_score": float(s)})
                        
    return results


class OCRHandler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass  # Suppress request logging noise

    def do_POST(self):
        if self.path != '/ocr':
            self.send_response(404)
            self.end_headers()
            return

        content_length = int(self.headers.get('Content-Length', 0))
        post_data = self.rfile.read(content_length)

        try:
            # Expect JSON with base64-encoded image
            body = json.loads(post_data.decode('utf-8'))
            img_b64 = body.get('image', '')
            img_bytes = base64.b64decode(img_b64)
            nparr = np.frombuffer(img_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if img is None:
                self.send_response(400)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({"error": "Invalid image data"}).encode('utf-8'))
                return

            results = run_ocr(img)

            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({"results": results}).encode('utf-8'))

        except Exception as e:
            logging.error(f"Error processing OCR request: {e}")
            logging.error(traceback.format_exc())
            self.send_response(500)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({"error": str(e)}).encode('utf-8'))


def main():
    server_address = ('127.0.0.1', 5001)
    httpd = HTTPServer(server_address, OCRHandler)
    logging.info(f"Starting PaddleOCR server on http://127.0.0.1:5001")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        logging.info("Shutting down OCR server...")
        httpd.server_close()


if __name__ == '__main__':
    main()
