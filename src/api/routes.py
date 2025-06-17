"""
Flask routes for the translation API
"""
import os
import time
import requests
from flask import request, jsonify, send_from_directory, send_file
from datetime import datetime
from pathlib import Path

from src.utils.security import SecureFileHandler, rate_limiter, get_client_ip, SecurityError

from src.config import (
    API_ENDPOINT as DEFAULT_OLLAMA_API_ENDPOINT,
    DEFAULT_MODEL,
    MAIN_LINES_PER_CHUNK,
    REQUEST_TIMEOUT,
    OLLAMA_NUM_CTX,
    MAX_TRANSLATION_ATTEMPTS,
    RETRY_DELAY_SECONDS
)


def configure_routes(app, active_translations, output_dir, start_translation_job):
    """Configure Flask routes"""
    
    # Initialize secure file handler
    upload_dir = Path(output_dir) / 'uploads'
    secure_file_handler = SecureFileHandler(upload_dir)
    
    @app.route('/')
    def serve_interface():
        interface_path = 'src/web/templates/translation_interface.html'
        if os.path.exists(interface_path):
            return send_from_directory('src/web/templates', 'translation_interface.html')
        return "<h1>Error: Interface not found</h1>", 404

    @app.route('/api/health', methods=['GET'])
    def health_check():
        return jsonify({
            "status": "ok",
            "message": "Translation API is running",
            "translate_module": "loaded",
            "ollama_default_endpoint": DEFAULT_OLLAMA_API_ENDPOINT,
            "supported_formats": ["txt", "epub", "srt"]
        })

    @app.route('/api/models', methods=['GET'])
    def get_available_models():
        provider = request.args.get('provider', 'ollama')
        
        if provider == 'gemini':
            # Get Gemini models
            api_key = request.args.get('api_key')
            if not api_key:
                import os
                api_key = os.getenv('GEMINI_API_KEY')
                
            if not api_key:
                return jsonify({
                    "models": [],
                    "default": "gemini-2.0-flash",
                    "status": "api_key_missing",
                    "count": 0,
                    "error": "Gemini API key is required. Set GEMINI_API_KEY environment variable or pass api_key parameter."
                })
            
            # Use async function to get models
            import asyncio
            from src.core.llm_providers import GeminiProvider
            
            try:
                gemini_provider = GeminiProvider(api_key=api_key)
                models = asyncio.run(gemini_provider.get_available_models())
                
                if models:
                    model_names = [m['name'] for m in models]
                    return jsonify({
                        "models": models,  # Return full model info
                        "model_names": model_names,  # Just the names for compatibility
                        "default": "gemini-2.0-flash",
                        "status": "gemini_connected",
                        "count": len(models)
                    })
                else:
                    return jsonify({
                        "models": [],
                        "default": "gemini-2.0-flash",
                        "status": "gemini_error",
                        "count": 0,
                        "error": "Failed to retrieve Gemini models"
                    })
                    
            except Exception as e:
                print(f"❌ Error retrieving Gemini models: {e}")
                return jsonify({
                    "models": [],
                    "default": "gemini-2.0-flash",
                    "status": "gemini_error",
                    "count": 0,
                    "error": f"Error connecting to Gemini API: {str(e)}"
                })
        
        else:
            # Original Ollama logic
            ollama_base_from_ui = request.args.get('api_endpoint', DEFAULT_OLLAMA_API_ENDPOINT)
            try:
                base_url = ollama_base_from_ui.split('/api/')[0]
                tags_url = f"{base_url}/api/tags"
                response = requests.get(tags_url, timeout=5)

                if response.status_code == 200:
                    data = response.json()
                    models_data = data.get('models', [])
                    model_names = [m.get('name') for m in models_data if m.get('name')]

                    return jsonify({
                        "models": model_names,
                        "default": DEFAULT_MODEL if DEFAULT_MODEL in model_names else (model_names[0] if model_names else DEFAULT_MODEL),
                        "status": "ollama_connected",
                        "count": len(model_names)
                    })
            except requests.exceptions.RequestException as e:
                print(f"❌ Could not connect to Ollama at {ollama_base_from_ui}: {e}")
            except Exception as e:
                print(f"❌ Error retrieving models from {ollama_base_from_ui}: {e}")

            return jsonify({
                "models": [],
                "default": DEFAULT_MODEL,
                "status": "ollama_offline_or_error",
                "count": 0,
                "error": f"Ollama is not accessible at {ollama_base_from_ui} or an error occurred. Verify that Ollama is running ('ollama serve') and the endpoint is correct."
            })

    @app.route('/api/config', methods=['GET'])
    def get_default_config():
        # Get Gemini API key from environment
        import os
        gemini_api_key = os.getenv('GEMINI_API_KEY', '')
        
        return jsonify({
            "api_endpoint": DEFAULT_OLLAMA_API_ENDPOINT,
            "default_model": DEFAULT_MODEL,
            "chunk_size": MAIN_LINES_PER_CHUNK,
            "timeout": REQUEST_TIMEOUT,
            "context_window": OLLAMA_NUM_CTX,
            "max_attempts": MAX_TRANSLATION_ATTEMPTS,
            "retry_delay": RETRY_DELAY_SECONDS,
            "supported_formats": ["txt", "epub", "srt"],
            "gemini_api_key": gemini_api_key
        })

    @app.route('/api/translate', methods=['POST'])
    def start_translation_request():
        data = request.json
        # Uncomment for debugging
        # print(f"[DEBUG] Received translation request: {data}")

        if 'file_path' in data:
            required_fields = ['file_path', 'source_language', 'target_language', 'model', 'llm_api_endpoint', 'output_filename', 'file_type']
        else:
            required_fields = ['text', 'source_language', 'target_language', 'model', 'llm_api_endpoint', 'output_filename']
        
        for field in required_fields:
            if field not in data or (isinstance(data[field], str) and not data[field].strip()) or (not isinstance(data[field], str) and data[field] is None):
                if field == 'text' and data.get('file_type') == 'txt' and data.get('text') == "":
                    pass
                else:
                    # print(f"[DEBUG] Missing or empty field: {field}")
                    return jsonify({"error": f"Missing or empty field: {field}"}), 400

        translation_id = f"trans_{int(time.time() * 1000)}"

        config = {
            'source_language': data['source_language'],
            'target_language': data['target_language'],
            'model': data['model'],
            'chunk_size': int(data.get('chunk_size', MAIN_LINES_PER_CHUNK)),
            'llm_api_endpoint': data['llm_api_endpoint'],
            'request_timeout': int(data.get('timeout', REQUEST_TIMEOUT)),
            'context_window': int(data.get('context_window', OLLAMA_NUM_CTX)),
            'max_attempts': int(data.get('max_attempts', 2)),
            'retry_delay': int(data.get('retry_delay', 2)),
            'output_filename': data['output_filename'],
            'custom_instructions': data.get('custom_instructions', ''),
            'llm_provider': data.get('llm_provider', 'ollama'),
            'gemini_api_key': data.get('gemini_api_key') or os.getenv('GEMINI_API_KEY', ''),
            'enable_post_processing': data.get('enable_post_processing', False),
            'post_processing_instructions': data.get('post_processing_instructions', '')
        }

        if 'file_path' in data:
            config['file_path'] = data['file_path']
            config['file_type'] = data['file_type']
        else:
            config['text'] = data['text']
            config['file_type'] = data.get('file_type', 'txt')

        active_translations[translation_id] = {
            'status': 'queued',
            'progress': 0,
            'stats': {'start_time': time.time(), 'total_chunks': 0, 'completed_chunks': 0, 'failed_chunks': 0},
            'logs': [f"[{datetime.now().strftime('%H:%M:%S')}] Translation {translation_id} queued."],
            'result': None,
            'config': config,
            'interrupted': False,
            'output_filepath': None
        }

        start_translation_job(translation_id, config)

        return jsonify({
            "translation_id": translation_id,
            "message": "Translation queued.",
            "config_received": config
        })

    @app.route('/api/upload', methods=['POST'])
    def upload_file():
        """Secure file upload with comprehensive validation"""
        
        # Rate limiting
        client_ip = get_client_ip(request)
        if not rate_limiter.is_allowed(client_ip):
            return jsonify({
                "error": "Rate limit exceeded. Please wait before uploading again.",
                "remaining_requests": rate_limiter.get_remaining_requests(client_ip)
            }), 429
        
        # Check if file is present
        if 'file' not in request.files:
            return jsonify({"error": "No file part in request"}), 400
        
        file = request.files['file']
        if not file or file.filename == '':
            return jsonify({"error": "No file selected"}), 400
        
        # Security check: limit filename length in request
        if len(file.filename) > 255:
            return jsonify({"error": "Filename too long"}), 400
        
        try:
            # Read file data
            file_data = file.read()
            
            # Quick size check before validation
            if len(file_data) == 0:
                return jsonify({"error": "Empty file not allowed"}), 400
            
            # Validate and save file securely
            validation_result = secure_file_handler.validate_and_save_file(
                file_data, file.filename
            )
            
            if not validation_result.is_valid:
                return jsonify({
                    "error": validation_result.error_message,
                    "details": "File validation failed"
                }), 400
            
            # Determine file type
            original_filename = file.filename.lower()
            if original_filename.endswith('.epub'):
                file_type = "epub"
            elif original_filename.endswith('.srt'):
                file_type = "srt"
            else:
                file_type = "txt"
            
            # Get file info
            file_size = len(file_data)
            secure_path = validation_result.file_path
            
            # Return success response
            response_data = {
                "success": True,
                "file_path": str(secure_path),
                "filename": file.filename,
                "secure_filename": secure_path.name,
                "file_type": file_type,
                "size": file_size,
                "size_mb": round(file_size / (1024 * 1024), 2)
            }
            
            # Add warnings if any
            if validation_result.warnings:
                response_data["warnings"] = validation_result.warnings
            
            # Log successful upload
            app.logger.info(f"Secure file upload successful: {file.filename} -> {secure_path.name}, Size: {file_size} bytes, IP: {client_ip}")
            
            return jsonify(response_data), 200
            
        except SecurityError as e:
            app.logger.warning(f"Security violation in file upload: {str(e)}, IP: {client_ip}, Filename: {file.filename}")
            return jsonify({
                "error": "Security validation failed",
                "details": str(e)
            }), 403
            
        except Exception as e:
            app.logger.error(f"File upload error: {str(e)}, IP: {client_ip}, Filename: {file.filename}")
            return jsonify({
                "error": "Upload failed due to server error",
                "details": "Please try again or contact support"
            }), 500

    @app.route('/api/translation/<translation_id>', methods=['GET'])
    def get_translation_job_status(translation_id):
        if translation_id not in active_translations:
            return jsonify({"error": "Translation not found"}), 404
        
        job_data = active_translations[translation_id]
        stats = job_data.get('stats', {'start_time': time.time(), 'total_chunks': 0, 'completed_chunks': 0, 'failed_chunks': 0})
        
        if job_data.get('status') == 'running' or job_data.get('status') == 'queued':
            elapsed = time.time() - stats.get('start_time', time.time())
        else:
            elapsed = stats.get('elapsed_time', time.time() - stats.get('start_time', time.time()))

        return jsonify({
            "translation_id": translation_id,
            "status": job_data.get('status'),
            "progress": job_data.get('progress'),
            "stats": {
                'total_chunks': stats.get('total_chunks', 0),
                'completed_chunks': stats.get('completed_chunks', 0),
                'failed_chunks': stats.get('failed_chunks', 0),
                'start_time': stats.get('start_time'),
                'elapsed_time': elapsed
            },
            "logs": job_data.get('logs', [])[-100:],
            "result_preview": "[Preview functionality removed. Download file to view content.]" if job_data.get('status') in ['completed', 'interrupted'] else None,
            "error": job_data.get('error'),
            "config": job_data.get('config'),
            "output_filepath": job_data.get('output_filepath')
        })

    @app.route('/api/translation/<translation_id>/interrupt', methods=['POST'])
    def interrupt_translation_job(translation_id):
        if translation_id not in active_translations:
            return jsonify({"error": "Translation not found"}), 404
        
        job = active_translations[translation_id]
        if job.get('status') == 'running' or job.get('status') == 'queued':
            job['interrupted'] = True
            return jsonify({"message": "Interruption signal sent. Translation will stop after the current segment."}), 200
        return jsonify({"message": "The translation is not in an interruptible state (e.g., already completed or failed)."}), 400

    @app.route('/api/download/<translation_id>', methods=['GET'])
    def download_translated_output_file(translation_id):
        if translation_id not in active_translations:
            return jsonify({"error": "Translation ID not found."}), 404
        
        job_data = active_translations[translation_id]
        server_filepath = job_data.get('output_filepath')

        if not server_filepath:
            config_output_filename = job_data.get('config', {}).get('output_filename')
            if config_output_filename:
                server_filepath = os.path.join(output_dir, config_output_filename)
            else:
                return jsonify({"error": "Output filename configuration missing."}), 404
                
        if not os.path.exists(server_filepath):
            print(f"Download error: File '{server_filepath}' for TID {translation_id} not found on server.")
            return jsonify({"error": f"File '{os.path.basename(server_filepath)}' not found. It might have failed, been interrupted before saving, or been cleaned up."}), 404
        
        try:
            directory = os.path.abspath(os.path.dirname(server_filepath))
            filename = os.path.basename(server_filepath)
            return send_from_directory(directory, filename, as_attachment=True)
        except Exception as e:
            print(f"Error sending file '{server_filepath}' for {translation_id}: {e}")
            return jsonify({"error": f"Server error during download preparation: {str(e)}"}), 500

    @app.route('/api/translations', methods=['GET'])
    def list_all_translations():
        summary_list = []
        for tid, data in active_translations.items():
            summary_list.append({
                "translation_id": tid,
                "status": data.get('status'),
                "progress": data.get('progress'),
                "start_time": data.get('stats', {}).get('start_time'),
                "output_filename": data.get('config', {}).get('output_filename'),
                "file_type": data.get('config', {}).get('file_type', 'txt')
            })
        return jsonify({"translations": sorted(summary_list, key=lambda x: x.get('start_time', 0), reverse=True)})

    @app.route('/api/security/cleanup', methods=['POST'])
    def cleanup_old_files():
        """Clean up old uploaded files (admin endpoint)"""
        try:
            # Get max age from request (default 24 hours)
            max_age_hours = request.json.get('max_age_hours', 24) if request.json else 24
            
            # Validate input
            if not isinstance(max_age_hours, (int, float)) or max_age_hours < 1:
                return jsonify({"error": "Invalid max_age_hours parameter"}), 400
            
            # Perform cleanup
            secure_file_handler.cleanup_old_files(max_age_hours)
            
            return jsonify({
                "success": True,
                "message": f"Cleanup completed for files older than {max_age_hours} hours"
            })
            
        except Exception as e:
            app.logger.error(f"Cleanup error: {str(e)}")
            return jsonify({"error": "Cleanup failed"}), 500

    @app.route('/api/security/info', methods=['GET'])
    def get_security_info():
        """Get security configuration and limits"""
        client_ip = get_client_ip(request)
        
        return jsonify({
            "file_limits": {
                "max_size_mb": SecureFileHandler.MAX_FILE_SIZE // (1024 * 1024),
                "allowed_extensions": list(SecureFileHandler.ALLOWED_EXTENSIONS),
                "allowed_mime_types": list(SecureFileHandler.ALLOWED_MIME_TYPES)
            },
            "rate_limit": {
                "remaining_requests": rate_limiter.get_remaining_requests(client_ip),
                "window_seconds": rate_limiter._window_seconds,
                "max_requests": rate_limiter._max_requests
            },
            "upload_directory": str(secure_file_handler.upload_dir)
        })

    @app.route('/api/files', methods=['GET'])
    def list_all_files():
        """List all files in the translated_files directory with metadata"""
        try:
            files_info = []
            translated_dir = Path(output_dir)
            
            # Get all files in translated_files directory (excluding upload subdirectory)
            for file_path in translated_dir.iterdir():
                if file_path.is_file():
                    stat = file_path.stat()
                    files_info.append({
                        "filename": file_path.name,
                        "file_path": str(file_path),
                        "size_bytes": stat.st_size,
                        "size_mb": round(stat.st_size / (1024 * 1024), 2),
                        "modified_time": stat.st_mtime,
                        "modified_date": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                        "file_type": file_path.suffix.lower()[1:] if file_path.suffix else "unknown"
                    })
            
            # Check uploads directory too
            uploads_dir = translated_dir / 'uploads'
            if uploads_dir.exists():
                for file_path in uploads_dir.iterdir():
                    if file_path.is_file():
                        stat = file_path.stat()
                        files_info.append({
                            "filename": file_path.name,
                            "file_path": str(file_path),
                            "size_bytes": stat.st_size,
                            "size_mb": round(stat.st_size / (1024 * 1024), 2),
                            "modified_time": stat.st_mtime,
                            "modified_date": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                            "file_type": file_path.suffix.lower()[1:] if file_path.suffix else "unknown",
                            "is_upload": True
                        })
            
            # Sort by modified time (newest first)
            files_info.sort(key=lambda x: x['modified_time'], reverse=True)
            
            # Calculate total size
            total_size = sum(f['size_bytes'] for f in files_info)
            
            return jsonify({
                "files": files_info,
                "total_files": len(files_info),
                "total_size_bytes": total_size,
                "total_size_mb": round(total_size / (1024 * 1024), 2)
            })
            
        except Exception as e:
            app.logger.error(f"Error listing files: {str(e)}")
            return jsonify({"error": "Failed to list files", "details": str(e)}), 500

    @app.route('/api/files/<path:filename>', methods=['GET'])
    def download_file_by_name(filename):
        """Download a specific file by name"""
        try:
            # Security check - prevent directory traversal
            if '..' in filename or filename.startswith('/'):
                return jsonify({"error": "Invalid filename"}), 400
            
            # Check in main translated_files directory
            file_path = Path(output_dir) / filename
            if file_path.exists() and file_path.is_file():
                return send_from_directory(output_dir, filename, as_attachment=True)
            
            # Check in uploads subdirectory
            upload_path = Path(output_dir) / 'uploads' / filename
            if upload_path.exists() and upload_path.is_file():
                return send_from_directory(str(Path(output_dir) / 'uploads'), filename, as_attachment=True)
            
            return jsonify({"error": "File not found"}), 404
            
        except Exception as e:
            app.logger.error(f"Error downloading file {filename}: {str(e)}")
            return jsonify({"error": "Download failed", "details": str(e)}), 500

    @app.route('/api/files/<path:filename>', methods=['DELETE'])
    def delete_file(filename):
        """Delete a specific file"""
        try:
            # Security check - prevent directory traversal
            if '..' in filename or filename.startswith('/'):
                return jsonify({"error": "Invalid filename"}), 400
            
            deleted = False
            
            # Try to delete from main directory
            file_path = Path(output_dir) / filename
            if file_path.exists() and file_path.is_file():
                file_path.unlink()
                deleted = True
            else:
                # Try uploads subdirectory
                upload_path = Path(output_dir) / 'uploads' / filename
                if upload_path.exists() and upload_path.is_file():
                    upload_path.unlink()
                    deleted = True
            
            if deleted:
                app.logger.info(f"File deleted: {filename}")
                return jsonify({"success": True, "message": f"File {filename} deleted successfully"})
            else:
                return jsonify({"error": "File not found"}), 404
                
        except Exception as e:
            app.logger.error(f"Error deleting file {filename}: {str(e)}")
            return jsonify({"error": "Delete failed", "details": str(e)}), 500

    @app.route('/api/files/batch/download', methods=['POST'])
    def batch_download_files():
        """Download multiple files as a zip archive"""
        try:
            import zipfile
            import io
            import time
            
            # Get list of filenames from request
            data = request.json
            if not data or 'filenames' not in data:
                return jsonify({"error": "No filenames provided"}), 400
            
            filenames = data['filenames']
            if not isinstance(filenames, list) or len(filenames) == 0:
                return jsonify({"error": "Invalid filenames list"}), 400
            
            # Create in-memory zip file
            zip_buffer = io.BytesIO()
            
            with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                files_added = 0
                
                for filename in filenames:
                    # Security check
                    if '..' in filename or filename.startswith('/'):
                        continue
                    
                    # Try to find file
                    file_path = Path(output_dir) / filename
                    if not file_path.exists():
                        file_path = Path(output_dir) / 'uploads' / filename
                    
                    if file_path.exists() and file_path.is_file():
                        zip_file.write(file_path, filename)
                        files_added += 1
            
            if files_added == 0:
                return jsonify({"error": "No valid files found to download"}), 404
            
            # Prepare zip for download
            zip_buffer.seek(0)
            
            # Generate filename with timestamp
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            zip_filename = f"translated_files_{timestamp}.zip"
            
            return send_file(
                zip_buffer,
                mimetype='application/zip',
                as_attachment=True,
                download_name=zip_filename
            )
            
        except Exception as e:
            app.logger.error(f"Error creating batch download: {str(e)}")
            return jsonify({"error": "Batch download failed", "details": str(e)}), 500

    @app.route('/api/files/batch/delete', methods=['POST'])
    def batch_delete_files():
        """Delete multiple files"""
        try:
            # Get list of filenames from request
            data = request.json
            if not data or 'filenames' not in data:
                return jsonify({"error": "No filenames provided"}), 400
            
            filenames = data['filenames']
            if not isinstance(filenames, list) or len(filenames) == 0:
                return jsonify({"error": "Invalid filenames list"}), 400
            
            deleted_files = []
            failed_files = []
            
            for filename in filenames:
                # Security check
                if '..' in filename or filename.startswith('/'):
                    failed_files.append({"filename": filename, "reason": "Invalid filename"})
                    continue
                
                try:
                    deleted = False
                    
                    # Try main directory
                    file_path = Path(output_dir) / filename
                    if file_path.exists() and file_path.is_file():
                        file_path.unlink()
                        deleted = True
                    else:
                        # Try uploads subdirectory
                        upload_path = Path(output_dir) / 'uploads' / filename
                        if upload_path.exists() and upload_path.is_file():
                            upload_path.unlink()
                            deleted = True
                    
                    if deleted:
                        deleted_files.append(filename)
                    else:
                        failed_files.append({"filename": filename, "reason": "File not found"})
                        
                except Exception as e:
                    failed_files.append({"filename": filename, "reason": str(e)})
            
            return jsonify({
                "success": True,
                "deleted": deleted_files,
                "failed": failed_files,
                "total_deleted": len(deleted_files)
            })
            
        except Exception as e:
            app.logger.error(f"Error in batch delete: {str(e)}")
            return jsonify({"error": "Batch delete failed", "details": str(e)}), 500

    @app.errorhandler(404)
    def route_not_found(error): 
        return jsonify({"error": "API Endpoint not found"}), 404
    
    @app.errorhandler(500)
    def internal_server_error(error):
        import traceback
        tb_str = traceback.format_exc()
        print(f"INTERNAL SERVER ERROR: {error}\nTRACEBACK:\n{tb_str}")
        return jsonify({"error": "Internal server error", "details": str(error)}), 500