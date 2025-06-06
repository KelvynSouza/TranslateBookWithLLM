"""
Security utilities for file validation and protection
"""
import os
import re
import secrets
import mimetypes
from pathlib import Path
from typing import Set, Optional, Dict, Any
from dataclasses import dataclass


class SecurityError(Exception):
    """Custom exception for security-related errors"""
    pass


@dataclass
class FileValidationResult:
    """Result of file validation"""
    is_valid: bool
    file_path: Optional[Path] = None
    error_message: Optional[str] = None
    warnings: list = None
    
    def __post_init__(self):
        if self.warnings is None:
            self.warnings = []


class SecureFileHandler:
    """Secure file upload and validation handler"""
    
    # Allowed file extensions
    ALLOWED_EXTENSIONS: Set[str] = {'.txt', '.epub'}
    
    # Allowed MIME types
    ALLOWED_MIME_TYPES: Set[str] = {
        'text/plain',
        'application/epub+zip',
        'application/zip',  # Some EPUB files are detected as zip
    }
    
    # Maximum file size (100MB)
    MAX_FILE_SIZE: int = 100 * 1024 * 1024
    
    # Suspicious patterns to scan for in text files
    SUSPICIOUS_PATTERNS: Set[str] = {
        '<script',
        'javascript:',
        'data:',
        'vbscript:',
        'onload=',
        'onerror=',
        'eval(',
        'document.cookie',
        'window.location',
        '<?php',
        '<%',
        'exec(',
        'system(',
        'shell_exec(',
    }
    
    def __init__(self, upload_dir: Path):
        """
        Initialize secure file handler
        
        Args:
            upload_dir: Directory where uploaded files will be stored
        """
        self.upload_dir = Path(upload_dir)
        self.upload_dir.mkdir(parents=True, exist_ok=True)
    
    def validate_and_save_file(self, file_data: bytes, original_filename: str) -> FileValidationResult:
        """
        Validate and securely save an uploaded file
        
        Args:
            file_data: Raw file data
            original_filename: Original filename from upload
            
        Returns:
            FileValidationResult with validation status and secure file path
        """
        try:
            # Step 1: Validate filename and extension
            validation_result = self._validate_filename(original_filename)
            if not validation_result.is_valid:
                return validation_result
            
            # Step 2: Check file size
            if len(file_data) > self.MAX_FILE_SIZE:
                return FileValidationResult(
                    is_valid=False,
                    error_message=f"File too large: {len(file_data)/1024/1024:.1f}MB. Maximum allowed: {self.MAX_FILE_SIZE/1024/1024:.0f}MB"
                )
            
            # Step 3: Create secure filename and path
            secure_filename = self._create_secure_filename(original_filename)
            secure_path = self._get_secure_path(secure_filename)
            
            # Step 4: Save file temporarily for validation
            temp_path = secure_path.with_suffix(secure_path.suffix + '.tmp')
            with open(temp_path, 'wb') as f:
                f.write(file_data)
            
            try:
                # Step 5: Validate file content
                content_validation = self._validate_file_content(temp_path, original_filename)
                if not content_validation.is_valid:
                    temp_path.unlink()  # Remove temp file
                    return content_validation
                
                # Step 6: Move temp file to final location
                temp_path.rename(secure_path)
                
                return FileValidationResult(
                    is_valid=True,
                    file_path=secure_path,
                    warnings=content_validation.warnings
                )
                
            except Exception as e:
                # Clean up temp file on error
                if temp_path.exists():
                    temp_path.unlink()
                raise e
                
        except Exception as e:
            return FileValidationResult(
                is_valid=False,
                error_message=f"Validation failed: {str(e)}"
            )
    
    def _validate_filename(self, filename: str) -> FileValidationResult:
        """Validate filename format and extension"""
        if not filename or not filename.strip():
            return FileValidationResult(is_valid=False, error_message="Filename cannot be empty")
        
        # Remove any path components (security)
        clean_filename = os.path.basename(filename.strip())
        
        if not clean_filename:
            return FileValidationResult(is_valid=False, error_message="Invalid filename")
        
        # Check file extension
        file_ext = Path(clean_filename).suffix.lower()
        
        # Handle case where path traversal removes extension
        if not file_ext and clean_filename:
            # Try to find extension in original filename
            original_ext = Path(filename).suffix.lower()
            if original_ext:
                file_ext = original_ext
        
        if file_ext not in self.ALLOWED_EXTENSIONS:
            return FileValidationResult(
                is_valid=False,
                error_message=f"File type '{file_ext}' not allowed. Allowed types: {', '.join(self.ALLOWED_EXTENSIONS)}"
            )
        
        # Check for suspicious characters
        if re.search(r'[<>:"|?*\x00-\x1f]', clean_filename):
            return FileValidationResult(is_valid=False, error_message="Filename contains invalid characters")
        
        # Check filename length
        if len(clean_filename) > 255:
            return FileValidationResult(is_valid=False, error_message="Filename too long")
        
        return FileValidationResult(is_valid=True)
    
    def _create_secure_filename(self, original_filename: str) -> str:
        """Create a secure filename preventing path traversal and conflicts"""
        # Get clean filename
        clean_name = os.path.basename(original_filename.strip())
        
        # Generate random prefix to prevent conflicts and add security
        random_prefix = secrets.token_hex(8)
        
        # Sanitize filename - keep only safe characters
        safe_chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.-_"
        sanitized = ''.join(c if c in safe_chars else '_' for c in clean_name)
        
        # Ensure it's not too long
        if len(sanitized) > 100:
            name_part = sanitized[:80]
            ext_part = Path(sanitized).suffix[-20:] if Path(sanitized).suffix else ''
            sanitized = name_part + ext_part
        
        return f"{random_prefix}_{sanitized}"
    
    def _get_secure_path(self, filename: str) -> Path:
        """Get secure file path within upload directory"""
        file_path = self.upload_dir / filename
        
        # Resolve path and ensure it's within upload directory
        resolved_path = file_path.resolve()
        upload_dir_resolved = self.upload_dir.resolve()
        
        if not str(resolved_path).startswith(str(upload_dir_resolved)):
            raise SecurityError("Path traversal attempt detected")
        
        return resolved_path
    
    def _validate_file_content(self, file_path: Path, original_filename: str) -> FileValidationResult:
        """Validate file content based on type"""
        warnings = []
        
        # Determine expected file type
        file_ext = Path(original_filename).suffix.lower()
        
        try:
            # Check MIME type
            mime_type, _ = mimetypes.guess_type(str(file_path))
            if mime_type and mime_type not in self.ALLOWED_MIME_TYPES:
                return FileValidationResult(
                    is_valid=False,
                    error_message=f"Invalid file type detected: {mime_type}"
                )
            
            # Validate based on file type
            if file_ext == '.txt':
                return self._validate_text_file(file_path)
            elif file_ext == '.epub':
                return self._validate_epub_file(file_path)
            else:
                return FileValidationResult(
                    is_valid=False,
                    error_message=f"Unsupported file type: {file_ext}"
                )
                
        except Exception as e:
            return FileValidationResult(
                is_valid=False,
                error_message=f"Content validation failed: {str(e)}"
            )
    
    def _validate_text_file(self, file_path: Path) -> FileValidationResult:
        """Validate text file content"""
        warnings = []
        
        try:
            # Read first few KB to scan for suspicious content
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                sample_content = f.read(8192)  # Read first 8KB
            
            # Check for suspicious patterns
            content_lower = sample_content.lower()
            found_patterns = []
            
            for pattern in self.SUSPICIOUS_PATTERNS:
                if pattern in content_lower:
                    found_patterns.append(pattern)
            
            if found_patterns:
                return FileValidationResult(
                    is_valid=False,
                    error_message=f"Suspicious content detected: {', '.join(found_patterns[:3])}"
                )
            
            # Check for excessive special characters (potential obfuscation)
            if len(sample_content) > 0:
                special_char_ratio = sum(1 for c in sample_content if not c.isalnum() and not c.isspace()) / len(sample_content)
                if special_char_ratio > 0.3:
                    warnings.append("High ratio of special characters detected")
            else:
                # Empty file case - reject empty files
                return FileValidationResult(
                    is_valid=False,
                    error_message="Empty file not allowed"
                )
            
            # Check encoding validity
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    f.read()
            except UnicodeDecodeError:
                warnings.append("File encoding may not be UTF-8")
            
            return FileValidationResult(is_valid=True, warnings=warnings)
            
        except Exception as e:
            return FileValidationResult(
                is_valid=False,
                error_message=f"Text file validation failed: {str(e)}"
            )
    
    def _validate_epub_file(self, file_path: Path) -> FileValidationResult:
        """Validate EPUB file structure"""
        warnings = []
        
        try:
            import zipfile
            
            # Check if it's a valid ZIP file
            if not zipfile.is_zipfile(file_path):
                return FileValidationResult(
                    is_valid=False,
                    error_message="EPUB file is not a valid ZIP archive"
                )
            
            # Basic EPUB structure validation
            with zipfile.ZipFile(file_path, 'r') as epub_zip:
                file_list = epub_zip.namelist()
                
                # Check for required EPUB files
                if 'mimetype' not in file_list:
                    warnings.append("Missing mimetype file")
                
                # Check for META-INF directory
                has_meta_inf = any(f.startswith('META-INF/') for f in file_list)
                if not has_meta_inf:
                    warnings.append("Missing META-INF directory")
                
                # Check for potential zip bombs (too many files)
                if len(file_list) > 10000:
                    return FileValidationResult(
                        is_valid=False,
                        error_message="EPUB contains too many files (potential zip bomb)"
                    )
                
                # Check for suspicious file extensions in EPUB
                suspicious_exts = {'.exe', '.bat', '.cmd', '.scr', '.com', '.pif', '.jar'}
                for file_name in file_list:
                    file_ext = Path(file_name).suffix.lower()
                    if file_ext in suspicious_exts:
                        return FileValidationResult(
                            is_valid=False,
                            error_message=f"EPUB contains suspicious file: {file_name}"
                        )
            
            return FileValidationResult(is_valid=True, warnings=warnings)
            
        except Exception as e:
            return FileValidationResult(
                is_valid=False,
                error_message=f"EPUB validation failed: {str(e)}"
            )
    
    def cleanup_old_files(self, max_age_hours: int = 24):
        """Clean up old uploaded files"""
        import time
        
        current_time = time.time()
        max_age_seconds = max_age_hours * 3600
        
        for file_path in self.upload_dir.iterdir():
            if file_path.is_file():
                file_age = current_time - file_path.stat().st_mtime
                if file_age > max_age_seconds:
                    try:
                        file_path.unlink()
                        print(f"Cleaned up old file: {file_path.name}")
                    except Exception as e:
                        print(f"Failed to cleanup {file_path.name}: {e}")


class RateLimiter:
    """Simple in-memory rate limiter"""
    
    def __init__(self):
        self._requests = {}  # IP -> list of timestamps
        self._max_requests = 10  # requests per window
        self._window_seconds = 60  # 1 minute window
    
    def is_allowed(self, client_ip: str) -> bool:
        """Check if request is allowed for this IP"""
        import time
        
        current_time = time.time()
        window_start = current_time - self._window_seconds
        
        # Clean old requests
        if client_ip in self._requests:
            self._requests[client_ip] = [
                timestamp for timestamp in self._requests[client_ip]
                if timestamp > window_start
            ]
        else:
            self._requests[client_ip] = []
        
        # Check if under limit
        if len(self._requests[client_ip]) >= self._max_requests:
            return False
        
        # Add current request
        self._requests[client_ip].append(current_time)
        return True
    
    def get_remaining_requests(self, client_ip: str) -> int:
        """Get remaining requests for this IP"""
        if client_ip not in self._requests:
            return self._max_requests
        return max(0, self._max_requests - len(self._requests[client_ip]))


# Global instances
rate_limiter = RateLimiter()


def get_client_ip(request) -> str:
    """Get client IP address from Flask request"""
    # Check for X-Forwarded-For header (proxy/load balancer)
    if 'X-Forwarded-For' in request.headers:
        return request.headers['X-Forwarded-For'].split(',')[0].strip()
    
    # Check for X-Real-IP header (nginx)
    if 'X-Real-IP' in request.headers:
        return request.headers['X-Real-IP']
    
    # Fallback to remote address
    return request.remote_addr or '127.0.0.1'