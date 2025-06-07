"""
Unified logging system for TranslateBookWithLLM
Provides consistent logging across CLI, Web, and all file types
"""
import sys
import os
import json
from datetime import datetime
from typing import Optional, Dict, Any, Callable
from enum import Enum


class LogLevel(Enum):
    """Log levels with priority values"""
    DEBUG = 10
    INFO = 20
    WARNING = 30
    ERROR = 40
    CRITICAL = 50


class LogType(Enum):
    """Types of log messages for special handling"""
    GENERAL = "general"
    LLM_REQUEST = "llm_request"
    LLM_RESPONSE = "llm_response"
    PROGRESS = "progress"
    CHUNK_INFO = "chunk_info"
    FILE_OPERATION = "file_operation"
    TRANSLATION_START = "translation_start"
    TRANSLATION_END = "translation_end"
    ERROR_DETAIL = "error_detail"


class Colors:
    """ANSI color codes for terminal output - simplified to 3 colors"""
    # Check if colors should be disabled
    NO_COLOR = os.environ.get('NO_COLOR') is not None or not sys.stdout.isatty()
    
    YELLOW = '' if NO_COLOR else '\033[93m'  # Pour les requêtes LLM
    WHITE = '' if NO_COLOR else '\033[97m'   # Pour les réponses LLM et texte principal
    GRAY = '' if NO_COLOR else '\033[90m'    # Pour les infos techniques
    ENDC = '' if NO_COLOR else '\033[0m'     # Reset
    
    @classmethod
    def disable(cls):
        """Disable all colors"""
        cls.YELLOW = cls.WHITE = cls.GRAY = cls.ENDC = ''


class UnifiedLogger:
    """
    Unified logger that provides consistent logging across all interfaces
    """
    
    def __init__(self, 
                 name: str = "TranslateBookWithLLM",
                 console_output: bool = True,
                 enable_colors: bool = True,
                 min_level: LogLevel = LogLevel.INFO,
                 web_callback: Optional[Callable] = None,
                 storage_callback: Optional[Callable] = None):
        """
        Initialize the unified logger
        
        Args:
            name: Logger name/identifier
            console_output: Whether to output to console
            enable_colors: Whether to use colored output
            min_level: Minimum log level to display
            web_callback: Callback for web interface (WebSocket emission)
            storage_callback: Callback for storing logs (e.g., in memory)
        """
        self.name = name
        self.console_output = console_output
        self.enable_colors = enable_colors
        self.min_level = min_level
        self.web_callback = web_callback
        self.storage_callback = storage_callback
        
        # Translation state
        self.translation_state = {
            'current_chunk': 0,
            'total_chunks': 0,
            'source_lang': '',
            'target_lang': '',
            'file_type': '',
            'model': '',
            'start_time': None,
            'in_progress': False
        }
        
        if not enable_colors:
            Colors.disable()
    
    def _format_timestamp(self) -> str:
        """Format current timestamp"""
        return datetime.now().strftime("%H:%M:%S")
    
    def _print_separator(self, char: str = '=', length: int = 80, color: str = Colors.GRAY):
        """Print a colored separator line"""
        if self.console_output:
            print(f"{color}{char * length}{Colors.ENDC}")
    
    def _format_console_message(self, level: LogLevel, message: str, 
                               log_type: LogType = LogType.GENERAL,
                               data: Optional[Dict[str, Any]] = None) -> str:
        """Format message for console output"""
        timestamp = self._format_timestamp()
        
        # Color mapping
        level_colors = {
            LogLevel.DEBUG: Colors.GRAY,
            LogLevel.INFO: Colors.WHITE,
            LogLevel.WARNING: Colors.YELLOW,
            LogLevel.ERROR: Colors.WHITE,
            LogLevel.CRITICAL: Colors.WHITE
        }
        
        color = level_colors.get(level, Colors.WHITE)
        
        # Special formatting for different log types
        if log_type == LogType.LLM_REQUEST:
            return self._format_llm_request(data or {})
        elif log_type == LogType.LLM_RESPONSE:
            return self._format_llm_response(data or {})
        elif log_type == LogType.PROGRESS:
            return self._format_progress(data or {})
        elif log_type == LogType.TRANSLATION_START:
            return self._format_translation_start(message, data or {})
        elif log_type == LogType.TRANSLATION_END:
            return self._format_translation_end(message, data or {})
        elif log_type == LogType.ERROR_DETAIL:
            return self._format_error_detail(message, data or {})
        else:
            # General message format
            level_str = f"[{level.name}]" if level != LogLevel.INFO else ""
            return f"{color}[{timestamp}] {level_str} {message}{Colors.ENDC}"
    
    def _format_llm_request(self, data: Dict[str, Any]) -> str:
        """Format LLM request with full details"""
        output = []
        
        # Une seule ligne de séparation avant le "SENDING TO LLM"
        output.append(f"{Colors.YELLOW}{'=' * 80}{Colors.ENDC}")
        timestamp = self._format_timestamp()
        output.append(f"{Colors.YELLOW}[{timestamp}] SENDING TO LLM{Colors.ENDC}")
        
        # Chunk info
        if self.translation_state['in_progress']:
            current = self.translation_state['current_chunk']
            total = self.translation_state['total_chunks']
            percentage = (current / total * 100) if total > 0 else 0
            output.append(f"{Colors.YELLOW}Chunk: {current}/{total} ({percentage:.1f}% complete){Colors.ENDC}")
        
        # Model info (en gris)
        if 'model' in data:
            output.append(f"{Colors.GRAY}Model: {data['model']}{Colors.ENDC}")
        
        # Full prompt
        output.append(f"\n{Colors.WHITE}RAW PROMPT:{Colors.ENDC}")
        output.append(f"{Colors.WHITE}{data.get('prompt', '')}{Colors.ENDC}")
        
        return '\n'.join(output)
    
    def _format_llm_response(self, data: Dict[str, Any]) -> str:
        """Format LLM response with full details"""
        output = []
        
        timestamp = self._format_timestamp()
        output.append(f"{Colors.WHITE}[{timestamp}] LLM RESPONSE{Colors.ENDC}")
        
        # Execution time (en gris)
        if 'execution_time' in data:
            output.append(f"{Colors.GRAY}Execution time: {data['execution_time']:.2f} seconds{Colors.ENDC}")
        
        # Full response
        output.append(f"\n{Colors.WHITE}RAW RESPONSE (including tags):{Colors.ENDC}")
        output.append(f"{Colors.WHITE}{data.get('response', '')}{Colors.ENDC}")
        
        return '\n'.join(output)
    
    def _format_progress(self, data: Dict[str, Any]) -> str:
        """Format progress summary"""
        output = []
        
        percentage = data.get('percentage', 0)
        current = data.get('current', self.translation_state['current_chunk'])
        total = data.get('total', self.translation_state['total_chunks'])
        
        output.append(f"\n{Colors.WHITE}PROGRESS: {current}/{total} chunks ({percentage:.1f}%){Colors.ENDC}")
        
        # Progress bar simple
        bar_length = 30
        filled = int(bar_length * percentage / 100)
        bar = '█' * filled + '░' * (bar_length - filled)
        output.append(f"{Colors.WHITE}[{bar}] {percentage:.1f}%{Colors.ENDC}")
        
        return '\n'.join(output)
    
    def _format_translation_start(self, message: str, data: Dict[str, Any]) -> str:
        """Format translation start message"""
        output = []
        
        output.append(f"{Colors.YELLOW}TRANSLATION STARTED{Colors.ENDC}")
        
        # Update translation state
        self.translation_state.update({
            'source_lang': data.get('source_lang', 'Unknown'),
            'target_lang': data.get('target_lang', 'Unknown'),
            'file_type': data.get('file_type', 'Unknown'),
            'model': data.get('model', 'Unknown'),
            'total_chunks': data.get('total_chunks', 0),
            'current_chunk': 0,
            'start_time': datetime.now(),
            'in_progress': True
        })
        
        output.append(f"{Colors.WHITE}File Type: {self.translation_state['file_type']}{Colors.ENDC}")
        output.append(f"{Colors.WHITE}Languages: {self.translation_state['source_lang']} → {self.translation_state['target_lang']}{Colors.ENDC}")
        output.append(f"{Colors.GRAY}Model: {self.translation_state['model']}{Colors.ENDC}")
        if self.translation_state['total_chunks'] > 0:
            output.append(f"{Colors.WHITE}Total Chunks: {self.translation_state['total_chunks']}{Colors.ENDC}")
        
        return '\n'.join(output)
    
    def _format_translation_end(self, message: str, data: Dict[str, Any]) -> str:
        """Format translation end message"""
        output = []
        
        output.append(f"\n{Colors.WHITE}TRANSLATION COMPLETE{Colors.ENDC}")
        
        # Calculate duration
        if self.translation_state['start_time']:
            duration = datetime.now() - self.translation_state['start_time']
            output.append(f"{Colors.GRAY}Duration: {duration}{Colors.ENDC}")
        
        if 'output_file' in data:
            output.append(f"{Colors.WHITE}Output saved to: {data['output_file']}{Colors.ENDC}")
        
        # Statistics
        if 'stats' in data:
            stats = data['stats']
            output.append(f"{Colors.WHITE}Completed chunks: {stats.get('completed', 0)}{Colors.ENDC}")
            if stats.get('failed', 0) > 0:
                output.append(f"{Colors.YELLOW}Failed chunks: {stats['failed']}{Colors.ENDC}")
        
        # Reset state
        self.translation_state['in_progress'] = False
        
        return '\n'.join(output)
    
    def _format_error_detail(self, message: str, data: Dict[str, Any]) -> str:
        """Format detailed error message"""
        output = []
        
        timestamp = self._format_timestamp()
        output.append(f"{Colors.WHITE}[{timestamp}] ERROR: {message}{Colors.ENDC}")
        
        if 'details' in data:
            output.append(f"{Colors.GRAY}Details: {data['details']}{Colors.ENDC}")
        if 'chunk' in data:
            output.append(f"{Colors.GRAY}Chunk: {data['chunk']}{Colors.ENDC}")
        
        return '\n'.join(output)
    
    def log(self, level: LogLevel, message: str, 
            log_type: LogType = LogType.GENERAL,
            data: Optional[Dict[str, Any]] = None):
        """
        Main logging method
        
        Args:
            level: Log level
            message: Log message
            log_type: Type of log for special formatting
            data: Additional data for the log entry
        """
        # Check minimum level
        if level.value < self.min_level.value:
            return
        
        # Update chunk counter for LLM requests
        if log_type == LogType.LLM_REQUEST and self.translation_state['in_progress']:
            self.translation_state['current_chunk'] += 1
        
        # Format for console
        if self.console_output:
            try:
                console_msg = self._format_console_message(level, message, log_type, data)
                if console_msg:  # Only print if there's actual content
                    print(console_msg)
            except Exception as e:
                # Fallback to simple message if formatting fails
                timestamp = datetime.now().strftime("%H:%M:%S")
                print(f"[{timestamp}] {message}")
        
        # Create structured log entry
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'level': level.name,
            'type': log_type.value,
            'message': message,
            'data': data or {}
        }
        
        # Web callback (for WebSocket)
        if self.web_callback:
            self.web_callback(log_entry)
        
        # Storage callback (for in-memory storage)
        if self.storage_callback:
            self.storage_callback(log_entry)
    
    # Convenience methods
    def debug(self, message: str, log_type: LogType = LogType.GENERAL, data: Optional[Dict[str, Any]] = None):
        self.log(LogLevel.DEBUG, message, log_type, data)
    
    def info(self, message: str, log_type: LogType = LogType.GENERAL, data: Optional[Dict[str, Any]] = None):
        self.log(LogLevel.INFO, message, log_type, data)
    
    def warning(self, message: str, log_type: LogType = LogType.GENERAL, data: Optional[Dict[str, Any]] = None):
        self.log(LogLevel.WARNING, message, log_type, data)
    
    def error(self, message: str, log_type: LogType = LogType.GENERAL, data: Optional[Dict[str, Any]] = None):
        self.log(LogLevel.ERROR, message, log_type, data)
    
    def critical(self, message: str, log_type: LogType = LogType.GENERAL, data: Optional[Dict[str, Any]] = None):
        self.log(LogLevel.CRITICAL, message, log_type, data)
    
    def update_total_chunks(self, total: int):
        """Update total chunks count"""
        self.translation_state['total_chunks'] = total
    
    def create_legacy_callback(self):
        """
        Create a legacy callback function for backward compatibility
        Returns a function that matches the old log_callback signature
        """
        def legacy_callback(message: str, details: str = "", data: Optional[Dict[str, Any]] = None):
            # Map old message types to new log types and levels
            if data and isinstance(data, dict):
                log_type = data.get('type')
                if log_type == 'llm_request':
                    self.log(LogLevel.INFO, "LLM Request", LogType.LLM_REQUEST, data)
                elif log_type == 'llm_response':
                    self.log(LogLevel.INFO, "LLM Response", LogType.LLM_RESPONSE, data)
                elif log_type == 'progress':
                    self.log(LogLevel.INFO, "Progress Update", LogType.PROGRESS, data)
                else:
                    self.info(details or message, data=data)
            else:
                # Map specific message patterns
                if "error" in message.lower():
                    self.error(details or message)
                elif "warning" in message.lower():
                    self.warning(details or message)
                elif message == "txt_translation_info_chunks1":
                    # Extract chunk count
                    import re
                    match = re.search(r'(\d+)\s+main segments', details)
                    if match:
                        self.update_total_chunks(int(match.group(1)))
                    self.info(details)
                elif message == "txt_translation_loop_start":
                    self.translation_state['in_progress'] = True
                    self.info(details)
                else:
                    self.info(details or message)
        
        return legacy_callback


# Global logger instance
_global_logger = None


def get_logger(name: str = "TranslateBookWithLLM", **kwargs) -> UnifiedLogger:
    """
    Get or create the global logger instance
    
    Args:
        name: Logger name
        **kwargs: Additional arguments for UnifiedLogger
    
    Returns:
        UnifiedLogger instance
    """
    global _global_logger
    if _global_logger is None:
        _global_logger = UnifiedLogger(name, **kwargs)
    return _global_logger


def setup_cli_logger(enable_colors: bool = True) -> UnifiedLogger:
    """Setup logger for CLI usage"""
    return get_logger(
        console_output=True,
        enable_colors=enable_colors,
        min_level=LogLevel.INFO
    )


def setup_web_logger(web_callback: Callable, storage_callback: Callable) -> UnifiedLogger:
    """Setup logger for web interface usage"""
    return get_logger(
        console_output=True,  # Also output to console for debugging
        enable_colors=True,   # Colors work in console even for web
        min_level=LogLevel.INFO,
        web_callback=web_callback,
        storage_callback=storage_callback
    )