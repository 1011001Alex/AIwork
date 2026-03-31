import subprocess
import json
import re
from typing import Optional, Dict, Any


class CommandExecutor:
    """Выполняет системные команды и возвращает результаты"""
    
    @staticmethod
    def execute(command: str, timeout: int = 30) -> Dict[str, Any]:
        """
        Выполняет команду в shell
        
        Args:
            command: Команда для выполнения
            timeout: Таймаут в секундах
        
        Returns:
            Dict с результатами выполнения
        """
        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd='/workspace'
            )
            
            return {
                'success': result.returncode == 0,
                'stdout': result.stdout,
                'stderr': result.stderr,
                'return_code': result.returncode,
                'command': command
            }
        except subprocess.TimeoutExpired:
            return {
                'success': False,
                'stdout': '',
                'stderr': f'Command timed out after {timeout} seconds',
                'return_code': -1,
                'command': command
            }
        except Exception as e:
            return {
                'success': False,
                'stdout': '',
                'stderr': str(e),
                'return_code': -1,
                'command': command
            }
    
    @staticmethod
    def is_safe_command(command: str) -> bool:
        """Проверяет, безопасна ли команда для выполнения"""
        dangerous_patterns = [
            r'\brm\s+-rf\s+/',
            r'\bmkfs',
            r'\bdd\s+',
            r'\bchmod\s+-R\s+777\s+/',
            r'\bchown\s+-R\s+',
            r':\(\)\{\s*:\|\:&\s*\};:',
            r'\bsudo\s+.*\bpas',
        ]
        
        for pattern in dangerous_patterns:
            if re.search(pattern, command, re.IGNORECASE):
                return False
        
        return True
    
    @staticmethod
    def get_system_info() -> Dict[str, Any]:
        """Получает информацию о системе"""
        info = {}
        
        # OS info
        os_result = CommandExecutor.execute('uname -a')
        info['os'] = os_result['stdout'].strip() if os_result['success'] else 'Unknown'
        
        # Python version
        py_result = CommandExecutor.execute('python --version')
        info['python'] = py_result['stdout'].strip() if py_result['success'] else 'Unknown'
        
        # Current directory
        pwd_result = CommandExecutor.execute('pwd')
        info['cwd'] = pwd_result['stdout'].strip() if pwd_result['success'] else 'Unknown'
        
        # Disk usage
        disk_result = CommandExecutor.execute('df -h /')
        info['disk'] = disk_result['stdout'].strip() if disk_result['success'] else 'Unknown'
        
        # Memory info
        mem_result = CommandExecutor.execute('free -h')
        info['memory'] = mem_result['stdout'].strip() if mem_result['success'] else 'Unknown'
        
        return info
