import json
import os
from datetime import datetime
from typing import List, Dict, Any


class ConversationMemory:
    """Хранит историю разговоров и извлекает паттерны поведения пользователя"""
    
    def __init__(self, memory_file: str):
        self.memory_file = memory_file
        self.conversations: List[Dict[str, Any]] = []
        self.user_patterns: Dict[str, Any] = {
            'common_phrases': [],
            'response_style': {},
            'topics': [],
            'vocabulary': set(),
            'avg_response_length': 0,
            'emoji_usage': 0,
            'question_frequency': 0,
            'command_patterns': []
        }
        self.load_memory()
    
    def load_memory(self):
        """Загружает память из файла"""
        if os.path.exists(self.memory_file):
            try:
                with open(self.memory_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.conversations = data.get('conversations', [])
                    self.user_patterns = data.get('user_patterns', self.user_patterns)
                    if 'vocabulary' in self.user_patterns:
                        self.user_patterns['vocabulary'] = set(self.user_patterns['vocabulary'])
            except (json.JSONDecodeError, IOError):
                self.conversations = []
    
    def save_memory(self):
        """Сохраняет память в файл"""
        os.makedirs(os.path.dirname(self.memory_file), exist_ok=True)
        data = {
            'conversations': self.conversations,
            'user_patterns': {
                **self.user_patterns,
                'vocabulary': list(self.user_patterns['vocabulary'])
            }
        }
        with open(self.memory_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    def add_conversation(self, user_input: str, bot_response: str, context: Dict = None):
        """Добавляет новую запись разговора"""
        conversation_entry = {
            'timestamp': datetime.now().isoformat(),
            'user_input': user_input,
            'bot_response': bot_response,
            'context': context or {}
        }
        self.conversations.append(conversation_entry)
        self._update_user_patterns(user_input)
        self.save_memory()
    
    def _update_user_patterns(self, text: str):
        """Обновляет паттерны поведения на основе текста пользователя"""
        # Обновляем словарный запас
        words = text.lower().split()
        self.user_patterns['vocabulary'].update(words)
        
        # Считаем длину ответа
        lengths = [len(c['user_input']) for c in self.conversations[-10:]]
        if lengths:
            self.user_patterns['avg_response_length'] = sum(lengths) / len(lengths)
        
        # Считаем эмодзи
        emoji_count = sum(1 for c in text if '\U0001F300' <= c <= '\U0001F9FF')
        total_texts = len(self.conversations)
        if emoji_count > 0:
            self.user_patterns['emoji_usage'] = (
                self.user_patterns['emoji_usage'] * (total_texts - 1) + 1
            ) / total_texts
        
        # Считаем вопросы
        if '?' in text:
            self.user_patterns['question_frequency'] = (
                self.user_patterns['question_frequency'] * (total_texts - 1) + 1
            ) / total_texts
    
    def get_training_data(self, min_samples: int = 10) -> List[Dict]:
        """Возвращает данные для обучения модели"""
        if len(self.conversations) < min_samples:
            return []
        
        training_data = []
        for conv in self.conversations[-min_samples:]:
            training_data.append({
                'input': conv['user_input'],
                'expected_style': self._analyze_response_style(conv['user_input']),
                'topics': self._extract_topics(conv['user_input'])
            })
        return training_data
    
    def _analyze_response_style(self, text: str) -> Dict:
        """Анализирует стиль ответа"""
        return {
            'length': len(text),
            'has_question': '?' in text,
            'has_emoji': any('\U0001F300' <= c <= '\U0001F9FF' for c in text),
            'formality': self._detect_formality(text),
            'avg_word_length': sum(len(word) for word in text.split()) / max(len(text.split()), 1)
        }
    
    def _detect_formality(self, text: str) -> str:
        """Определяет формальность текста"""
        informal_words = {'привет', 'пока', 'ок', 'круто', 'класс', 'хай', 'йоу'}
        formal_words = {'здравствуйте', 'до свидания', 'благодарю', 'пожалуйста'}
        
        text_lower = text.lower()
        informal_count = sum(1 for word in informal_words if word in text_lower)
        formal_count = sum(1 for word in formal_words if word in text_lower)
        
        if informal_count > formal_count:
            return 'informal'
        elif formal_count > informal_count:
            return 'formal'
        return 'neutral'
    
    def _extract_topics(self, text: str) -> List[str]:
        """Извлекает темы из текста"""
        topic_keywords = {
            'работа': ['работа', 'код', 'программ', 'разработк', 'проект', 'задач'],
            'технологии': ['технолог', 'ии', 'ai', 'python', 'js', 'веб', 'сервер'],
            'личное': ['семь', 'друг', 'хобби', 'отдых', 'путешеств'],
            'обучение': ['учусь', 'изуча', 'курс', 'книг', 'туториал']
        }
        
        topics = []
        text_lower = text.lower()
        for topic, keywords in topic_keywords.items():
            if any(keyword in text_lower for keyword in keywords):
                topics.append(topic)
        return topics
    
    def get_user_profile(self) -> Dict:
        """Возвращает полный профиль пользователя"""
        return {
            'patterns': self.user_patterns,
            'conversation_count': len(self.conversations),
            'first_interaction': self.conversations[0]['timestamp'] if self.conversations else None,
            'last_interaction': self.conversations[-1]['timestamp'] if self.conversations else None
        }
