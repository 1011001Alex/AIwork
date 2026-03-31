import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments
from datasets import Dataset
import json
import os
import random
import re
from typing import Optional, Dict, Any, List


class AdaptiveLanguageModel:
    """Модель на базе GPT-2, которая адаптируется под стиль пользователя"""
    
    def __init__(self, model_name: str = "distilgpt2", config_path: str = None):
        self.model_name = model_name
        self.config = self._load_config(config_path)
        
        # Флаг использования легкой модели или fallback режима
        self.use_fallback = self.config.get('fallback_mode', False) or self.config.get('skip_model_loading', False)
        self.use_pattern_matching = self.config.get('use_pattern_matching', True)
        
        # Паттерны ответов для fallback режима
        self.fallback_patterns = self._init_fallback_patterns()
        
        # Пытаемся загрузить модель, но если не получится - используем fallback
        self.model = None
        self.tokenizer = None
        self.is_fine_tuned = False
        self.training_history = []
        
        # Если skip_model_loading включен, сразу используем fallback
        if self.config.get('skip_model_loading', False):
            print("⚡ Режим быстрой загрузки: использую умные паттерны без нейросети")
            self.use_fallback = True
            return
        
        try:
            # Загружаем токенизатор и модель
            print(f"Загрузка модели {model_name}...")
            self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
            self.model = GPT2LMHeadModel.from_pretrained(model_name)
            
            # Устанавливаем специальные токены для контекста
            special_tokens = {
                'additional_special_tokens': ['<USER>', '<BOT>', '<SYSTEM>', '<STYLE>']
            }
            self.tokenizer.add_special_tokens(special_tokens)
            self.model.resize_token_embeddings(len(self.tokenizer))
            print("✅ Модель успешно загружена!")
        except Exception as e:
            print(f"⚠️ Не удалось загрузить модель ({e}), использую fallback режим")
            self.use_fallback = True
    
    def _init_fallback_patterns(self) -> Dict:
        """Инициализирует паттерны для fallback режима"""
        return {
            'greetings': [
                "Привет! Рад тебя слышать!",
                "Здравствуй! Как твои дела?",
                "Приветствую! Что нового?",
                "Хей! Как настроение?"
            ],
            'questions': [
                "Интересный вопрос! Давай подумаем вместе.",
                "Хм, хороший вопрос. Что ты сам об этом думаешь?",
                "Любопытно! А как ты сам на это смотришь?",
                "Отличный вопрос! У меня есть несколько мыслей на этот счет."
            ],
            'statements': [
                "Понимаю, продолжай.",
                "Интересно! Расскажи подробнее.",
                "Звучит убедительно.",
                "Я тебя услышал."
            ],
            'commands': [
                "Выполняю команду...",
                "Сейчас сделаю.",
                "Без проблем.",
                "Уже работаю над этим."
            ],
            'farewells': [
                "Пока! До связи!",
                "Удачи! Заходи ещё!",
                "Всего хорошего!",
                "До встречи!"
            ],
            'unknown': [
                "Хм, не совсем понял, но звучит интересно!",
                "Расскажи подробнее, я хочу понять лучше.",
                "Любопытная мысль! Продолжай.",
                "Я учусь у тебя, так что объясни ещё раз."
            ]
        }
    
    def _load_config(self, config_path: str) -> Dict:
        """Загружает конфигурацию"""
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {
            'max_length': 512,
            'temperature': 0.8,
            'top_p': 0.95,
            'learning_rate': 1e-4,
            'training_epochs': 3,
            'batch_size': 4
        }
    
    def _classify_input(self, user_input: str) -> str:
        """Классифицирует ввод пользователя"""
        text = user_input.lower()
        
        # Приветствия
        greetings = ['привет', 'здравствуй', 'хай', 'hello', 'hi', 'добрый']
        if any(word in text for word in greetings):
            return 'greetings'
        
        # Прощания
        farewells = ['пока', 'до свидания', 'bye', 'увидимся', 'всего']
        if any(word in text for word in farewells):
            return 'farewells'
        
        # Вопросы
        if '?' in text or any(word in text for word in ['как', 'что', 'где', 'когда', 'почему', 'зачем']):
            return 'questions'
        
        # Команды
        if text.startswith('/') or any(word in text for word in ['выполни', 'сделай', 'запусти']):
            return 'commands'
        
        return 'unknown'
    
    def generate_response(self, user_input: str, context: str = "", 
                         user_style: Optional[Dict] = None) -> str:
        """Генерирует ответ с учётом стиля пользователя"""
        
        # Если используем fallback режим или модель не загружена
        if self.use_fallback or self.model is None:
            return self._generate_fallback_response(user_input, user_style)
        
        # Формируем промпт с контекстом и стилем
        prompt = self._build_prompt(user_input, context, user_style)
        
        # Токенизируем вход
        inputs = self.tokenizer.encode(prompt, return_tensors='pt')
        
        # Генерируем ответ
        with torch.no_grad():
            outputs = self.model.generate(
                inputs,
                max_new_tokens=self.config.get('max_length', 256),
                temperature=self.config.get('temperature', 0.7),
                top_p=self.config.get('top_p', 0.9),
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                repetition_penalty=1.2,
                no_repeat_ngram_size=3
            )
        
        # Декодируем ответ
        full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Извлекаем только ответ бота (после последнего <BOT>)
        bot_marker = '<BOT>'
        if bot_marker in full_response:
            response = full_response.split(bot_marker)[-1].strip()
        else:
            response = full_response[len(prompt):].strip()
        
        # Очищаем ответ от возможных артефактов
        response = self._clean_response(response)
        
        return response
    
    def _generate_fallback_response(self, user_input: str, user_style: Optional[Dict] = None) -> str:
        """Генерирует ответ в fallback режиме на основе паттернов"""
        
        # Классифицируем ввод
        category = self._classify_input(user_input)
        
        # Получаем базовый ответ из паттернов
        patterns = self.fallback_patterns.get(category, self.fallback_patterns['unknown'])
        base_response = random.choice(patterns)
        
        # Адаптируем ответ под стиль пользователя если есть данные
        if user_style:
            # Добавляем эмодзи если пользователь их использует
            if user_style.get('emoji_usage', 0) > 0.3:
                emojis = ['😊', '👍', '💡', '🤔', '✨', '🚀']
                base_response += f" {random.choice(emojis)}"
            
            # Делаем ответ короче если пользователь предпочитает краткость
            avg_length = user_style.get('avg_response_length', 100)
            if avg_length < 30 and len(base_response) > 50:
                base_response = base_response[:50] + "..."
            
            # Добавляем вопрос если пользователь часто задаёт вопросы
            if user_style.get('question_frequency', 0) > 0.5 and '?' not in base_response:
                questions = ["А ты?", "Как тебе?", "Что думаешь?", "Интересно же?"]
                base_response += f" {random.choice(questions)}"
        
        # Сохраняем в историю для обучения стилю
        self.training_history.append({
            'input': user_input,
            'response': base_response,
            'category': category
        })
        
        return base_response
    
    def _build_prompt(self, user_input: str, context: str, user_style: Optional[Dict]) -> str:
        """Строит промпт с учётом контекста и стиля"""
        prompt = ""
        
        # Добавляем системный контекст
        prompt += "<SYSTEM>Ты — адаптивный ИИ-ассистент, который учится у пользователя и перенимает его стиль общения.</SYSTEM>\n"
        
        # Добавляем информацию о стиле пользователя
        if user_style:
            style_desc = self._format_style_description(user_style)
            prompt += f"<STYLE>{style_desc}</STYLE>\n"
        
        # Добавляем контекст разговора
        if context:
            prompt += f"<CONTEXT>{context}</CONTEXT>\n"
        
        # Добавляем ввод пользователя
        prompt += f"<USER>{user_input}</USER>\n"
        prompt += "<BOT>"
        
        return prompt
    
    def _format_style_description(self, style: Dict) -> str:
        """Форматирует описание стиля пользователя"""
        desc_parts = []
        
        if style.get('formality') == 'informal':
            desc_parts.append("Общаешься неформально, используешь разговорные выражения")
        elif style.get('formality') == 'formal':
            desc_parts.append("Предпочитаешь формальный стиль общения")
        
        if style.get('has_emoji'):
            desc_parts.append("Используешь эмодзи в общении")
        
        if style.get('has_question'):
            desc_parts.append("Часто задаёшь вопросы")
        
        avg_length = style.get('length', 0)
        if avg_length < 50:
            desc_parts.append("Предпочитаешь краткие ответы")
        elif avg_length > 200:
            desc_parts.append("Любишь развёрнутые ответы")
        
        return "; ".join(desc_parts) if desc_parts else "Универсальный стиль общения"
    
    def _clean_response(self, response: str) -> str:
        """Очищает ответ от артефактов"""
        # Удаляем специальные токены
        for token in ['<USER>', '<BOT>', '<SYSTEM>', '<STYLE>', '<CONTEXT>']:
            response = response.replace(token, '')
        
        # Удаляем обрезанные предложения
        if response and not response[-1] in '.!?…':
            # Если ответ обрывается, пытаемся найти последнее полное предложение
            sentences = response.split('.')
            if len(sentences) > 1:
                response = '.'.join(sentences[:-1]) + '.'
        
        return response.strip()
    
    def fine_tune_on_user_data(self, training_data: list, user_profile: Dict):
        """Дообучает модель на данных пользователя"""
        if not training_data:
            print("Недостаточно данных для обучения")
            return
        
        print(f"Начинаю дообучение на {len(training_data)} примерах...")
        
        # Подготовка данных для обучения
        texts = []
        for sample in training_data:
            # Создаём текст в формате: стиль + ввод пользователя + ожидаемый ответ
            style_desc = self._format_style_description(sample.get('expected_style', {}))
            text = f"<STYLE>{style_desc}</STYLE>\n<USER>{sample['input']}</USER>\n<BOT>"
            texts.append(text)
        
        # Токенизация
        encodings = self.tokenizer(texts, truncation=True, padding=True, 
                                   max_length=self.config.get('max_length', 512))
        
        # Создаём датасет
        dataset = Dataset.from_dict(encodings)
        dataset.set_format(type='torch', columns=['input_ids', 'attention_mask'])
        
        # Настройка обучения
        training_args = TrainingArguments(
            output_dir=self.config.get('model_save_path', './fine_tuned_model'),
            num_train_epochs=self.config.get('training_epochs', 3),
            per_device_train_batch_size=self.config.get('batch_size', 4),
            learning_rate=self.config.get('learning_rate', 1e-4),
            logging_steps=10,
            save_steps=50,
            warmup_steps=10,
            fp16=torch.cuda.is_available(),
            report_to="none"
        )
        
        # Создаём трейнера
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=dataset
        )
        
        # Обучаем
        trainer.train()
        
        # Сохраняем модель
        save_path = self.config.get('model_save_path', './fine_tuned_model')
        trainer.save_model(save_path)
        self.tokenizer.save_pretrained(save_path)
        
        self.is_fine_tuned = True
        self.training_history.append({
            'timestamp': torch.__version__,
            'samples_count': len(training_data),
            'user_profile_summary': {
                'conversations': user_profile.get('conversation_count', 0),
                'style': user_profile.get('patterns', {}).get('response_style', {})
            }
        })
        
        print(f"Модель успешно дообучена и сохранена в {save_path}")
    
    def load_fine_tuned_model(self, model_path: str):
        """Загружает дообученную модель"""
        if os.path.exists(model_path):
            print(f"Загрузка дообученной модели из {model_path}...")
            self.model = GPT2LMHeadModel.from_pretrained(model_path)
            self.tokenizer = GPT2Tokenizer.from_pretrained(model_path)
            self.is_fine_tuned = True
            print("Модель загружена успешно!")
        else:
            print(f"Модель по пути {model_path} не найдена")
    
    def get_model_stats(self) -> Dict:
        """Возвращает статистику модели"""
        return {
            'model_name': self.model_name,
            'is_fine_tuned': self.is_fine_tuned,
            'training_sessions': len(self.training_history),
            'parameters_count': sum(p.numel() for p in self.model.parameters()) if self.model else 0,
            'device': str(self.model.device) if self.model else 'fallback_mode',
            'fallback_mode': self.use_fallback,
            'total_interactions': len(self.training_history)
        }
