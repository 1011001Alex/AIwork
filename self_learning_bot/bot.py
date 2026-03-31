import json
import os
from datetime import datetime
from typing import List, Dict, Optional

from memory.memory_manager import ConversationMemory
from models.adaptive_model import AdaptiveLanguageModel
from utils.command_executor import CommandExecutor


class SelfLearningBot:
    """
    Самообучающийся консольный бот, который адаптируется под стиль пользователя
    и может выполнять любые запросы через CLI
    """
    
    def __init__(self, config_path: str = 'config/config.json'):
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        os.chdir(self.base_dir)
        
        # Загружаем конфигурацию
        self.config = self._load_config(config_path)
        
        # Инициализируем память
        memory_file = os.path.join(self.base_dir, self.config['memory_file'])
        self.memory = ConversationMemory(memory_file)
        
        # Инициализируем модель
        config_file = os.path.join(self.base_dir, config_path)
        self.model = AdaptiveLanguageModel(
            model_name=self.config['model_name'],
            config_path=config_file
        )
        
        # Пытаемся загрузить дообученную модель если она существует
        model_path = os.path.join(self.base_dir, self.config['model_save_path'])
        if os.path.exists(model_path):
            self.model.load_fine_tuned_model(model_path)
        
        # Контекст разговора
        self.conversation_context: List[Dict] = []
        self.max_context_length = 10
        
        # Статистика
        self.stats = {
            'interactions': 0,
            'commands_executed': 0,
            'training_sessions': 0,
            'started_at': datetime.now().isoformat()
        }
        
        print("\n" + "="*60)
        print("🤖 САМООБУЧАЮЩИЙСЯ БОТ ЗАПУЩЕН")
        print("="*60)
        print(f"📊 Разговоров в памяти: {len(self.memory.conversations)}")
        print(f"🧠 Модель: {self.config['model_name']}")
        print(f"📚 Дообучение начнётся после {self.config['min_samples_for_training']} диалогов")
        print("="*60)
        print("\n💡 Команды:")
        print("  /stats - показать статистику")
        print("  /profile - показать мой профиль")
        print("  /train - принудительное дообучение")
        print("  /clear - очистить контекст")
        print("  /exec <command> - выполнить команду")
        print("  /help - помощь")
        print("  /quit - выйти")
        print("="*60 + "\n")
    
    def _load_config(self, config_path: str) -> Dict:
        """Загружает конфигурацию"""
        full_path = os.path.join(self.base_dir, config_path)
        if os.path.exists(full_path):
            with open(full_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {
            'model_name': 'gpt2',
            'max_length': 512,
            'temperature': 0.8,
            'top_p': 0.95,
            'learning_rate': 1e-4,
            'training_epochs': 3,
            'batch_size': 4,
            'memory_file': 'memory/conversation_history.json',
            'model_save_path': 'models/fine_tuned_model',
            'min_samples_for_training': 10,
            'adaptation_speed': 0.1
        }
    
    def _get_context_string(self) -> str:
        """Формирует строку контекста из последних сообщений"""
        if not self.conversation_context:
            return ""
        
        context_parts = []
        for msg in self.conversation_context[-self.max_context_length:]:
            context_parts.append(f"User: {msg['user']}")
            context_parts.append(f"Bot: {msg['bot']}")
        
        return "\n".join(context_parts)
    
    def _update_context(self, user_input: str, bot_response: str):
        """Обновляет контекст разговора"""
        self.conversation_context.append({
            'user': user_input,
            'bot': bot_response,
            'timestamp': datetime.now().isoformat()
        })
        
        # Ограничиваем размер контекста
        if len(self.conversation_context) > self.max_context_length:
            self.conversation_context.pop(0)
    
    def _should_trigger_training(self) -> bool:
        """Проверяет, пора ли проводить дообучение"""
        return (len(self.memory.conversations) >= 
                self.config['min_samples_for_training'] and
                len(self.memory.conversations) % self.config['min_samples_for_training'] == 0)
    
    def _try_auto_training(self):
        """Пытается автоматически запустить дообучение"""
        if self._should_trigger_training():
            print("\n🔄 АВТОМАТИЧЕСКОЕ ДООБУЧЕНИЕ...")
            self.run_training()
    
    def process_command(self, command: str) -> Optional[str]:
        """Обрабатывает специальные команды"""
        parts = command.strip().split(' ', 1)
        cmd = parts[0].lower()
        args = parts[1] if len(parts) > 1 else ""
        
        if cmd == '/quit' or cmd == '/exit':
            print("\n👋 Сохраняю прогресс и выхожу...")
            self.save_state()
            return None
        
        elif cmd == '/stats':
            return self._get_stats()
        
        elif cmd == '/profile':
            return self._get_user_profile()
        
        elif cmd == '/train':
            self.run_training()
            return "✅ Дообучение завершено!"
        
        elif cmd == '/clear':
            self.conversation_context = []
            return "🗑️ Контекст очищен"
        
        elif cmd == '/exec':
            if not args:
                return "❌ Укажите команду для выполнения: /exec <command>"
            
            if not CommandExecutor.is_safe_command(args):
                return "⚠️ Эта команда потенциально опасна и не может быть выполнена"
            
            print(f"\n⚙️ Выполняю: {args}")
            result = CommandExecutor.execute(args)
            
            self.stats['commands_executed'] += 1
            
            output = []
            if result['success']:
                output.append("✅ Успешно")
            else:
                output.append(f"❌ Ошибка (код {result['return_code']})")
            
            if result['stdout']:
                output.append(f"\n📤 STDOUT:\n{result['stdout']}")
            if result['stderr']:
                output.append(f"\n📥 STDERR:\n{result['stderr']}")
            
            return "\n".join(output)
        
        elif cmd == '/help':
            return self._get_help()
        
        else:
            return None
    
    def _get_stats(self) -> str:
        """Возвращает статистику бота"""
        model_stats = self.model.get_model_stats()
        
        stats_text = [
            "\n📊 СТАТИСТИКА БОТА",
            "=" * 40,
            f"💬 Всего взаимодействий: {self.stats['interactions']}",
            f"⌨️ Команд выполнено: {self.stats['commands_executed']}",
            f"🎯 Сессий обучения: {self.stats['training_sessions']}",
            f"📚 Записей в памяти: {len(self.memory.conversations)}",
            "",
            "🧠 ИНФОРМАЦИЯ О МОДЕЛИ:",
            f"   Название: {model_stats['model_name']}",
            f"   Дообучена: {'✅ Да' if model_stats['is_fine_tuned'] else '❌ Нет'}",
            f"   Параметров: {model_stats['parameters_count']:,}",
            f"   Устройство: {model_stats['device']}",
            "=" * 40
        ]
        
        return "\n".join(stats_text)
    
    def _get_user_profile(self) -> str:
        """Возвращает профиль пользователя"""
        profile = self.memory.get_user_profile()
        patterns = profile['patterns']
        
        profile_text = [
            "\n👤 ПРОФИЛЬ ПОЛЬЗОВАТЕЛЯ",
            "=" * 40,
            f"📅 Первый разговор: {profile['first_interaction'] or 'Нет данных'}",
            f"🕐 Последний разговор: {profile['last_interaction'] or 'Нет данных'}",
            f"💬 Всего разговоров: {profile['conversation_count']}",
            "",
            "🎭 СТИЛЬ ОБЩЕНИЯ:",
            f"   Средняя длина ответа: {patterns['avg_response_length']:.1f} симв.",
            f"   Использование эмодзи: {patterns['emoji_usage']*100:.1f}%",
            f"   Частота вопросов: {patterns['question_frequency']*100:.1f}%",
            f"   Словарный запас: {len(patterns['vocabulary'])} уникальных слов",
            "",
            "🏷️ Популярные темы:",
        ]
        
        # Добавляем темы из последних разговоров
        recent_topics = []
        for conv in self.memory.conversations[-20:]:
            topics = self.memory._extract_topics(conv['user_input'])
            recent_topics.extend(topics)
        
        if recent_topics:
            topic_counts = {}
            for topic in recent_topics:
                topic_counts[topic] = topic_counts.get(topic, 0) + 1
            
            for topic, count in sorted(topic_counts.items(), key=lambda x: x[1], reverse=True)[:5]:
                profile_text.append(f"   • {topic}: {count}")
        else:
            profile_text.append("   Пока недостаточно данных")
        
        profile_text.append("=" * 40)
        
        return "\n".join(profile_text)
    
    def _get_help(self) -> str:
        """Возвращает справку"""
        help_text = [
            "\n📖 СПРАВКА",
            "=" * 40,
            "Этот бот самообучается и адаптируется под ваш стиль общения.",
            "Чем больше вы общаетесь, тем больше он становится похож на вас!",
            "",
            "КОМАНДЫ:",
            "  /stats - показать статистику бота",
            "  /profile - показать ваш профиль и стиль общения",
            "  /train - принудительно запустить дообучение модели",
            "  /clear - очистить контекст разговора",
            "  /exec <command> - выполнить системную команду",
            "  /help - показать эту справку",
            "  /quit - сохранить и выйти",
            "",
            "ПРОСТОЕ ОБЩЕНИЕ:",
            "  Просто пишите сообщения, и бот будет отвечать,",
            "  постепенно перенимая ваш стиль!",
            "=" * 40
        ]
        
        return "\n".join(help_text)
    
    def run_training(self):
        """Запускает процесс дообучения"""
        training_data = self.memory.get_training_data(
            min_samples=self.config['min_samples_for_training']
        )
        
        if not training_data:
            print(f"❌ Недостаточно данных для обучения (минимум {self.config['min_samples_for_training']})")
            return
        
        user_profile = self.memory.get_user_profile()
        self.model.fine_tune_on_user_data(training_data, user_profile)
        self.stats['training_sessions'] += 1
    
    def generate_response(self, user_input: str) -> str:
        """Генерирует ответ на ввод пользователя"""
        # Получаем контекст
        context = self._get_context_string()
        
        # Получаем профиль стиля пользователя
        user_profile = self.memory.get_user_profile()
        user_style = user_profile['patterns']
        
        # Генерируем ответ
        response = self.model.generate_response(
            user_input=user_input,
            context=context,
            user_style=user_style
        )
        
        return response
    
    def save_state(self):
        """Сохраняет состояние бота"""
        self.memory.save_memory()
        print("💾 Прогресс сохранён")
    
    def run(self):
        """Запускает основной цикл бота"""
        print("🚀 Бот готов к работе! Начните общение.\n")
        
        while True:
            try:
                # Получаем ввод пользователя
                user_input = input("👤 Вы: ").strip()
                
                if not user_input:
                    continue
                
                self.stats['interactions'] += 1
                
                # Проверяем команды
                if user_input.startswith('/'):
                    response = self.process_command(user_input)
                    if response is None:  # Команда выхода
                        break
                    print(f"\n🤖 Бот: {response}\n")
                    continue
                
                # Генерируем ответ
                print("\n🤖 Бот: ", end='', flush=True)
                response = self.generate_response(user_input)
                print(response)
                
                # Сохраняем в память
                self.memory.add_conversation(
                    user_input=user_input,
                    bot_response=response,
                    context={'context_length': len(self.conversation_context)}
                )
                
                # Обновляем контекст
                self._update_context(user_input, response)
                
                # Проверяем авто-обучение
                self._try_auto_training()
                
                print()  # Пустая строка для разделения
                
            except KeyboardInterrupt:
                print("\n\n👋 Прервано пользователем. Сохраняю прогресс...")
                self.save_state()
                break
            except Exception as e:
                print(f"\n❌ Ошибка: {e}")
                print("Попробуйте ещё раз...\n")


def main():
    """Точка входа"""
    bot = SelfLearningBot()
    bot.run()


if __name__ == '__main__':
    main()
