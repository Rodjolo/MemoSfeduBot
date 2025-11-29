import logging
import os
from dotenv import load_dotenv
from telegram import Update
from telegram.ext import ApplicationBuilder, ContextTypes, CommandHandler, MessageHandler, filters
from openai import AsyncOpenAI

load_dotenv()

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
LM_STUDIO_URL = "http://localhost:1234/v1"
MODEL_ID = "local-model" 

logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

client = AsyncOpenAI(base_url=LM_STUDIO_URL, api_key="lm-studio")

# Хранилище контекста: ключ = user_id, значение = список словарей с сообщениями
# Это обеспечивает независимую историю диалога для каждого пользователя
user_context = {}

SYSTEM_PROMPT = """You are a helpful and friendly AI assistant. You have memory of our conversation, 
so you can reference previous messages and provide contextual responses. Always be polite and clear."""

# Максимальная длина контекста для предотвращения переполнения токенов
MAX_CONTEXT_LENGTH = 20

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Инициализирует или сбрасывает диалог для пользователя."""
    user_id = update.effective_user.id
    user_name = update.effective_user.first_name
    user_context[user_id] = [{"role": "system", "content": SYSTEM_PROMPT}]
    
    logging.info(f"User {user_id} ({user_name}) started the bot")
    
    welcome_message = (
        f"Привет, {user_name}! 👋\n\n"
        "Я LocalMind — Telegram-бот с локальной языковой моделью.\n\n"
        "🧠 Я запоминаю наш разговор и могу отвечать с учетом контекста.\n"
        "💬 Просто напиши мне что-нибудь, и я отвечу!\n\n"
        "Команды:\n"
        "/help — показать справку\n"
        "/clear — очистить историю диалога"
    )
    await update.message.reply_text(welcome_message)

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Отображает справочную информацию."""
    help_text = (
        "📖 *Справка по боту LocalMind*\n\n"
        "Я — бот с локальной языковой моделью (LM Studio).\n"
        "Я помню историю нашего диалога и отвечаю с учетом контекста.\n\n"
        "*Доступные команды:*\n"
        "/start — начать/перезапустить бота\n"
        "/help — показать эту справку\n"
        "/clear — очистить историю диалога\n\n"
        "*Как использовать контекст:*\n"
        "Попробуйте задать несколько связанных вопросов.\n"
        "Например:\n"
        "1️⃣ 'Расскажи про Python'\n"
        "2️⃣ 'А какие у него преимущества?' (я пойму, что речь о Python)\n"
        "3️⃣ 'Где он используется?' (контекст сохраняется)\n\n"
        "Используйте /clear, чтобы начать новую тему."
    )
    await update.message.reply_text(help_text, parse_mode='Markdown')

async def clear(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Очищает историю диалога для пользователя.
    Шаг 4: Реализация команды очистки контекста
    """
    user_id = update.effective_user.id
    
    # Проверяем, есть ли у пользователя контекст
    messages_count = len(user_context.get(user_id, [])) - 1  # -1 для системного сообщения
    
    # Очищаем контекст
    user_context[user_id] = [{"role": "system", "content": SYSTEM_PROMPT}]
    
    logging.info(f"User {user_id} cleared context ({messages_count} messages removed)")
    
    await update.message.reply_text(
        "🗑️ Контекст диалога очищен.\n\n"
        f"Удалено сообщений: {messages_count}\n\n"
        "Я забыл всё, о чем мы говорили ранее. Давайте начнем сначала!"
    )

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Обрабатывает сообщения пользователей с управлением контекстом.
    Шаг 3: Реализация системы контекста
    - Сохраняет историю для каждого user_id (независимо для каждого пользователя)
    - Добавляет сообщения пользователя с меткой 'user'
    - Отправляет полный контекст в LM Studio
    - Добавляет ответы ассистента с меткой 'assistant'
    """
    user_id = update.effective_user.id
    user_message = update.message.text

    # Шаг 3: Инициализируем контекст, если пользователь новый
    if user_id not in user_context:
        user_context[user_id] = [{"role": "system", "content": SYSTEM_PROMPT}]
        logging.info(f"Created new context for user {user_id}")

    # Шаг 3: Добавляем сообщение пользователя в контекст
    user_context[user_id].append({"role": "user", "content": user_message})
    
    # Обрезаем контекст, если он слишком длинный (сохраняем системное сообщение + последние N сообщений)
    if len(user_context[user_id]) > MAX_CONTEXT_LENGTH:
        user_context[user_id] = [user_context[user_id][0]] + user_context[user_id][-(MAX_CONTEXT_LENGTH-1):]
        logging.info(f"Trimmed context for user {user_id}")
    
    # Показываем индикатор печати
    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action='typing')
    
    logging.info(f"User {user_id} sent message: {user_message[:50]}... (context length: {len(user_context[user_id])})")

    try:
        # Шаг 3: Отправляем полный контекст в LM Studio
        response = await client.chat.completions.create(
            model=MODEL_ID,
            messages=user_context[user_id],
            temperature=0.7,
            max_tokens=500
        )

        bot_response = response.choices[0].message.content

        # Шаг 3: Добавляем ответ ассистента в контекст
        user_context[user_id].append({"role": "assistant", "content": bot_response})
        
        logging.info(f"Bot responded to user {user_id}: {bot_response[:50]}...")

        await update.message.reply_text(bot_response)

    except Exception as e:
        logging.error(f"Error communicating with LM Studio for user {user_id}: {e}")
        await update.message.reply_text(
            "❌ Произошла ошибка при обращении к LM Studio.\n\n"
            "Пожалуйста, убедитесь что:\n"
            "1️⃣ LM Studio запущен\n"
            "2️⃣ Модель загружена\n"
            "3️⃣ Сервер активен (кнопка Start Server)\n"
            "4️⃣ Сервер доступен по адресу http://localhost:1234\n\n"
            f"Детали ошибки: {str(e)[:100]}"
        )

if __name__ == '__main__':
    if not TELEGRAM_TOKEN or TELEGRAM_TOKEN == "your_telegram_bot_token_here":
        print("Error: Please set your TELEGRAM_TOKEN in the .env file.")
    else:
        application = ApplicationBuilder().token(TELEGRAM_TOKEN).build()

        # Регистрируем обработчики команд
        start_handler = CommandHandler('start', start)
        help_handler = CommandHandler('help', help_command)
        clear_handler = CommandHandler('clear', clear)
        message_handler = MessageHandler(filters.TEXT & (~filters.COMMAND), handle_message)

        application.add_handler(start_handler)
        application.add_handler(help_handler)
        application.add_handler(clear_handler)
        application.add_handler(message_handler)

        print("="*50)
        print("🤖 LocalMind Bot is running...")
        print("📡 Connected to LM Studio at", LM_STUDIO_URL)
        print("✅ Ready to accept messages!")
        print("="*50)
        application.run_polling()
