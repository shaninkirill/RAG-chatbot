import logging
from aiogram import Bot, Dispatcher, types
from aiogram.utils import executor
import os
from dotenv import load_dotenv
from rag import RagWorker

load_dotenv()

API_TOKEN = os.getenv('API_TOKEN')

logging.basicConfig(level=logging.INFO)

rag_worker = RagWorker()
bot = Bot(token=API_TOKEN)
dp = Dispatcher(bot)

@dp.message_handler(commands=['start'])
async def send_welcome(message: types.Message):
    await message.reply("Здравствуйте! 📚 Я – бот, который знает всё! (ну, почти всё, что есть в Википедии 😉). Спрашивайте! 😄")

@dp.message_handler()
async def handle_message(message: types.Message):
    user_message = message.text

    response = rag_worker.search(user_message)

    await message.answer(response)

if __name__ == '__main__':
    executor.start_polling(dp, skip_updates=True)