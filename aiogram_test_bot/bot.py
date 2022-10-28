import re
from .messages import get_message_text, main_keyboard

import logging

from aiogram import Bot, Dispatcher, executor, types
from aiogram.dispatcher import FSMContext
from aiogram.dispatcher.filters.state import StatesGroup, State

from aiogram.contrib.fsm_storage.files import JSONStorage
from database import UsersTable, PizzaTable, OrdersTable, FileTable
from aiogram.types import InlineKeyboardMarkup, InlineKeyboardButton, ReplyKeyboardMarkup, KeyboardButton, \
    InputMediaPhoto
from .settings import API_TOKEN
import os

logging.basicConfig(level=logging.DEBUG)

if "https_proxy" in os.environ:
    proxy_url = os.environ["https_proxy"]
    bot = Bot(token=API_TOKEN, proxy=proxy_url)
else:
    bot = Bot(token=API_TOKEN)



storage = JSONStorage("states.json")

dp = Dispatcher(bot, storage=storage)

get_intent_callback = lambda text: "intent_not_found"


# analog
# def get_intent_callback(text):
#     return "intent_not_found"

class StateMachine(StatesGroup):
    main_state = State()

async def send_photo(message, filename, caption=None, reply_markup=None):
    file_id = FileTable.get_file_id_by_file_name(filename)
    if file_id is None:
        # upload_file
        with open(filename, 'rb') as photo:
            result = await message.answer_photo(
                photo,
                caption=caption,
                reply_markup=reply_markup
            )
            file_id = result.photo[0].file_id
            FileTable.create(telegram_file_id=file_id, file_name=filename)
    else:
        await bot.send_photo(
            message.from_user.id,
            file_id,
            caption=caption,
            reply_markup=reply_markup
        )


@dp.message_handler(commands=['start', 'help'], state="*")
async def send_welcome(message: types.Message):
    await StateMachine.main_state.set()
    user_last_name = message.from_user.last_name
    user_first_name = message.from_user.first_name
    markup = ReplyKeyboardMarkup(resize_keyboard=True, one_time_keyboard=True) \
        .add("номер", "забронировать", "расписание", "услуги", "парковка")
    await message.answer(get_message_text("hello",
                                          user_name=user_first_name,
                                          user_name2=user_last_name), reply_markup=markup)

    await StateMachine.main_state.set()

    logging.info(f"{message.from_user.username}: {message.text}")

@dp.message_handler(state=StateMachine.main_state)
async def main_state_handler(message: types.Message, state: FSMContext):

    intent = get_intent_callback(message.text)

    messages_from_intent = {
        "start": "hello",
        "стоянка": "intent_parking",
        "парковка": "intent_parking",
        "время": "intent_time",
        "расписание": "intent_time",
        "номер": "intent_rooms",
        "услуга": "intent_services",
        "забронировать": "intent_reservation",
    }

    if intent == "номер":
        for pizza in PizzaTable.get_menu():
            markup = InlineKeyboardMarkup()
            markup.add(InlineKeyboardButton("забронировать", callback_data="забронировать"))
            await send_photo(
                message,
                f'room_{pizza.pizza_id}.jpg',
                reply_markup=markup
            )

    elif intent == "забронировать":
        pass

    elif intent == "расписание":
        await send_photo(
            message,
            f'time.jpg')

    elif intent == "время":
        await send_photo(
            message,
            f'time.jpg')


    elif intent == "услуга":
        await send_photo(
            message,
            f'price.jpg')

    elif intent == "парковка":
        await send_photo(
            message,
            f'parking.jpg')

    elif intent == "стоянка":
        await send_photo(
            message,
            f'parking.jpg')


    await message.answer(get_message_text(messages_from_intent[intent]))

    logging.info(f"{message.from_user.username}: ({intent})  {message.text}")


@dp.callback_query_handler(text_startswith="забронировать", state=StateMachine.main_state)
async def main_state_handler(call: types.CallbackQuery, state: FSMContext):
    intent = get_intent_callback(call.data)
    messages_from_intent = {
        "забронировать": "intent_reservation",
    }
    await call.message.answer(get_message_text(messages_from_intent[intent]))
    await StateMachine.main_state.set()


def run_bot(_get_intent_callback):
    if _get_intent_callback is not None:
        global get_intent_callback
        get_intent_callback = _get_intent_callback
    executor.start_polling(dp, skip_updates=True)


if __name__ == '__main__':
    run_bot(get_intent_callback)