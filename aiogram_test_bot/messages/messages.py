from aiogram.types import ReplyKeyboardMarkup

msgs = {
    "hello": "Здравствуйте! Какой у Вас вопрос?",
    "intent_parking": "База знаний по парковкам: \nDONE",
    "intent_time": "База знаний по рассписанию: \nDONE",
    "intent_reservation": "База знаний по бронированию: \nDONE",
    "intent_services": "База знаний по услугам: \nDONE",
    "intent_rooms": "База знаний по комнатам: \nDONE",
    "intent_none": "Я Вас не понял =( ",
}

main_keyboard = ReplyKeyboardMarkup(resize_keyboard=True)
main_keyboard.add("Вывести список пицц")
main_keyboard.insert("Сделать заказ")