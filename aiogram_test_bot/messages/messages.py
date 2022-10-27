from aiogram.types import ReplyKeyboardMarkup

msgs = {
    "hello": "Здравствуйтe! Здесь Вы можете узнать о нашем замечательном 🏨 'Bot-Hotel' 👇",

    "intent_reservation": "База знаний про бронирование 🛏",
    "intent_services": "База зананий про услуги 🧹",
    "intent_rooms": "База знаний про номера 📄",
    "intent_time": "База знаний про расписание 🕐",
    "intent_parking": "База знаний про парковку 🚗",
}

main_keyboard = ReplyKeyboardMarkup(resize_keyboard=True)
main_keyboard.add("Вывести список пицц")
main_keyboard.insert("Сделать заказ")