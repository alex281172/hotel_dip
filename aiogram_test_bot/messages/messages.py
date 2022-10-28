from aiogram.types import ReplyKeyboardMarkup

msgs = {
    "hello": "Здравствуйте, {user_name} {user_name2}! Здесь Вы можете узнать о нашем замечательном 🏨 'Bot-Hotel' 👇",

    "intent_reservation": "Для бронирования Вам надо позвонить ☎️+7(999)999-99-99  🛏",
    "intent_services": "База зананий про услуги 🧹",
    "intent_rooms": "База знаний про номера 📄",
    "intent_time": "База знаний про расписание 🕐",
    "intent_parking": "База знаний про парковку 🚗",


    "registered": "{user_name} {user_name2}! Вы уже зарегестрированы. Можете войти или удалить аккаунт",
}

main_keyboard = ReplyKeyboardMarkup(resize_keyboard=True)
main_keyboard.add("Вывести список пицц")
main_keyboard.insert("Сделать заказ")