import telebot
import os
from pipeline import DramaPipeline

# CONFIGURACIÓN
TOKEN = 'TU_TELEGRAM_TOKEN_AQUI'
bot = telebot.TeleBot(TOKEN)
pipeline = DramaPipeline()

print("Drama Bot iniciado y esperando fotos...")

@bot.message_handler(commands=['start', 'help'])
def send_welcome(message):
    bot.reply_to(message, "¡Bienvenido al Oráculo del Drama! 🎭\nEnvíame una foto de tu cara (o de alguien sufriendo el fin de semestre) y revelaré su trágica historia.")

@bot.message_handler(content_types=['photo'])
def handle_photo(message):
    try:
        msg = bot.reply_to(message, "Procesando el drama... (YOLO está buscando tu alma, SAM la está recortando...)")
        
        # Descargar la foto
        file_info = bot.get_file(message.photo[-1].file_id)
        downloaded_file = bot.download_file(file_info.file_path)
        
        photo_path = "input_photo.jpg"
        with open(photo_path, 'wb') as new_file:
            new_file.write(downloaded_file)
            
        # Ejecutar Pipeline
        story, crop_path = pipeline.process_image(photo_path)
        
        # Responder con la historia
        bot.edit_message_text(story, message.chat.id, msg.message_id)
        
        # Opcional: Enviar el recorte que vio Gemma para transparencia pedagógica
        if crop_path:
            with open(crop_path, 'rb') as photo:
                bot.send_photo(message.chat.id, photo, caption="Esto es lo que Gemma analizó.")

    except Exception as e:
        bot.reply_to(message, f"Hubo un error en el multiverso: {str(e)}")

if __name__ == "__main__":
    bot.infinity_polling()
