import os
import logging
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
from telegram import Update
from telegram.ext import ApplicationBuilder, ContextTypes, CommandHandler, MessageHandler, filters
from pipeline import DramaPipeline

# 1. Configuración de Logging para debug en clase
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

# 2. Inicialización
load_dotenv()
TOKEN = os.getenv('TELEGRAM_TOKEN')
pipeline = DramaPipeline()

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Comando /start"""
    await update.message.reply_text(
        "Kien eres? 👀\n"
        "Envíame una foto de tu cara y adivinaré tu historia."
    )

async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Maneja la recepción de fotos y dispara el pipeline de IA"""
    status_msg = await update.message.reply_text("Procesando... (YOLO -> SAM -> Gemma -> Imagen)")
    
    try:
        # 1. Crear directorio si no existe
        folder = Path(r"parcial_3\telegram_bot\weights\inputs")
        folder.mkdir(exist_ok=True)

        # 2. Generar nombre dinámico: ID_YYYYMMDD_HHMMSS.jpg
        user_id = update.message.from_user.id
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        input_path = folder / f"{user_id}_{timestamp}.jpg"

        # 3. Descargar la foto enviada
        photo_file = await update.message.photo[-1].get_file()
        await photo_file.download_to_drive(input_path)
        
        # Ejecutar el Pipeline de IA enviando la ruta como string
        story, crop_path = pipeline.process_image(str(input_path))
        
        # Enviar respuesta al usuario
        await status_msg.edit_text(story)
        
        if crop_path:
            caption = "Caricatura dramática (Nano Banana Style) 🎨" if "caricature" in str(crop_path) else "Recorte de SAM."
            with open(crop_path, 'rb') as photo:
                await update.message.reply_photo(photo=photo, caption=caption)

    except Exception as e:
        await update.message.reply_text(f"Error en el multiverso: {str(e)}")

if __name__ == '__main__':
    # 3. Construcción de la Aplicación (Estructura v20+)
    application = ApplicationBuilder().token(TOKEN).build()
    
    # Registro de handlers
    application.add_handler(CommandHandler('start', start))
    application.add_handler(MessageHandler(filters.PHOTO, handle_photo))
    
    print("Bot iniciado! 🤖")
    application.run_polling()
