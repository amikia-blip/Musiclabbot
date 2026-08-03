"""
ربات تلگرام تشخیص سبک موسیقی
---------------------------------
این ربات یک فایل صوتی (آهنگ/ویس) از کاربر می‌گیرد و:
1) سبک موسیقی را با یک مدل آماده‌ی هوش مصنوعی حدس می‌زند.
2) درباره‌ی «با چه هوش مصنوعی ساخته شده» یک برآورد بسیار محتاطانه و آزمایشی می‌دهد
   (چون هیچ ابزار قابل‌اعتمادی برای این کار به‌صورت عمومی وجود ندارد).

نحوه‌ی اجرا در README.md توضیح داده شده.
"""

import os
import logging
import tempfile

from telegram import Update
from telegram.ext import Application, MessageHandler, ContextTypes, filters

from transformers import pipeline
import librosa
import numpy as np
import imageio_ffmpeg
import subprocess

FFMPEG_PATH = imageio_ffmpeg.get_ffmpeg_exe()

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# توکن ربات را از متغیر محیطی BOT_TOKEN می‌خوانیم (در README توضیح داده شده)
BOT_TOKEN = os.environ.get("BOT_TOKEN")

# مدل تشخیص سبک موسیقی (رایگان، از هاگینگ‌فیس)
logger.info("در حال بارگذاری مدل تشخیص سبک موسیقی... (اولین بار کمی طول می‌کشد)")
genre_classifier = pipeline(
    "audio-classification",
    model="dima806/music_genres_classification",
)
logger.info("مدل با موفقیت بارگذاری شد.")


def convert_to_wav(input_path: str) -> str:
    """هر فرمت صوتی (ogg, mp3, m4a, ...) را به wav ۱۶kHz تبدیل می‌کند."""
    out_path = input_path + "_converted.wav"
    subprocess.run(
        [
            FFMPEG_PATH, "-y", "-i", input_path,
            "-ar", "16000", "-ac", "1", out_path,
        ],
        check=True,
        capture_output=True,
    )
    return out_path


def guess_genre(wav_path: str):
    """سبک موسیقی را با مدل هوش مصنوعی حدس می‌زند و ۳ نتیجه‌ی برتر را برمی‌گرداند."""
    results = genre_classifier(wav_path, top_k=3)
    # هر نتیجه شامل label و score (احتمال) است
    return results


def rough_ai_generation_estimate(wav_path: str) -> str:
    """
    یک برآورد بسیار ابتدایی و آزمایشی از «مصنوعی بودن احتمالی» صدا،
    بر اساس ویژگی‌های طیفی ساده (نه یک مدل تشخیص واقعی).
    این بخش قطعی نیست و فقط یک حدس کلی است.
    """
    y, sr = librosa.load(wav_path, sr=16000)

    # چند ویژگی ساده‌ی طیفی که گاهی در صداهای تولیدشده با AI متفاوت‌اند
    spectral_flatness = float(np.mean(librosa.feature.spectral_flatness(y=y)))
    spectral_bandwidth = float(np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr)))
    zero_crossing = float(np.mean(librosa.feature.zero_crossing_rate(y=y)))

    # این آستانه‌ها کاملاً تجربی و غیرقطعی‌اند — فقط برای نمایش یک برآورد کلی
    ai_score = 0
    if spectral_flatness > 0.02:
        ai_score += 1
    if spectral_bandwidth < 2200:
        ai_score += 1
    if zero_crossing < 0.05:
        ai_score += 1

    if ai_score >= 2:
        return "احتمال کمی وجود دارد که این قطعه با هوش مصنوعی تولید شده باشد (برآورد آزمایشی، غیرقطعی)"
    else:
        return "نشانه‌ی قوی از تولید با هوش مصنوعی در ویژگی‌های ساده‌ی صدا دیده نشد (این یعنی احتمالاً طبیعی/ضبط‌شده است، اما قطعی نیست)"


async def handle_audio(update: Update, context: ContextTypes.DEFAULT_TYPE):
    message = update.message
    audio_file = message.audio or message.voice or message.document

    if audio_file is None:
        await message.reply_text("لطفاً یک فایل صوتی یا آهنگ برام بفرست 🎵")
        return

    await message.reply_text("در حال تحلیل آهنگ... چند لحظه صبر کن ⏳")

    try:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tg_file = await context.bot.get_file(audio_file.file_id)
            input_path = os.path.join(tmp_dir, "input_audio")
            await tg_file.download_to_drive(input_path)

            wav_path = convert_to_wav(input_path)

            genre_results = guess_genre(wav_path)
            ai_note = rough_ai_generation_estimate(wav_path)

        genre_lines = "\n".join(
            f"• {r['label']} — {r['score'] * 100:.1f}٪"
            for r in genre_results
        )

        reply = (
            f"🎧 نتیجه‌ی تحلیل:\n\n"
            f"سبک موسیقی (به ترتیب احتمال):\n{genre_lines}\n\n"
            f"🤖 وضعیت هوش مصنوعی:\n{ai_note}\n\n"
            f"⚠️ توجه: تشخیص «دقیقاً با چه هوش مصنوعی و چه استایلی ساخته شده» "
            f"(مثلاً Suno یا Udio) در حال حاضر با هیچ ابزار قابل‌اعتمادی امکان‌پذیر نیست، "
            f"پس این بخش را فقط به‌عنوان یک حدس کلی در نظر بگیر."
        )
        await message.reply_text(reply)

    except Exception as e:
        logger.exception("خطا در تحلیل فایل صوتی")
        await message.reply_text(
            "متأسفانه در تحلیل این فایل مشکلی پیش اومد. "
            "لطفاً یک فایل صوتی معتبر (mp3, ogg, wav, ...) بفرست."
        )


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "سلام! 👋\n"
        "یه آهنگ یا فایل صوتی برام بفرست تا سبکش رو تحلیل کنم "
        "و یه برآورد کلی از احتمال تولید با هوش مصنوعی بدم."
    )


def main():
    if not BOT_TOKEN:
        raise RuntimeError(
            "متغیر محیطی BOT_TOKEN تنظیم نشده. توکن ربات رو از BotFather بگیر و ست کن."
        )

    app = Application.builder().token(BOT_TOKEN).build()

    app.add_handler(MessageHandler(filters.Regex("^/start$"), start))
    app.add_handler(
        MessageHandler(filters.AUDIO | filters.VOICE | filters.Document.AUDIO, handle_audio)
    )

    logger.info("ربات روشن شد و منتظر پیام است...")
    app.run_polling()


if __name__ == "__main__":
    main()
