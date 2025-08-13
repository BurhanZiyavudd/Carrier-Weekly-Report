from aiogram import Bot, Dispatcher, F, types
from aiogram.filters import CommandStart, Command
from aiogram.types import CallbackQuery, InlineKeyboardButton, InlineKeyboardMarkup, FSInputFile, ForceReply, BufferedInputFile
import asyncio
import logging
from logging.handlers import RotatingFileHandler
import re
import os, tempfile, time
import json
from pathlib import Path
from datetime import datetime
from itertools import chain
import pandas as pd
import difflib

from config import BOT_TOKEN
from loadbot.excel_file_writer import save_json_to_excel, flatten_dict
from loadbot.ocr import extract_info_from_file
from loadbot.parser import get_json

# ===================== Paths & Logging (temp dir; stateless) =====================

BASE_DATA_DIR = Path(os.getenv("REPORT_BOT_DATA_DIR", tempfile.gettempdir())) / "report_maker"
DOWNLOAD_DIR = BASE_DATA_DIR / "downloads"
REPORTS_DIR  = BASE_DATA_DIR / "reports"
DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

log_path = BASE_DATA_DIR / "bot.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        RotatingFileHandler(str(log_path), maxBytes=2*1024*1024, backupCount=3),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ================================ Bot & Dispatcher ================================

bot = Bot(token=BOT_TOKEN)
dp = Dispatcher()

# ================================ Globals & Helpers ================================

ALLOWED_EXTENSIONS = {"pdf", "png", "jpg", "jpeg"}
SIMILARITY_THRESHOLD = 0.86
MAX_UPLOAD = 25 * 1024 * 1024  # 25MB hard cap for documents

# Pending maps:
# key = (chat_id, prompt_message_id)
# values carry row + temp file + original uploader id
PENDING: dict[tuple[int, int], dict] = {}
PENDING_CONFIRM: dict[tuple[int, int], dict] = {}

def report_path(chat_id: int) -> Path:
    """One workbook per chat (DM or group)."""
    return REPORTS_DIR / f"{chat_id}_monthly_report.xlsx"

def purge_temp_files(older_than_days: int | None = None) -> tuple[int, int]:
    deleted = 0
    freed = 0
    cutoff = None if older_than_days is None else (time.time() - older_than_days * 86400)
    for p in chain(REPORTS_DIR.glob("*.xlsx"), DOWNLOAD_DIR.glob("*")):
        try:
            if not p.exists() or not p.is_file():
                continue
            if cutoff and p.stat().st_mtime >= cutoff:
                continue
            freed += p.stat().st_size
            p.unlink()
            deleted += 1
        except Exception as e:
            logger.warning("Failed to delete %s: %s", p, e)
    return deleted, freed

def sanitize_filename(name: str) -> str:
    name = (name or "").strip().replace(" ", "_")
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name)[:200] or f"file_{int(datetime.now().timestamp())}"

def strip_code_fences(s: str) -> str:
    s = s.strip()
    if s.startswith("```"):
        s = re.sub(r"^```(?:json)?\s*", "", s)
        s = re.sub(r"\s*```$", "", s)
    return s.strip()

def ensure_dict(structured_items):
    if isinstance(structured_items, dict):
        return structured_items
    if isinstance(structured_items, str):
        payload = strip_code_fences(structured_items)
        try:
            return json.loads(payload)
        except json.JSONDecodeError:
            logger.warning("Parser returned non-JSON; storing as raw text")
            return {"raw_text": payload}
    return {"raw_text": str(structured_items)}

def normalize_driver(name: str) -> str:
    name = re.sub(r"[,\s]+", " ", (name or "")).strip()
    return name.title()

def get_existing_drivers(chat_id: int) -> list[str]:
    xlsx = report_path(chat_id)
    if not xlsx.exists():
        return []
    try:
        df = pd.read_excel(xlsx, sheet_name="All Loads")
    except Exception:
        try:
            df = pd.read_excel(xlsx)
        except Exception:
            return []
    if "Driver name" not in df.columns:
        return []
    names = (
        df["Driver name"].dropna().astype(str).map(normalize_driver).str.strip()
    )
    return sorted({n for n in names if n})

async def run_ocr(path: Path) -> str:
    return await asyncio.to_thread(extract_info_from_file, str(path))

# Per-file locks so different chats don’t block each other
_LOCKS: dict[Path, asyncio.Lock] = {}

def _get_lock(path: Path) -> asyncio.Lock:
    p = path.resolve()                          # normalize
    lock = _LOCKS.get(p)
    if lock is None:
        lock = _LOCKS.setdefault(p, asyncio.Lock())
    return lock

async def save_excel_threadsafe(row: dict, excel_path: Path):
    async with _get_lock(excel_path):
        # run the writer off the event loop
        await asyncio.to_thread(save_json_to_excel, row, str(excel_path))

def make_action_kb() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        inline_keyboard=[[
            InlineKeyboardButton(text="📥 Download Report", callback_data="download_report"),
            InlineKeyboardButton(text="🧮 Calculate Total", callback_data="calculate_total")
        ]]
    )

# ================================ Commands / Handlers ================================

@dp.message(CommandStart())
async def start(message: types.Message):
    await message.answer(
        "Welcome to Report Maker Bot!\n\n"
        "Send a load confirmation as a **PDF/JPG/PNG**.\n"
        "I’ll parse it, then ask which driver to save it under."
    )

@dp.message(F.content_type == types.ContentType.DOCUMENT)
async def handle_document(message: types.Message):
    file_name = sanitize_filename(message.document.file_name or f"{message.document.file_id}.bin")
    file_ext = file_name.split(".")[-1].lower() if "." in file_name else ""
    if file_ext not in ALLOWED_EXTENSIONS:
        await message.answer("❌ Only PDF, PNG, JPG, or JPEG files are accepted.")
        return
    # size guard
    if (message.document.file_size or 0) > MAX_UPLOAD:
        await message.answer("❌ File too large. Please send a file under 25MB.")
        return

    file_info = await bot.get_file(message.document.file_id)
    local_path = DOWNLOAD_DIR / file_name
    await bot.download_file(file_info.file_path, destination=local_path)

    processing_msg = await message.answer("⏳ Processing your document…")
    try:
        extracted_info = await run_ocr(local_path)
        if not extracted_info or not str(extracted_info).strip():
            await processing_msg.edit_text("❌ OCR returned empty text.")
            local_path.unlink(missing_ok=True); return

        parsed = get_json(extracted_info)
        if not parsed or (isinstance(parsed, str) and not parsed.strip()):
            await processing_msg.edit_text("❌ Could not parse extracted data into valid JSON.")
            local_path.unlink(missing_ok=True); return

        parsed_dict = ensure_dict(parsed)
        flattened = flatten_dict(parsed_dict)

        # Ask for driver & stash pending
        existing = get_existing_drivers(message.chat.id)
        if existing:
            rows = [[InlineKeyboardButton(text=name, callback_data=f"pick_driver:{name}")]
                    for name in existing[:8]]
            rows.append([InlineKeyboardButton(text="➕ Type a new name", callback_data="pick_driver:__new__")])
            prompt = await message.answer(
                "📝 For which driver?\n"
                "Tip: use the same spelling you used before. Pick below or type a new one.",
                reply_markup=InlineKeyboardMarkup(inline_keyboard=rows),
            )
        else:
            prompt = await message.answer(
                "📝 For which driver? Please reply with the driver’s full name.\n"
                "Tip: use the same spelling every time (e.g., “John Doe”).",
                reply_markup=ForceReply(selective=True),
            )

        PENDING[(message.chat.id, prompt.message_id)] = {
            "row": flattened,
            "src_path": local_path,
            "user_id": message.from_user.id,
        }

        await processing_msg.edit_text(
            "✅ Parsed. Now please reply with the driver name to the message I just sent.\n"
            "❗️Make sure the spelling matches previous loads for the same driver."
        )
    except Exception as e:
        logger.exception("Document processing failed: %s", e)
        await processing_msg.edit_text("❌ An unexpected error occurred while processing this document.")
        local_path.unlink(missing_ok=True)

@dp.message(F.content_type == types.ContentType.PHOTO)
async def handle_photo(message: types.Message):
    photo = message.photo[-1]
    file_info = await bot.get_file(photo.file_id)
    local_path = DOWNLOAD_DIR / f"{photo.file_id}.jpg"
    await bot.download_file(file_info.file_path, destination=local_path)

    processing_msg = await message.answer("⏳ Processing your photo…")
    try:
        extracted_info = await run_ocr(local_path)
        if not extracted_info or not str(extracted_info).strip():
            await processing_msg.edit_text("❌ OCR returned empty text.")
            local_path.unlink(missing_ok=True); return

        parsed = get_json(extracted_info)
        if not parsed or (isinstance(parsed, str) and not parsed.strip()):
            await processing_msg.edit_text("❌ Could not parse extracted data into valid JSON.")
            local_path.unlink(missing_ok=True); return

        parsed_dict = ensure_dict(parsed)
        flattened = flatten_dict(parsed_dict)

        prompt = await message.answer(
            "📝 For which driver? Please reply with the driver’s full name.",
            reply_markup=ForceReply(selective=True)
        )
        PENDING[(message.chat.id, prompt.message_id)] = {
            "row": flattened,
            "src_path": local_path,
            "user_id": message.from_user.id,
        }

        await processing_msg.edit_text("✅ Parsed. Now please reply with the driver name to the message I just sent.")
    except Exception as e:
        logger.exception("Photo processing failed: %s", e)
        await processing_msg.edit_text("❌ An unexpected error occurred while processing this photo.")
        local_path.unlink(missing_ok=True)

@dp.message(F.reply_to_message)
async def handle_driver_name_reply(message: types.Message):
    key = (message.chat.id, message.reply_to_message.message_id)
    payload = PENDING.get(key)
    if not payload:
        return
    if message.from_user.id != payload["user_id"]:
        await message.answer("This prompt belongs to another user. Please upload your own document.")
        return

    typed = normalize_driver(message.text or "")
    if not typed:
        await message.answer("❌ Driver name cannot be empty. Please type the driver’s full name.")
        return

    row = payload["row"]
    src_path: Path = payload["src_path"]
    PENDING.pop(key, None)

    # Fuzzy check against existing for this chat
    existing = get_existing_drivers(message.chat.id)
    best = difflib.get_close_matches(typed, existing, n=1, cutoff=SIMILARITY_THRESHOLD)

    if best and best[0] != typed:
        suggested = best[0]
        kb = InlineKeyboardMarkup(inline_keyboard=[
            [InlineKeyboardButton(text=f"Use {suggested}", callback_data=f"confirm_driver:use:{suggested}")],
            [InlineKeyboardButton(text=f"Keep {typed}", callback_data=f"confirm_driver:keep:{typed}")],
        ])
        confirm_msg = await message.answer(
            f"Looks like you meant **{suggested}**.\n"
            f"Your entry: **{typed}**\n"
            "Which one should I use?",
            parse_mode="Markdown",
            reply_markup=kb,
        )
        PENDING_CONFIRM[(message.chat.id, confirm_msg.message_id)] = {
            "row": row, "src_path": src_path, "typed": typed, "suggested": suggested, "user_id": message.from_user.id
        }
        return

    # No strong match — save with typed
    row["Driver name"] = typed
    try:
        await save_excel_threadsafe(row, report_path(message.chat.id))
        await message.answer(
            f"✅ Saved under driver **{typed}**.",
            parse_mode="Markdown",
            reply_markup=make_action_kb(),
        )
    except Exception as e:
        logger.exception("Saving failed: %s", e)
        await message.answer("❌ Failed to save this load. Please try again.")
    finally:
        try: src_path.unlink(missing_ok=True)
        except Exception as e: logger.warning("Could not delete source file: %s", e)

@dp.callback_query(lambda c: c.data and c.data.startswith("pick_driver:"))
async def handle_pick_driver(callback: CallbackQuery):
    choice = callback.data.split(":", 1)[1]
    key = (callback.message.chat.id, callback.message.message_id)
    payload = PENDING.get(key)
    if not payload:
        await callback.answer(); return
    if callback.from_user.id != payload["user_id"]:
        await callback.answer("Not your upload.", show_alert=True); return

    PENDING.pop(key, None)
    row = payload["row"]
    src_path: Path = payload["src_path"]

    if choice == "__new__":
        prompt = await callback.message.answer(
            "Please type the driver’s full name:",
            reply_markup=ForceReply(selective=True),
        )
        PENDING[(callback.message.chat.id, prompt.message_id)] = {
            "row": row, "src_path": src_path, "user_id": callback.from_user.id
        }
        await callback.answer(); return

    row["Driver name"] = choice
    try:
        await save_excel_threadsafe(row, report_path(callback.message.chat.id))
        await callback.message.answer(
            f"✅ Saved under driver **{choice}**.",
            parse_mode="Markdown",
            reply_markup=make_action_kb(),
        )
    except Exception as e:
        logger.exception("Saving failed: %s", e)
        await callback.message.answer("❌ Failed to save this load. Please try again.")
    finally:
        try: src_path.unlink(missing_ok=True)
        except Exception as e: logger.warning("Could not delete source file: %s", e)
    await callback.answer()

@dp.callback_query(lambda c: c.data and c.data.startswith("confirm_driver:"))
async def handle_confirm_driver(callback: CallbackQuery):
    _, action, name = callback.data.split(":", 2)
    key = (callback.message.chat.id, callback.message.message_id)
    payload = PENDING_CONFIRM.get(key)
    if not payload:
        await callback.answer(); return
    if callback.from_user.id != payload["user_id"]:
        await callback.answer("Not your upload.", show_alert=True); return

    PENDING_CONFIRM.pop(key, None)
    row = payload["row"]
    src_path: Path = payload["src_path"]
    chosen = payload["suggested"] if action == "use" else payload["typed"]

    row["Driver name"] = chosen
    try:
        await save_excel_threadsafe(row, report_path(callback.message.chat.id))
        await callback.message.edit_text(f"✅ Saved under driver **{chosen}**.", parse_mode="Markdown")
        await callback.message.answer(reply_markup=make_action_kb(), text="You can download or total the report anytime.")
    except Exception as e:
        logger.exception("Saving failed: %s", e)
        await callback.message.answer("❌ Failed to save this load. Please try again.")
    finally:
        try: src_path.unlink(missing_ok=True)
        except Exception as e: logger.warning("Could not delete source file: %s", e)
    await callback.answer()

@dp.callback_query(F.data == "download_report")
async def handle_download_report(callback: CallbackQuery):
    xlsx = report_path(callback.message.chat.id)
    if not xlsx.exists():
        await callback.message.answer("No report found yet!")
        await callback.answer(); return
    try:
        await callback.message.answer_document(
            FSInputFile(str(xlsx)),
            caption="Here is your report."
        )
    except Exception as e:
        logger.exception("Failed to send report: %s", e)
        await callback.message.answer("❌ Failed to send the report. Please try again.")
        await callback.answer(); return
    # delete after sending (stateless)
    try:
        xlsx.unlink(missing_ok=True)
        await callback.message.answer("Report deleted. New entries will create a fresh file.")
    except Exception as e:
        logger.warning("Delete failed: %s", e)
    await callback.answer()

# -------- Totals helpers + handler --------

def _read_master_df(xlsx_path: Path) -> pd.DataFrame:
    try:
        return pd.read_excel(xlsx_path, sheet_name="All Loads")
    except ValueError:
        return pd.read_excel(xlsx_path)

def _clean_money_series(s: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(s):
        return s.fillna(0.0).astype(float)
    return pd.to_numeric(
        s.astype(str).str.replace(r"[\$\s,]", "", regex=True),
        errors="coerce"
    ).fillna(0.0)

def _find_total_series(df: pd.DataFrame):
    exact = [
        "total_rate_usd", "total_usd", "total rate usd",
        "total rate", "grand total", "total", "amount", "pay", "load total",
    ]
    lower_map = {c.lower(): c for c in df.columns}
    for key in exact:
        if key in lower_map:
            col = lower_map[key]
            return _clean_money_series(df[col]), col
    for col in df.columns:
        n = col.lower()
        if (("total" in n and any(k in n for k in ("usd","rate","amount","pay")))
            or n.endswith("_total") or "total rate" in n):
            return _clean_money_series(df[col]), col
    return None, None

def _reconstruct_total_from_parts(df: pd.DataFrame):
    import re as _re
    used, parts = [], []
    def pick(pats):
        for pat in pats:
            for col in df.columns:
                if _re.search(pat, col, flags=_re.IGNORECASE):
                    used.append(col); parts.append(_clean_money_series(df[col])); return True
        return False
    pick([r"\bline\s*haul\b", r"\blinehaul\b", r"\brate\b"])
    pick([r"\bfsc\b", r"\bfuel\s*surcharge\b", r"\bfuel\b"])
    pick([r"\baccessorials?\b"]); pick([r"\bdetention\b"]); pick([r"\blumper\b"])
    pick([r"\blayover\b"]); pick([r"\btonu\b"]); pick([r"\bstop\s*pay\b", r"\bextra\s*stop\b"])
    if not parts: return None, []
    total = parts[0]
    for s in parts[1:]: total = total.add(s, fill_value=0.0)
    return total, used

@dp.callback_query(F.data == "calculate_total")
async def handle_calculate_total(callback: CallbackQuery):
    xlsx = report_path(callback.message.chat.id)
    if not xlsx.exists():
        await callback.message.answer("No report found yet!")
        await callback.answer(); return

    def _calc():
        df = _read_master_df(xlsx)
        total_series, colname = _find_total_series(df)
        if total_series is not None:
            return float(total_series.sum()), f"from '{colname}'"
        total_series, used = _reconstruct_total_from_parts(df)
        if total_series is not None:
            return float(total_series.sum()), f"from parts: {', '.join(used)}"
        return None, None

    total_sum, how = await asyncio.to_thread(_calc)
    if total_sum is None:
        try:
            cols = list(_read_master_df(xlsx).columns)
            await callback.message.answer(
                "Could not find a total-like column or reconstruct it from parts.\n"
                f"Columns I see: {cols}\n"
                "Tip: if you can include a 'total_rate_usd' column, I'll use it automatically."
            )
        except Exception:
            await callback.message.answer("Could not find a total-like column or reconstruct it from parts.")
    else:
        await callback.message.answer(
            f"💰 Total of all saved loads: **${total_sum:,.2f}** ({how})",
            parse_mode="Markdown",
        )
    await callback.answer()

# ------------------------------ Maintenance ------------------------------

@dp.message(Command("purge_archives"))
async def purge_archives_cmd(message: types.Message):
    deleted = 0
    freed = 0
    for p in REPORTS_DIR.glob("monthly_report_*.xlsx"):
        try:
            freed += p.stat().st_size
            p.unlink(); deleted += 1
        except Exception as e:
            logger.warning("Failed to delete %s: %s", p, e)
    await message.answer(f"🧹 Deleted {deleted} archived report(s), freed {freed/1_048_576:.2f} MB.")

def purge_archives_older_than(days: int = 7):
    cutoff = time.time() - days * 86400
    for p in REPORTS_DIR.glob("monthly_report_*.xlsx"):
        try:
            if p.stat().st_mtime < cutoff: p.unlink()
        except Exception as e:
            logger.warning("Purge failed for %s: %s", p, e)

# optional: run once at startup
purge_archives_older_than(7)

# ================================== Main ==================================

async def main():
    deleted, freed = purge_temp_files()
    if deleted:
        logger.info("Startup purge: deleted %s file(s), freed %.2f MB", deleted, freed/1_048_576)
    await dp.start_polling(bot)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Exiting...")