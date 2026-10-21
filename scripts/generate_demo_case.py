"""Synthetic Forensic Case Generator for Drishti-Sutra.

Generates realistic files for Case_DS_001 with deliberate, calibrated forensic discrepancies:
1. synthetic_pan_card.png: KYC card for 'Aarav Sharma', DOB: 1994-05-12, PAN: ABCPS1234D.
2. synthetic_id_card.png: Aadhaar/ID card for 'Aarav Sharma', printed DOB: 1994-08-12 (mismatch),
   with a real QR code encoding JSON with dob: 1994-05-12 (QR vs printed text mismatch),
   and localized lower-quality compression over the DOB block to trigger Error Level Analysis (ELA).
3. whatsapp_screenshot.png: 4 messages between Aarav and Vikram with an out-of-order timestamp (10:15 AM after 10:16 AM)
   and a transfer of Rs 50,000.
4. whatsapp_chat_export.txt: WhatsApp text export where the amount is modified to Rs 5,000 and 1 message is missing.
5. system_events.log: Device auth, banking transaction of Rs 50,000 at 10:15:15 AM, and WhatsApp database modification at 10:18:22 AM.
6. manifest.json: Ground-truth discrepancy catalog with dynamically calculated SHA-256 hashes.
"""

import hashlib
import json
import os
from io import BytesIO
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from PIL.PngImagePlugin import PngInfo
import qrcode


def compute_sha256(file_path: Path) -> str:
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def get_font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    """Attempt to load a clean TTF font; fallback to default if not found."""
    candidates = [
        "C:\\Windows\\Fonts\\arialbd.ttf" if bold else "C:\\Windows\\Fonts\\arial.ttf",
        "C:\\Windows\\Fonts\\calibrib.ttf" if bold else "C:\\Windows\\Fonts\\calibri.ttf",
        "C:\\Windows\\Fonts\\segoeui.ttf",
    ]
    for c in candidates:
        if os.path.exists(c):
            try:
                return ImageFont.truetype(c, size)
            except Exception:
                pass
    return ImageFont.load_default()


def generate_pan_card(output_path: Path) -> None:
    """Generate high-contrast synthetic PAN Card image."""
    width, height = 900, 560
    img = Image.new("RGB", (width, height), color=(240, 245, 255))
    draw = ImageDraw.Draw(img)

    # Card border and header
    draw.rectangle([10, 10, width - 10, height - 10], outline=(40, 70, 140), width=4)
    draw.rectangle([15, 15, width - 15, 90], fill=(30, 65, 130))

    header_font = get_font(24, bold=True)
    sub_font = get_font(16, bold=False)
    label_font = get_font(16, bold=False)
    val_font = get_font(20, bold=True)
    pan_font = get_font(26, bold=True)

    draw.text((30, 25), "INCOME TAX DEPARTMENT", fill=(255, 255, 255), font=header_font)
    draw.text((30, 58), "GOVERNMENT OF INDIA", fill=(220, 230, 255), font=sub_font)

    # Photo Box Placeholder
    photo_box = [45, 120, 225, 340]
    draw.rectangle(photo_box, outline=(120, 120, 120), fill=(215, 225, 235), width=2)
    draw.text((65, 220), "[ PHOTO ]", fill=(90, 100, 110), font=label_font)

    # Signature Box
    sig_box = [45, 380, 225, 460]
    draw.rectangle(sig_box, outline=(180, 180, 180), fill=(255, 255, 255), width=1)
    draw.text((60, 410), "Aarav Sharma", fill=(20, 30, 80), font=get_font(22, bold=False))

    # Text fields
    x_offset = 270
    draw.text((x_offset, 120), "Name / Name:", fill=(100, 100, 100), font=label_font)
    draw.text((x_offset, 145), "Aarav Sharma", fill=(20, 20, 20), font=val_font)

    draw.text((x_offset, 195), "Father's Name:", fill=(100, 100, 100), font=label_font)
    draw.text((x_offset, 220), "Ramesh Sharma", fill=(20, 20, 20), font=val_font)

    draw.text((x_offset, 270), "Date of Birth / DOB:", fill=(100, 100, 100), font=label_font)
    draw.text((x_offset, 295), "1994-05-12", fill=(20, 20, 20), font=val_font)

    draw.text((x_offset, 350), "Permanent Account Number:", fill=(100, 100, 100), font=label_font)
    draw.text((x_offset, 380), "ABCPS1234D", fill=(180, 20, 20), font=pan_font)

    draw.line([270, 440, width - 40, 440], fill=(200, 210, 225), width=2)
    draw.text((270, 460), "Card Verification Status: ISSUED", fill=(80, 100, 120), font=sub_font)

    # Embed layout metadata in PNG text chunk for deterministic fallback parsing
    doc_text = (
        "INCOME TAX DEPARTMENT\nGOVERNMENT OF INDIA\n"
        "Name: Aarav Sharma\nFather's Name: Ramesh Sharma\n"
        "DOB: 1994-05-12\nPermanent Account Number: ABCPS1234D\nStatus: ISSUED"
    )
    pnginfo = PngInfo()
    pnginfo.add_text("raw_text", doc_text)
    pnginfo.add_text("doc_type", "PAN_CARD")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(output_path, "PNG", pnginfo=pnginfo)
    print(f"Generated PAN Card: {output_path}")


def generate_id_card_with_tampering(output_path: Path) -> None:
    """Generate synthetic ID card with QR contradiction and localized ELA re-compression patch."""
    width, height = 900, 560
    base_img = Image.new("RGB", (width, height), color=(255, 253, 245))
    draw = ImageDraw.Draw(base_img)

    # Tricolor header bar
    draw.rectangle([10, 10, width - 10, height - 10], outline=(180, 140, 40), width=3)
    draw.rectangle([15, 15, width - 15, 40], fill=(255, 153, 51))
    draw.rectangle([15, 40, width - 15, 65], fill=(255, 255, 255))
    draw.rectangle([15, 65, width - 15, 90], fill=(19, 136, 8))

    header_font = get_font(22, bold=True)
    label_font = get_font(16, bold=False)
    val_font = get_font(20, bold=True)
    id_font = get_font(26, bold=True)

    draw.text((30, 38), "UNIQUE IDENTIFICATION AUTHORITY OF INDIA", fill=(20, 20, 20), font=header_font)

    # Photo Box
    photo_box = [45, 120, 225, 340]
    draw.rectangle(photo_box, outline=(120, 120, 120), fill=(235, 235, 225), width=2)
    draw.text((70, 220), "[ PHOTO ]", fill=(90, 90, 90), font=label_font)

    # Text fields
    x_offset = 260
    draw.text((x_offset, 120), "Name:", fill=(100, 100, 100), font=label_font)
    draw.text((x_offset, 145), "Aarav Sharma", fill=(20, 20, 20), font=val_font)

    draw.text((x_offset, 195), "DOB / Date of Birth:", fill=(100, 100, 100), font=label_font)
    # PRINTED TEXT: 1994-08-12 (Mismatched with PAN 1994-05-12 and QR 1994-05-12)
    dob_box = (x_offset - 5, 218, x_offset + 250, 252)
    draw.text((x_offset, 220), "1994-08-12", fill=(10, 10, 10), font=val_font)

    draw.text((x_offset, 275), "Gender / Sex:", fill=(100, 100, 100), font=label_font)
    draw.text((x_offset, 300), "Male", fill=(20, 20, 20), font=val_font)

    # Aadhaar Number
    draw.text((x_offset, 370), "9876 5432 1098", fill=(180, 40, 20), font=id_font)

    # Real QR code generation
    # QR Payload contains {"name": "Aarav Sharma", "dob": "1994-05-12", "id": "9876 5432 1098"}
    qr_payload = json.dumps({
        "name": "Aarav Sharma",
        "dob": "1994-05-12",
        "id": "9876 5432 1098"
    })
    qr = qrcode.QRCode(
        version=1,
        error_correction=qrcode.constants.ERROR_CORRECT_M,
        box_size=5,
        border=2,
    )
    qr.add_data(qr_payload)
    qr.make(fit=True)
    qr_img = qr.make_image(fill_color="black", back_color="white").convert("RGB")
    qr_resized = qr_img.resize((190, 190))
    base_img.paste(qr_resized, (660, 130))

    # --- Introduce localized ELA tampering artifact on the DOB region ---
    # Crop the DOB text region, compress it heavily as JPEG (quality 20), and paste back.
    dob_crop = base_img.crop(dob_box)
    buffer = BytesIO()
    dob_crop.save(buffer, format="JPEG", quality=20)
    buffer.seek(0)
    low_q_crop = Image.open(buffer).convert("RGB")
    base_img.paste(low_q_crop, (dob_box[0], dob_box[1]))

    # Embed layout metadata in PNG text chunk for deterministic fallback parsing
    doc_text = (
        "UNIQUE IDENTIFICATION AUTHORITY OF INDIA\n"
        "Name: Aarav Sharma\n"
        "DOB: 1994-08-12\n"
        "Gender: Male\n"
        "Aadhaar Number: 9876 5432 1098"
    )
    pnginfo = PngInfo()
    pnginfo.add_text("raw_text", doc_text)
    pnginfo.add_text("doc_type", "ID_CARD")
    pnginfo.add_text("tampered_box", json.dumps({"x": dob_box[0], "y": dob_box[1], "w": dob_box[2] - dob_box[0], "h": dob_box[3] - dob_box[1]}))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    base_img.save(output_path, "PNG", pnginfo=pnginfo)
    print(f"Generated Tampered ID Card: {output_path}")


def generate_whatsapp_screenshot(output_path: Path) -> None:
    """Generate synthetic WhatsApp chat UI with out-of-order timestamp."""
    width, height = 750, 950
    img = Image.new("RGB", (width, height), color=(236, 229, 221))  # WhatsApp wallpaper color
    draw = ImageDraw.Draw(img)

    # WhatsApp App Header
    header_h = 100
    draw.rectangle([0, 0, width, header_h], fill=(7, 94, 84))  # WhatsApp green
    title_font = get_font(24, bold=True)
    sub_font = get_font(15, bold=False)
    draw.text((80, 25), "Vikram Patel", fill=(255, 255, 255), font=title_font)
    draw.text((80, 58), "online", fill=(200, 240, 220), font=sub_font)

    # Date chip in center
    date_font = get_font(13, bold=False)
    draw.rounded_rectangle([280, 120, 470, 150], radius=8, fill=(255, 255, 255), outline=(220, 220, 220))
    draw.text((315, 128), "October 20, 2026", fill=(100, 100, 100), font=date_font)

    msg_font = get_font(18, bold=False)
    time_font = get_font(13, bold=False)

    bubbles_meta = []

    def draw_bubble(y: int, sender: str, text: str, time_str: str, incoming: bool) -> int:
        bbox = draw.textbbox((0, 0), text, font=msg_font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]

        bubble_w = max(text_w + 100, 260)
        bubble_h = text_h + 38

        if incoming:
            x1 = 30
            x2 = x1 + bubble_w
            bg_color = (255, 255, 255)
        else:
            x2 = width - 30
            x1 = x2 - bubble_w
            bg_color = (220, 248, 198)

        draw.rounded_rectangle([x1, y, x2, y + bubble_h], radius=12, fill=bg_color, outline=(210, 210, 210), width=1)
        draw.text((x1 + 16, y + 10), text, fill=(15, 15, 15), font=msg_font)
        draw.text((x2 - 75, y + bubble_h - 22), time_str, fill=(130, 130, 130), font=time_font)

        bubbles_meta.append({
            "sender": sender,
            "text": text,
            "timestamp": time_str,
            "box": {"x": x1, "y": y, "w": bubble_w, "h": bubble_h},
            "incoming": incoming
        })
        return y + bubble_h + 24

    # Message 1 (Incoming): 10:14 AM
    curr_y = draw_bubble(180, "Vikram Patel", "Hey Aarav, can you send the payment for the project?", "10:14 AM", incoming=True)
    # Message 2 (Outgoing): 10:16 AM
    curr_y = draw_bubble(curr_y, "Aarav Sharma", "Sure, doing it right now.", "10:16 AM", incoming=False)
    # Message 3 (Outgoing): 10:15 AM (OUT-OF-ORDER TIMESTAMP! 10:15 AM appears visually after 10:16 AM)
    curr_y = draw_bubble(curr_y, "Aarav Sharma", "I transferred \u20b950,000 at 10:15 AM", "10:15 AM", incoming=False)
    # Message 4 (Incoming): 10:20 AM
    curr_y = draw_bubble(curr_y, "Vikram Patel", "Thanks, confirming received.", "10:20 AM", incoming=True)

    pnginfo = PngInfo()
    pnginfo.add_text("bubbles_data", json.dumps(bubbles_meta))
    pnginfo.add_text("platform", "WHATSAPP")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(output_path, "PNG", pnginfo=pnginfo)
    print(f"Generated WhatsApp Screenshot: {output_path}")


def generate_whatsapp_chat_export(output_path: Path) -> None:
    """Generate companion WhatsApp text export with amount and message alterations."""
    content = (
        "[20/10/2026, 10:14:10 AM] Vikram Patel: Hey Aarav, can you send the payment for the project?\n"
        "[20/10/2026, 10:15:22 AM] Aarav Sharma: I transferred \u20b95,000 at 10:15 AM\n"
        "[20/10/2026, 10:20:05 AM] Vikram Patel: Thanks, confirming received.\n"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"Generated WhatsApp Chat Export: {output_path}")


def generate_system_events_log(output_path: Path) -> None:
    """Generate structured system and app event log."""
    lines = [
        "2026-10-20 10:10:02 INFO  [auth] User 'aarav.sharma' authenticated via Biometrics (Device: ANDROID-SM-G998B)",
        "2026-10-20 10:12:15 INFO  [app] Opened WhatsApp Messenger v2.24.18 (PID: 14820)",
        "2026-10-20 10:14:48 INFO  [app] Opened HDFC Mobile Banking v11.4 (PID: 15302)",
        "2026-10-20 10:15:15 INFO  [transaction] IMPS Transfer initiated: TxnID=TXN948201948 Amount=50000.00 Payee='Vikram Patel' Status=SUCCESS",
        "2026-10-20 10:18:22 WARN  [filesystem] File modified: /sdcard/WhatsApp/Databases/msgstore.db (SHA256 mismatch with cloud backup)",
        "2026-10-20 10:22:04 INFO  [network] Connection closed to api.hdfcbank.com:443 (Bytes sent: 4820, received: 1204)",
        "2026-10-20 10:25:00 INFO  [system] Screen lock engaged by user",
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Generated System Events Log: {output_path}")


def generate_manifest(case_dir: Path) -> None:
    """Compile ground-truth forensic discrepancies and compute dynamic SHA-256 hashes."""
    files_info = {}
    for item in case_dir.glob("*"):
        if item.is_file() and item.name not in ["manifest.json", "test_meta.png"]:
            files_info[item.name] = {
                "sha256": compute_sha256(item),
                "size_bytes": item.stat().st_size,
            }

    manifest = {
        "case_id": "Case_DS_001",
        "title": "Investigation of Financial Dispute & Evidence Authenticity",
        "primary_entity": "Aarav Sharma",
        "counterparty": "Vikram Patel",
        "generated_timestamp": "2026-10-20T10:30:00Z",
        "files": files_info,
        "ground_truth_anomalies": [
            {
                "type": "DOB_CROSS_DOCUMENT_MISMATCH",
                "sources": ["synthetic_pan_card.png", "synthetic_id_card.png"],
                "description": "PAN card shows DOB 1994-05-12, whereas National ID card text shows 1994-08-12.",
                "severity": "HIGH",
            },
            {
                "type": "QR_VS_PRINTED_MISMATCH",
                "sources": ["synthetic_id_card.png"],
                "description": "The QR code on synthetic_id_card.png encodes DOB 1994-05-12, contradicting printed text 1994-08-12.",
                "severity": "CRITICAL",
            },
            {
                "type": "ELA_LOCALIZED_TAMPERING",
                "sources": ["synthetic_id_card.png"],
                "description": "Error Level Analysis reveals high compression variance on the DOB region of the ID card.",
                "severity": "HIGH",
            },
            {
                "type": "CHRONOLOGICAL_INVERSION",
                "sources": ["whatsapp_screenshot.png"],
                "description": "Message at 10:15 AM appears visually after a message at 10:16 AM in the screenshot.",
                "severity": "HIGH",
            },
            {
                "type": "AMOUNT_CONTRADICTION",
                "sources": ["whatsapp_screenshot.png", "whatsapp_chat_export.txt", "system_events.log"],
                "description": "Screenshot claims Rs 50,000 transferred; chat export text claims Rs 5,000; bank transaction log confirms Rs 50,000.00 transferred.",
                "severity": "CRITICAL",
            },
            {
                "type": "MISSING_MESSAGE_IN_EXPORT",
                "sources": ["whatsapp_screenshot.png", "whatsapp_chat_export.txt"],
                "description": "Message 'Sure, doing it right now.' present in screenshot is absent in the exported chat file.",
                "severity": "MEDIUM",
            },
            {
                "type": "UNAUTHORIZED_DB_MODIFICATION",
                "sources": ["system_events.log"],
                "description": "WhatsApp msgstore.db modified at 10:18:22 AM following bank transaction, indicating post-facto tampering.",
                "severity": "HIGH",
            },
        ],
    }

    manifest_path = case_dir / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"Generated Ground-Truth Manifest: {manifest_path}")


def main() -> None:
    case_dir = Path("sample_data/Case_DS_001")
    case_dir.mkdir(parents=True, exist_ok=True)

    # Clean up any test artifact
    test_file = case_dir / "test_meta.png"
    if test_file.exists():
        test_file.unlink()

    generate_pan_card(case_dir / "synthetic_pan_card.png")
    generate_id_card_with_tampering(case_dir / "synthetic_id_card.png")
    generate_whatsapp_screenshot(case_dir / "whatsapp_screenshot.png")
    generate_whatsapp_chat_export(case_dir / "whatsapp_chat_export.txt")
    generate_system_events_log(case_dir / "system_events.log")
    generate_manifest(case_dir)
    print("\nCase_DS_001 demo generation complete!")


if __name__ == "__main__":
    main()
