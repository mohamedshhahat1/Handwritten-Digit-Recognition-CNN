"""
OCR Synthetic Dataset Generator
=================================

Generates training images by rendering text in various handwriting-style fonts.
This avoids the need for large pre-existing handwriting datasets.

Strategy:
    1. Pick random words from English/Arabic word lists
    2. Render them as images using PIL with handwriting-like fonts
    3. Apply augmentation (rotation, noise, blur, thickness variation)
    4. Resize to fixed height (32px), variable width

The generated images look like handwritten text and cover the full
character set (English + Arabic + digits).
"""

import os
import random
import string
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import torch
from torch.utils.data import Dataset

from model.ocr_utils import OCRCharset, ARABIC_LETTERS


# Common English words for synthetic training data
ENGLISH_WORDS = [
    "hello", "world", "python", "machine", "learning", "deep", "neural",
    "network", "computer", "science", "data", "model", "train", "test",
    "image", "text", "word", "code", "function", "class", "object",
    "array", "list", "string", "number", "value", "input", "output",
    "file", "read", "write", "open", "close", "start", "stop", "run",
    "build", "make", "create", "delete", "update", "find", "search",
    "sort", "filter", "map", "reduce", "loop", "while", "for", "if",
    "true", "false", "null", "none", "return", "print", "import",
    "from", "class", "def", "self", "init", "main", "app", "web",
    "api", "server", "client", "database", "query", "table", "user",
    "name", "email", "password", "login", "sign", "page", "home",
    "about", "contact", "help", "error", "success", "warning", "info",
    "the", "is", "are", "was", "were", "will", "can", "have", "has",
    "good", "great", "best", "new", "old", "big", "small", "fast",
    "AI", "ML", "CNN", "RNN", "LSTM", "GPU", "CPU", "RAM", "API",
]

# Common Arabic words for synthetic training data
ARABIC_WORDS = [
    "مرحبا", "عالم", "برمجة", "تعلم", "ذكاء", "بيانات", "شبكة",
    "حاسوب", "علم", "نموذج", "تدريب", "اختبار", "صورة", "نص",
    "كلمة", "كود", "دالة", "كائن", "مصفوفة", "قائمة", "سلسلة",
    "رقم", "قيمة", "ادخال", "اخراج", "ملف", "قراءة", "كتابة",
    "فتح", "بدء", "انشاء", "حذف", "تحديث", "بحث", "ترتيب",
    "خوارزمية", "حلقة", "شرط", "صحيح", "خطأ", "عودة", "طباعة",
    "استيراد", "واجهة", "خادم", "عميل", "قاعدة", "استعلام",
    "مستخدم", "اسم", "بريد", "كلمة", "دخول", "صفحة", "رئيسية",
    "مساعدة", "خطأ", "نجاح", "تحذير", "معلومات", "جديد", "قديم",
    "كبير", "صغير", "سريع", "بطيء", "جيد", "ممتاز", "افضل",
]


class SyntheticOCRDataset(Dataset):
    """
    Generates synthetic text images for OCR training.

    Each sample is a rendered text image with its corresponding label.
    Text is randomly generated from word lists or random character sequences.

    Args:
        charset (OCRCharset): Character set for encoding labels.
        num_samples (int): Number of samples to generate per epoch.
        img_height (int): Output image height (default 32).
        max_text_len (int): Maximum text length in characters (default 20).
        languages (list): Which languages to include ['en', 'ar', 'both'].
    """

    def __init__(self, charset, num_samples=10000, img_height=32,
                 max_text_len=20, languages=None):
        self.charset = charset
        self.num_samples = num_samples
        self.img_height = img_height
        self.max_text_len = max_text_len
        self.languages = languages or ['en', 'ar']

        # Try to find system fonts
        self.fonts = self._find_fonts()

    def _find_fonts(self):
        """Find available fonts on the system."""
        font_paths = []

        # Common font directories
        font_dirs = [
            "/usr/share/fonts",
            "/usr/local/share/fonts",
            os.path.expanduser("~/.fonts"),
            "/System/Library/Fonts",  # macOS
        ]

        for font_dir in font_dirs:
            if os.path.exists(font_dir):
                for root, dirs, files in os.walk(font_dir):
                    for f in files:
                        if f.endswith(('.ttf', '.otf', '.TTF', '.OTF')):
                            font_paths.append(os.path.join(root, f))

        # If no fonts found, we'll use PIL default
        if not font_paths:
            font_paths = [None]

        return font_paths

    def _get_font(self, size=28):
        """Get a random font at the specified size."""
        font_path = random.choice(self.fonts)
        try:
            if font_path:
                return ImageFont.truetype(font_path, size)
            else:
                return ImageFont.load_default()
        except (OSError, IOError):
            return ImageFont.load_default()

    def _generate_text(self):
        """Generate random text for training."""
        choice = random.random()

        if choice < 0.3:
            # Single English word
            return random.choice(ENGLISH_WORDS)
        elif choice < 0.5:
            # English phrase (2-3 words)
            num_words = random.randint(2, 3)
            return ' '.join(random.choices(ENGLISH_WORDS, k=num_words))
        elif choice < 0.7:
            # Single Arabic word
            return random.choice(ARABIC_WORDS)
        elif choice < 0.85:
            # Random digits
            length = random.randint(1, 6)
            return ''.join(random.choices(string.digits, k=length))
        else:
            # Random character sequence
            chars = string.ascii_letters + string.digits
            length = random.randint(2, 8)
            return ''.join(random.choices(chars, k=length))

    def _render_text(self, text):
        """
        Render text as a grayscale image.

        Creates a white-on-black image (like writing on a blackboard)
        with the text rendered using a random font.

        Returns:
            PIL.Image: Grayscale image of the rendered text.
        """
        # Get font
        font_size = random.randint(24, 36)
        font = self._get_font(font_size)

        # Calculate text size
        dummy_img = Image.new('L', (1, 1))
        dummy_draw = ImageDraw.Draw(dummy_img)
        bbox = dummy_draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0] + 20  # padding
        text_height = bbox[3] - bbox[1] + 20

        # Create image (black background)
        img_width = max(text_width, 32)
        img_height = max(text_height, self.img_height + 10)
        img = Image.new('L', (img_width, img_height), color=0)
        draw = ImageDraw.Draw(img)

        # Draw text (white on black)
        x = random.randint(5, 15)
        y = (img_height - (bbox[3] - bbox[1])) // 2
        draw.text((x, y), text, fill=255, font=font)

        # Resize to fixed height, maintaining aspect ratio
        aspect = img_width / img_height
        new_width = max(int(self.img_height * aspect), 32)
        img = img.resize((new_width, self.img_height), Image.BILINEAR)

        return img

    def _augment(self, img):
        """Apply random augmentation to the image."""
        # Random rotation (slight)
        if random.random() < 0.3:
            angle = random.uniform(-5, 5)
            img = img.rotate(angle, fillcolor=0)

        # Random blur
        if random.random() < 0.2:
            img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5, 1.5)))

        # Random noise
        if random.random() < 0.3:
            arr = np.array(img).astype(np.float32)
            noise = np.random.normal(0, random.uniform(5, 15), arr.shape)
            arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
            img = Image.fromarray(arr)

        # Random erosion/dilation (thickness variation)
        if random.random() < 0.2:
            if random.random() < 0.5:
                img = img.filter(ImageFilter.MinFilter(3))  # thinner
            else:
                img = img.filter(ImageFilter.MaxFilter(3))  # thicker

        return img

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        """
        Generate a single training sample.

        Returns:
            tuple: (image_tensor, label_tensor, label_length)
                - image_tensor: (1, 32, W) normalized grayscale
                - label_tensor: encoded text as int tensor
                - label_length: number of characters in label
        """
        # Generate random text
        text = self._generate_text()

        # Truncate to max length
        text = text[:self.max_text_len]

        # Filter to only charset-supported characters
        text = ''.join(c for c in text if c in self.charset.char_to_idx)
        if not text:
            text = random.choice(ENGLISH_WORDS)[:self.max_text_len]

        # Render as image
        img = self._render_text(text)

        # Augment
        img = self._augment(img)

        # Convert to tensor and normalize
        img_tensor = torch.FloatTensor(np.array(img)).unsqueeze(0) / 255.0

        # Ensure minimum width (at least 32px)
        if img_tensor.size(2) < 32:
            padding = torch.zeros(1, self.img_height, 32 - img_tensor.size(2))
            img_tensor = torch.cat([img_tensor, padding], dim=2)

        # Encode label
        label = self.charset.encode(text)
        label_tensor = torch.IntTensor(label)

        return img_tensor, label_tensor, len(label)
