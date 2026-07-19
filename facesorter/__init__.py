# Enable HEIC/HEIF decoding for Pillow when pillow-heif is installed
# (iPhone photos). Safe no-op otherwise.
try:
    from pillow_heif import register_heif_opener
    register_heif_opener()
    HEIC_SUPPORTED = True
except ImportError:
    HEIC_SUPPORTED = False
