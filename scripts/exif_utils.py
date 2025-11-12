# Handles extraction of timestamp metadata from image EXIF data

from PIL import ExifTags
import datetime

def extract_exif_metadata(image):
    try:
        exif_data = image._getexif()
        if not exif_data:
            return None, None

        exif = {
            ExifTags.TAGS.get(tag, tag): value
            for tag, value in exif_data.items()
            if tag in ExifTags.TAGS
        }

        # Date/time
        dt_value = exif.get("DateTimeOriginal", None)
        if dt_value:
            dt = datetime.datetime.strptime(dt_value, "%Y:%m:%d %H:%M:%S")
            capture_date, capture_time = dt.date(), dt.time()
        else:
            capture_date, capture_time = None, None

        return capture_date, capture_time, 

    except Exception:
        return None, None
