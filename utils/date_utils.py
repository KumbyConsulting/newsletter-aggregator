from datetime import datetime, timezone, timedelta
import re
import calendar
from email.utils import parsedate_to_datetime

def normalize_datetime(date_input):
    """
    Parse and normalize a date input to a UTC, offset-naive datetime object.
    Returns None if parsing fails.
    """
    if not date_input:
        return None
    if isinstance(date_input, datetime):
        if date_input.tzinfo is not None:
            return date_input.astimezone(timezone.utc).replace(tzinfo=None)
        return date_input
    if isinstance(date_input, (int, float)):
        try:
            return datetime.utcfromtimestamp(float(date_input))
        except Exception:
            return None
    if isinstance(date_input, str):
        s = date_input.strip()
        # Try ISO
        try:
            dt = datetime.fromisoformat(s.replace('Z', '+00:00'))
            return dt.astimezone(timezone.utc).replace(tzinfo=None)
        except Exception:
            pass
        # Try RFC
        try:
            dt = parsedate_to_datetime(s)
            return dt.astimezone(timezone.utc).replace(tzinfo=None)
        except Exception:
            pass
        # Try custom formats
        formats = [
            '%Y-%m-%d', '%Y/%m/%d', '%d-%m-%Y', '%d/%m/%Y', '%b %d, %Y', '%B %d, %Y',
            '%Y-%m-%d %H:%M:%S', '%Y/%m/%d %H:%M:%S', '%a, %d %b %Y %H:%M:%S',
            '%Y-%m-%dT%H:%M:%S', '%m/%d/%Y %H:%M', '%a, %m/%d/%Y - %H:%M'
        ]
        for fmt in formats:
            try:
                dt = datetime.strptime(s[:len(fmt)], fmt)
                return dt
            except Exception:
                continue
        # Handle problematic Drupal format: 'Tue, 03/04/2025 - 20:34'
        drupal_match = re.search(r'(\w+), (\d{2})/(\d{2})/(\d{4}) - (\d{2}):(\d{2})', s)
        if drupal_match:
            _, month, day, year, hour, minute = drupal_match.groups()
            try:
                dt = datetime(int(year), int(month), int(day), int(hour), int(minute))
                return dt
            except ValueError:
                if int(month) > 12 and int(day) <= 12:
                    dt = datetime(int(year), int(day), int(month), int(hour), int(minute))
                    return dt
    return None 