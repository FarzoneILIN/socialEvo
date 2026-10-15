"""Optional read-only allowance snapshots for the historical provider."""
import json
import math
import os
import urllib.error
import urllib.parse
import urllib.request
from utils.client import NoRedirect
from utils.io import stamp as utc_now


def allowance_snapshot(*, opener=None):
    result = {'recorded_utc': utc_now(), 'status': 'unavailable',
              'scope': 'token allowance; not verified whole account balance or request charges',
              'quota_per_credit_reference': 500000}
    key = os.environ.get('SOCIALEVO_API_KEY', '')
    base = urllib.parse.urlsplit(os.environ.get('SOCIALEVO_BASE_URL', ''))
    if not key or base.scheme != 'https' or base.hostname != 'yunwu.ai' or base.port not in (None, 443):
        result['reason'] = 'unsupported_or_unconfigured_provider'
        return result
    if base.username or base.password or base.query or base.fragment:
        result['reason'] = 'invalid_provider_url'
        return result
    req = urllib.request.Request('https://yunwu.ai/api/usage/token/',
                                 headers={'Authorization': 'Bearer ' + key})
    try:
        opener = opener or urllib.request.build_opener(NoRedirect())
        with opener.open(req, timeout=15) as response:
            raw = response.read(65537)
        if len(raw) > 65536:
            raise ValueError('oversized_metadata')
        payload = json.loads(raw)
        data = payload.get('data', payload)
        if not isinstance(data, dict):
            raise ValueError('invalid_metadata')
        safe = {}
        for field in ('total_used', 'total_available', 'total_granted'):
            value = data.get(field)
            if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
                safe[field] = value
        if isinstance(data.get('unlimited_quota'), bool):
            safe['unlimited_quota'] = data['unlimited_quota']
        if not safe:
            raise ValueError('no_whitelisted_metadata')
        result.update(status='available', allowance=safe)
        if 'total_available' in safe:
            result['available_credits_reference'] = safe['total_available'] / 500000
    except urllib.error.HTTPError as exc:
        result['reason'] = 'metadata_http_' + str(exc.code)
    except Exception:
        # Neither provider body nor exception string is saved.
        result['reason'] = 'metadata_unavailable'
    return result
