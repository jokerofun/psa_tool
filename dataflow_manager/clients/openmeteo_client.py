import openmeteo_requests

import requests_cache
from retry_requests import retry
openmeteo = None

def get_openmeteo_client():
    ## Create a singleton instance of the OpenMeteo client
    global openmeteo
    if openmeteo is None:
        cache_session = requests_cache.CachedSession('.cache', expire_after = 3600)
        retry_session = retry(cache_session, retries = 5, backoff_factor = 0.2)
        openmeteo = openmeteo_requests.Client(session = retry_session)
    return openmeteo


