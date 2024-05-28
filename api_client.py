import hashlib
import hmac
import time
import requests
import uuid
from urllib.parse import urlencode



class BitstampClient:
    def __init__(self, api_key, api_secret):
        self.api_key = api_key
        self.api_secret = bytes(api_secret, "utf-8")
        self.base_url = "https://www.bitstamp.net/api/v2/"

    def _generate_signature(self, endpoint, payload_string, nonce, timestamp):
        message = (
            "BITSTAMP "
            + self.api_key
            + "POST"
            + "www.bitstamp.net/api/v2/"
            + endpoint
            + ""
            + "application/x-www-form-urlencoded"
            + nonce
            + timestamp
            + "v2"
            + payload_string
        )
        message = message.encode("utf-8")
        signature = hmac.new(self.api_secret, msg=message, digestmod=hashlib.sha256).hexdigest()
        return signature

    def _send_request(self, method, endpoint, payload=None, params=None):
        url = self.base_url + endpoint
        headers = {
            "X-Auth": "BITSTAMP " + self.api_key,
            "Content-Type": "application/x-www-form-urlencoded",
        }
        payload_string = urlencode(payload) if payload else ""
        if payload:
            nonce = str(uuid.uuid4())
            timestamp = str(int(round(time.time() * 1000)))
            signature = self._generate_signature(endpoint, payload_string, nonce, timestamp)
            headers["X-Auth-Signature"] = signature
            headers["X-Auth-Nonce"] = nonce
            headers["X-Auth-Timestamp"] = timestamp
            headers["X-Auth-Version"] = "v2"

        response = requests.request(
            method, url, headers=headers, data=payload_string, params=params
        )
        response.raise_for_status()
        return response.json()

    def get(self, endpoint, params=None):
        return self._send_request("GET", endpoint, params=params)

    def post(self, endpoint, payload):
        return self._send_request("POST", endpoint, payload)
