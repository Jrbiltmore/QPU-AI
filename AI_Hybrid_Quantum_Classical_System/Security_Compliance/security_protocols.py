
# /Security_Compliance/security_protocols.py

import hashlib
import hmac
import base64

class SecurityProtocols:
    def __init__(self, secret_key):
        self.secret_key = secret_key.encode()

    def generate_hash(self, message):
        # Generate a SHA-256 hash of the message
        return hashlib.sha256(message.encode()).hexdigest()

    def generate_hmac(self, message):
        # Generate an HMAC using SHA-256
        return hmac.new(self.secret_key, message.encode(), hashlib.sha256).hexdigest()

    def encode_base64(self, message):
        # Base64 encode the message
        return base64.b64encode(message.encode()).decode()

    def decode_base64(self, encoded_message):
        # Base64 decode the message
        return base64.b64decode(encoded_message.encode()).decode()

# Example usage
if __name__ == '__main__':
    security = SecurityProtocols('your-secret-key')
    message = 'Hello, secure world!'
    print("SHA-256:", security.generate_hash(message))
    print("HMAC:", security.generate_hmac(message))
    encoded = security.encode_base64(message)
    print("Encoded:", encoded)
    print("Decoded:", security.decode_base64(encoded))
