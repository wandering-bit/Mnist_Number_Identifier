import http.server
import base64
from Crypto.PublicKey import RSA
from Crypto.Cipher import PKCS1_v1_5

class CustomHandler(http.server.BaseHTTPRequestHandler):
    def __init__(self):
        self.publicKey:str
        self.body:str
        self.contentLength:int
        self.pemKey:RSA.RsaKey

    def do_POST(self):
        self.GetBodyLength()
        self.GetBody()
        self.GetHeaders()
        if self.publicKey=="" or self.contentLength==0:
            self.Respond()
        self.ParsePemfromPubkey(self.publicKey)
        self.end_headers()

    def GetBodyLength(self):
        strContentLength = self.headers.get("Content-Length")
        if strContentLength is None:
            self.contentLength = 0
            return
        self.contentLength = int(strContentLength)

    def Respond(self):
        if self.publicKey == "":
            self.send_response(400)
            self.send_header("Content-type","application/json")
            self.end_headers()
            self.wfile.write(b"Missing Public Key 4 in Headers")
            return
        if self.contentLength == 0:
            self.send_response(400)
            self.send_header("Content-type","application/json")
            self.end_headers()
            self.wfile.write(b"Empty Response Body")
            return
        self.send_response(200)
        self.send_header("Content-type","application/json")
        self.end_headers()
        self.wfile.write(b"Success")
        return
    
    def GetBody(self):
        byteBody = self.rfile.read(self.contentLength)
        self.body = byteBody.decode("utf-8")

    def GetHeaders(self):
        pubKey = self.headers.get("Pub4")
        if pubKey is None:
            self.publicKey = ""
            return
        self.publicKey = pubKey

    def ParsePemfromPubkey(self,pubKey:str)-> RSA.RsaKey:
        chunks = [pubKey[i:i+64] for i in range(0, len(pubKey), 64)]
        pem_str = "-----BEGIN PUBLIC KEY-----\n" + "\n".join(chunks) + "\n-----END PUBLIC KEY-----"
        try:
            return RSA.import_key(pem_str)
        except Exception as e:
            raise ValueError(f"Invalid PEM public key: {e}")

server = http.server.HTTPServer(("127.0.0.1", 8080), CustomHandler)

print("starting http server")
server.serve_forever()
