import socketio
from main import appRecognition
from waitress import serve
import socket

sockIO = socketio.Server()
appServer = socketio.WSGIApp(sockIO, appRecognition)
hostname = socket.gethostname()
IPAddr = socket.gethostbyname(hostname)

if __name__ == '__main__':
    serve(appRecognition, host=IPAddr, port=8080, url_scheme='http', threads=4, log_untrusted_proxy_headers=True)