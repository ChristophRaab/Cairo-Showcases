import os
import tornado.ioloop 
from tornado.ioloop import IOLoop
import tornado.web

root = os.path.dirname(__file__)



if __name__ == '__main__':
    application = tornado.web.Application([
        (r"/(.*)", tornado.web.StaticFileHandler, {"path": root, "default_filename": "index.html"})
    ])
    tornado.log.enable_pretty_logging()
    sockets = tornado.netutil.bind_sockets(80)
    tornado.process.fork_processes(0)
    server = tornado.httpserver.HTTPServer(application)
    server.add_sockets(sockets)
    IOLoop.current().start()