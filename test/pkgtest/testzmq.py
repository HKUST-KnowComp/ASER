import zmq
import zmq.decorators as zmqd

@zmqd.context()
@zmqd.socket(zmq.PULL)
def fn(ctx, skt):
    print(type(ctx))
    print(type(skt))
    pass

fn()