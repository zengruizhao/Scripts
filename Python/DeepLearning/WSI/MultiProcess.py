# coding=utf-8
import multiprocessing
import time
ret = [0]

def func(ret):
    # print "*msg: ", msg
    # time.sleep(1)
    # print "*end: ", msg
    return ret+1


if __name__ == "__main__":
    # 维持执行的进程总数为processes，当一个进程执行完毕后会添加新的进程进去
    pool = multiprocessing.Pool(processes=5)
    results = []
    for i in range(10):
        for j in range(2):
            ret = pool.map(func, (ret[0],))        # 异步开启进程, 非阻塞型, 能够向池中添加进程而不等待其执行完毕就能再次执行循环
            print ret
    # print "--" * 10
    pool.close()   # 关闭pool, 则不会有新的进程添加进去
    pool.join()    # 必须在join之前close, 然后join等待pool中所有的线程执行完毕
    print "All process done."
    print ret