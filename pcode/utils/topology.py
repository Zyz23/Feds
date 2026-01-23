# -*- coding: utf-8 -*-
import functools


def configure_gpu(world_conf):
    # the logic of world_conf follows "a,b,c,d,e" where:
    # the block range from 'a' to 'b' with interval 'c' (and each integer will repeat for 'd' time);
    # the block will be repeated for 'e' times.
  # stop = 0, interval = 1, local_repeat = 1, block_repeat = 100
    """
    configure_gpu 的 Docstring
    
    :param world_conf: a (start): 起始 GPU 编号。
    b (stop): 结束 GPU 编号。
    c (interval): 编号间隔（步长）。
    d (local_repeat): 每个 GPU 编号连续重复的次数。例如 local_repeat=2，序列会变成 [0,0, 1,1...]。
    e (block_repeat): 整个生成的编号块重复的总次数。
    """    
    start, stop, interval, local_repeat, block_repeat = [
        int(x) for x in world_conf.split(",")
    ]
    _block = [
        [x] * local_repeat for x in range(start, stop + 1, interval)
    ] * block_repeat
    world_list = functools.reduce(lambda a, b: a + b, _block)
    return world_list


class PhysicalLayout(object):
    def __init__(self, n_participated, world, world_conf, on_cuda):
        self.n_participated = n_participated 
        self._world = self.configure_world(world, world_conf) 
        self._on_cuda = on_cuda
        self.rank = -1

    def configure_world(self, world, world_conf):
        if world is not None:
            world_list = world.split(",")
            assert self.n_participated <= len(world_list)
        elif world_conf is not None:
            # the logic of world_conf follows "a,b,c,d,e" where:
            # the block range from 'a' to 'b' with interval 'c' (and each integer will repeat for 'd' time);
            # the block will be repeated for 'e' times.
            return configure_gpu(world_conf)
        else:
            raise RuntimeError(
                "you should at least make sure world or world_conf is not None."
            )
        return [int(l) for l in world_list]

    @property
    def primary_device(self):
        return self.devices[0]

    @property
    def devices(self):
        return self.world

    @property
    def on_cuda(self):
        return self._on_cuda

    @property
    def ranks(self):
        return list(range(1 + self.n_participated))

    @property
    def world(self):
        return self._world

    def get_device(self, rank):
        return self.devices[rank]

    def change_n_participated(self, n_participated):
        self.n_participated = n_participated

# 计算网络拓扑结构
def define_graph_topology(world, world_conf, n_participated, on_cuda): # world参数为指定哪个进程使用哪个GPU，这里并未指定，world_conf为指定GPU序列，这里为'0,0,1,1,100'分别为start，stop，intrval，count，即100个0
    """
    define_graph_topology 的 Docstring
    
    :param world: 网络节点信息
    :param world_conf: 配置字符串 "0,0,1,1,100" 代表服务器数量 机器数量 和总客户端数量
    :param n_participated: 参与的客户端数量
    :param on_cuda: 说明
    """
    # 调用逻辑  返回一个对象
    return PhysicalLayout(
        n_participated=n_participated,
        world=world,
        world_conf=world_conf,
        on_cuda=on_cuda,
    )
