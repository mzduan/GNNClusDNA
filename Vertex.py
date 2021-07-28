class Veretx(object):
    def __init__(self, name):
        self.__name = name
        self.__links = set()

    @property
    def name(self):
        return self.__name

    @property
    def links(self):
        return set(self.__links)

    def add_link(self, other,score):
        self.__links.add(other)
        other.__links.add(self)


def connected_components_constraint(nodes,score_dict,th):
    result = []
    nodes = set(nodes)
    while nodes:
        n = nodes.pop() #选择一个元素作为开始
        group = {n}
        queue = [n]
        # valid = True
        while queue:
            n = queue.pop(0)
            # if th is not None:
            neighbors = {l for l in n.links if score_dict[tuple(sorted([n.name, l.name]))] >= th}
            # else:
            #     neighbors = n.links
            neighbors.difference_update(group) #求差
            nodes.difference_update(neighbors) #求差
            group.update(neighbors)  #并集
            queue.extend(neighbors)
            # if len(group) > max_sz or len(remain.intersection(neighbors)) > 0:
            #     valid = False
            #     remain.update(group)
            #     break
        # if valid:

        result.append(group)
    # return result, remain
    return result